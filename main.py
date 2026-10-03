"""
CGO 음악 렌더 서버 (Railway) v2 — 멜로디 중심 믹스·음량 보강 + VVIP AI 보컬
  GET  /             : 서버 깨우기·상태 확인
  POST /render       : (기존) 멜로디 한 줄 렌더  {bpm, notes:[{n,d}], instrument}
  POST /render_full  : (신규) 멜로디·베이스·코드·드럼 한 번에 렌더 → WAV 1개
  GET  /render_full  : 배포 확인용
  POST /vvip_generate: VVIP AI 보컬 곡 생성 (Suno via apiframe.ai)
"""
import glob
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np
import requests as http_requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

_CGO_BOOT_TS = time.time()   # cgo-467: 이 서버가 언제 올라왔는지

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
router = app


SR = 44100
MAX_TRACKS = 8
MAX_NOTES = 6000
MAX_SECONDS = 300
TAIL_SECONDS = 2.0


# ── 사운드폰트 찾기 (환경변수 CGO_SF2 > 앱 폴더의 .sf2 > 시스템 GM) ──
def _find_sf2() -> Optional[str]:
    env = os.environ.get("CGO_SF2")
    if env and os.path.isfile(env):
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    cands = sorted(glob.glob(os.path.join(here, "**", "*.sf2"), recursive=True),
                   key=lambda p: -os.path.getsize(p))
    cands += ["/usr/share/sounds/sf2/FluidR3_GM.sf2",
              "/usr/share/sounds/sf2/default-GM.sf2",
              "/usr/share/soundfonts/FluidR3_GM.sf2",
              "/usr/share/soundfonts/default.sf2"]
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


class Note(BaseModel):
    n: int = Field(..., ge=0, le=127)
    t: float = Field(..., ge=0)
    d: float = Field(..., gt=0)
    v: int = Field(100, ge=1, le=127)


class Track(BaseModel):
    name: str = ""
    instrument: int = Field(0, ge=0, le=127)
    drum: bool = False
    volume: int = Field(100, ge=0, le=127)
    notes: List[Note] = []


class FullReq(BaseModel):
    bpm: float = Field(..., ge=20, le=300)
    tracks: List[Track]


def _channels(tracks: List[Track]) -> List[int]:
    """드럼 → 9번 채널, 나머지는 9번을 건너뛰며 배정."""
    free = [c for c in range(16) if c != 9]
    out = []
    for tr in tracks:
        out.append(9 if tr.drum else free.pop(0))
    return out


def _events(req: FullReq, chans: List[int]):
    spb = 60.0 / req.bpm
    ev = []  # (sec, order, kind, ch, key, vel)  order: off(0) 가 on(1) 보다 먼저
    end = 0.0
    for tr, ch in zip(req.tracks, chans):
        for nt in tr.notes:
            on = nt.t * spb
            off = on + max(0.03, nt.d * spb - 0.01)
            ev.append((on, 1, "on", ch, nt.n, nt.v))
            ev.append((off, 0, "off", ch, nt.n, 0))
            end = max(end, off)
    ev.sort(key=lambda e: (e[0], e[1]))
    return ev, end


# ── 방법 A: pyfluidsynth 직접 렌더 (빠름, 사운드폰트 1회 로드 후 재사용) ──
_lock = threading.Lock()
_synth = {"fs": None, "sfid": None, "sf2": None}


def _get_synth(sf2):
    import fluidsynth  # pyfluidsynth
    if _synth["fs"] is None or _synth["sf2"] != sf2:
        fs = fluidsynth.Synth(samplerate=float(SR), gain=0.6)
        sfid = fs.sfload(sf2)
        _synth.update(fs=fs, sfid=sfid, sf2=sf2)
    return _synth["fs"], _synth["sfid"]


def _render_pyfs(req: FullReq, sf2: str) -> np.ndarray:
    chans = _channels(req.tracks)
    ev, end = _events(req, chans)
    with _lock:
        fs, sfid = _get_synth(sf2)
        # 이전 요청 잔향·설정 초기화
        for ch in range(16):
            fs.cc(ch, 123, 0)   # all notes off
            fs.cc(ch, 120, 0)   # all sound off
            fs.cc(ch, 121, 0)   # reset controllers
        fs.get_samples(int(SR * 0.05))
        for tr, ch in zip(req.tracks, chans):
            if tr.drum:
                fs.program_select(ch, sfid, 128, 0)
            else:
                fs.program_select(ch, sfid, 0, tr.instrument)
            fs.cc(ch, 7, tr.volume)
            fs.cc(ch, 91, 40)       # 리버브 살짝
        chunks, cur = [], 0
        for sec, _o, kind, ch, key, vel in ev:
            fr = int(round(sec * SR))
            if fr > cur:
                chunks.append(np.asarray(fs.get_samples(fr - cur), dtype=np.int16))
                cur = fr
            if kind == "on":
                fs.noteon(ch, key, vel)
            else:
                fs.noteoff(ch, key)
        chunks.append(np.asarray(fs.get_samples(int(TAIL_SECONDS * SR)), dtype=np.int16))
    pcm = np.concatenate(chunks).reshape(-1, 2).astype(np.float32) / 32768.0
    return pcm


# ── 방법 B: fluidsynth 명령행 (pyfluidsynth가 없을 때) ──
def _render_cli(req: FullReq, sf2: str) -> np.ndarray:
    import mido
    chans = _channels(req.tracks)
    tpb = 480
    mid = mido.MidiFile(ticks_per_beat=tpb)
    trk = mido.MidiTrack(); mid.tracks.append(trk)
    trk.append(mido.MetaMessage("set_tempo", tempo=int(60_000_000 / req.bpm), time=0))
    for tr, ch in zip(req.tracks, chans):
        if not tr.drum:
            trk.append(mido.Message("program_change", channel=ch, program=tr.instrument, time=0))
        trk.append(mido.Message("control_change", channel=ch, control=7, value=tr.volume, time=0))
        trk.append(mido.Message("control_change", channel=ch, control=91, value=40, time=0))
    raw = []
    for tr, ch in zip(req.tracks, chans):
        for nt in tr.notes:
            on = int(round(nt.t * tpb)); off = on + max(10, int(round(nt.d * tpb)) - 5)
            raw.append((on, 1, "note_on", ch, nt.n, nt.v))
            raw.append((off, 0, "note_off", ch, nt.n, 0))
    raw.sort(key=lambda e: (e[0], e[1]))
    last = 0
    for tk, _o, kind, ch, key, vel in raw:
        trk.append(mido.Message(kind, channel=ch, note=key, velocity=vel, time=tk - last)); last = tk
    trk.append(mido.MetaMessage("end_of_track", time=int(TAIL_SECONDS * req.bpm / 60 * tpb)))
    with tempfile.TemporaryDirectory() as d:
        mp, wp = os.path.join(d, "in.mid"), os.path.join(d, "out.wav")
        mid.save(mp)
        subprocess.run(["fluidsynth", "-ni", "-g", "0.6", "-r", str(SR), "-F", wp, sf2, mp],
                       check=True, capture_output=True, timeout=120)
        with wave.open(wp) as w:
            nch = w.getnchannels()
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data.reshape(-1, nch).astype(np.float32) / 32768.0


def _to_wav(pcm: np.ndarray, normalize: bool = True) -> bytes:
    pk = float(np.max(np.abs(pcm))) if pcm.size else 0.0
    if normalize and pk > 1e-4:
        pcm = pcm * (0.9 / pk)          # 피크 정규화
    i16 = np.clip(pcm * 32767, -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(i16.shape[1]); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(i16.tobytes())
    return buf.getvalue()


@router.get("/render_full")
def render_full_info():
    """배포 확인용: 브라우저에서 열면 사운드폰트·엔진 상태 표시"""
    try:
        import fluidsynth  # noqa
        engine = "pyfluidsynth"
    except Exception:
        engine = "cli" if shutil.which("fluidsynth") else "none"
    return JSONResponse({"ok": True, "endpoint": "render_full", "engine": engine, "sf2": _find_sf2()})


@router.post("/render_full")
def render_full(req: FullReq):
    if not req.tracks or len(req.tracks) > MAX_TRACKS:
        raise HTTPException(400, f"tracks 1~{MAX_TRACKS}개")
    if sum(len(t.notes) for t in req.tracks) > MAX_NOTES:
        raise HTTPException(400, f"notes 최대 {MAX_NOTES}개")
    if sum(1 for t in req.tracks if t.drum) > 1:
        raise HTTPException(400, "drum 트랙은 1개만")
    end_beats = max((n.t + n.d for t in req.tracks for n in t.notes), default=0)
    if end_beats * 60.0 / req.bpm > MAX_SECONDS:
        raise HTTPException(400, f"최대 {MAX_SECONDS}초")
    sf2 = _find_sf2()
    if not sf2:
        raise HTTPException(500, "사운드폰트(.sf2) 없음 — CGO_SF2 환경변수로 경로 지정")
    try:
        import fluidsynth  # noqa
        render_one = _render_pyfs
    except ImportError:
        if not shutil.which("fluidsynth"):
            raise HTTPException(500, "FluidSynth 엔진 없음")
        render_one = _render_cli
    # v2: 트랙별로 따로 렌더 → 멜로디 기준 밸런스 믹스 → 마스터(음량 키우기 + 리미터)
    stems = []
    for tr in req.tracks:
        tr1 = Track(name=tr.name, instrument=tr.instrument, drum=tr.drum, volume=127, notes=tr.notes)
        stems.append((tr, render_one(FullReq(bpm=req.bpm, tracks=[tr1]), sf2)))
    pcm = _mix_master(stems)
    return Response(content=_to_wav(pcm, normalize=False), media_type="audio/wav")


# ═══ v2 믹스·마스터 ═══════════════════════════════════════════════
# 멜로디를 0 dB 기준으로 두고 나머지를 이만큼 낮춤 (값이 작을수록 뒤로 물러남)
MIX_DB = {"melody": 4.0, "chords": 0.0, "bass": 0.0, "drums": 0.0}
TARGET_DBFS = -8.0      # 멜로디 드럼보다 1/3(+4dB) 높게, 배경음=드럼 동일
CEILING = 0.89          # 최고점 한계 (-1 dBFS)


def _level_db(x: np.ndarray) -> float:
    """소리가 나는 구간만의 평균 음량(dB). 쉼표·꼬리 무음은 제외."""
    blk = 2205  # 50ms
    n = (len(x) // blk) * blk
    if n == 0:
        return -120.0
    r = np.sqrt((x[:n].reshape(-1, blk, x.shape[1]) ** 2).mean(axis=(1, 2)))
    r = r[r > 1e-4]
    if r.size == 0:
        return -120.0
    r = np.sort(r)[len(r) // 3:]             # 조용한 1/3 제외
    return float(20 * np.log10(np.sqrt((r ** 2).mean())))


def _limit(x: np.ndarray, ceil: float = CEILING) -> np.ndarray:
    """룩어헤드 리미터: 튀는 순간(킥·심벌)만 살짝 눌러서 전체를 크게 키울 수 있게 함."""
    blk = 88  # 2ms
    n = len(x)
    nb = (n + blk - 1) // blk
    pk = np.pad(np.abs(x).max(axis=1), (0, nb * blk - n)).reshape(nb, blk).max(axis=1)
    look = np.maximum(pk, np.concatenate([pk[1:], pk[-1:]]))
    look = np.maximum(look, np.concatenate([pk[2:], pk[-1:], pk[-1:]]))
    g = np.minimum(1.0, ceil / np.maximum(look, 1e-9))
    rel = np.exp(-blk / (SR * 0.05))          # 50ms 회복
    out = np.empty_like(g)
    cur = 1.0
    for i in range(nb):
        cur = g[i] if g[i] < cur else cur * rel + g[i] * (1 - rel)
        out[i] = cur
    gs = np.repeat(out, blk)[:n]
    return np.clip(x * gs[:, None], -ceil, ceil)


def _compress(x: np.ndarray, thr_db: float, ratio: float = 3.0,
              att: float = 0.010, rel: float = 0.150) -> np.ndarray:
    """부드러운 컴프레서: 큰 소리와 작은 소리 차이를 줄여 전체를 크게 들리게 함."""
    blk = 220  # 5ms
    n = len(x)
    nb = (n + blk - 1) // blk
    sq = np.pad((x ** 2).mean(axis=1), (0, nb * blk - n)).reshape(nb, blk).mean(axis=1)
    lv = 10 * np.log10(np.maximum(sq, 1e-12))
    over = np.maximum(0.0, lv - thr_db)
    target = -over * (1 - 1 / ratio)          # dB 감쇄량
    a_c = np.exp(-blk / (SR * att)); r_c = np.exp(-blk / (SR * rel))
    g = np.empty_like(target); cur = 0.0
    for i in range(nb):
        c = a_c if target[i] < cur else r_c
        cur = c * cur + (1 - c) * target[i]
        g[i] = cur
    gs = np.interp(np.arange(n), np.arange(nb) * blk + blk / 2, 10 ** (g / 20))
    return x * gs[:, None]


def _mix_master(stems) -> np.ndarray:
    n = max(len(p) for _, p in stems)
    mix = np.zeros((n, 2), dtype=np.float32)
    for tr, p in stems:
        if p.shape[1] == 1:
            p = np.repeat(p, 2, axis=1)
        name = "drums" if tr.drum else (tr.name or "melody")
        lv = _level_db(p)
        if lv <= -119:
            continue
        rel = MIX_DB.get(name, -6.0)
        gain = 10 ** ((-20.0 + rel - lv) / 20)   # 각 트랙을 '기준 -20dB + 상대 dB'로 맞춤
        mix[:len(p)] += (p * gain).astype(np.float32)
    lv = _level_db(mix)
    if lv <= -119:
        return mix
    mix *= 10 ** ((-18.0 - lv) / 20)          # 작업 레벨로 맞춘 뒤
    mix = _compress(mix, thr_db=-24.0, ratio=3.0)   # 압축
    lv = _level_db(mix)
    mix *= 10 ** ((TARGET_DBFS - lv) / 20)    # 목표 음량까지 키우고
    return _limit(mix)                        # 튀는 순간만 리미터로 정리


# ── 기존 방식 호환: POST /render (음표를 순서대로 이어서 렌더) ──
class SimpleNote(BaseModel):
    n: int = Field(..., ge=0, le=127)
    d: float = Field(..., gt=0)


class SimpleReq(BaseModel):
    bpm: float = Field(..., ge=20, le=300)
    notes: List[SimpleNote]
    instrument: int = Field(0, ge=0, le=127)


@app.post("/render")
def render(req: SimpleReq):
    t, ns = 0.0, []
    for x in req.notes:
        ns.append({"n": x.n, "t": t, "d": x.d})
        t += x.d
    return render_full(FullReq(bpm=req.bpm, tracks=[Track(instrument=req.instrument, notes=ns)]))


# ── cgo-467: 서버가 자기 버전을 말하게 한다 ──────────────────────────
# 지금까지는 배포가 되었는지 눈으로 알 길이 없었다. 레일웨이 화면의
# "Deployment successful"은 '무언가'가 올라갔다는 뜻일 뿐, 그게 어느 판인지는
# 말해주지 않는다. 이제 주소만 열면 버전이 보인다.
CGO_SRV_VER = "cgo-479"
CGO_SRV_NOTE = "붐비면 옆 모델로(479) · 수노 스타일 압축(478) · 가사 주문 짧게(477)"


def _cgo_key_src() -> str:
    """키를 어디서 가져왔는지만 알린다. 키 자체는 절대 내보내지 않는다."""
    return "env" if os.environ.get('APIFRAME_KEY') else "builtin"


@app.get("/")
def root():
    return {"ok": True, "service": "cgo-render", "sf2": _find_sf2(), "vvip": True,
            "ver": CGO_SRV_VER, "key_src": _cgo_key_src()}


@app.get("/version")
def version():
    """배포 확인 전용. 휴대폰 브라우저에서 열어 'ver'만 보면 된다."""
    return {"ver": CGO_SRV_VER, "note": CGO_SRV_NOTE,
            "key_src": _cgo_key_src(),
            "lyrics_ai": (" → ".join(CGO_LLM_MODELS) if CGO_LLM_KEY else "꺼짐 — CGO_LLM_KEY 미설정"),
            "started": time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(_CGO_BOOT_TS)) + " UTC",
            "uptime_min": round((time.time() - _CGO_BOOT_TS) / 60, 1)}


# ═══════════════════════════════════════════════════════════════════
# VVIP AI 보컬 엔드포인트 — Suno via apiframe.ai
# CGO-FULI 작곡 데이터 + 사용자 프롬프트 → Suno API → AI 보컬 곡 생성
# 원칙: 최소 2명 이상의 서로 다른 보컬리스트 조합 필수
# ═══════════════════════════════════════════════════════════════════

# cgo-468: 새 키로 교체. 레일웨이 Variables에 APIFRAME_KEY를 넣으면 그쪽이 먼저다.
APIFRAME_KEY = os.environ.get('APIFRAME_KEY', 'afk_a23fbf3d6ffe106e4a8b23827cff16864457086f')

# ── 인류 역사상 최고의 보컬리스트 100명 (남50 + 여50) ──
# 한국어(붙여쓰기+띄어쓰기) + 영문 이름 → 영어 보컬 설명
VOICE_MAP: Dict[str, str] = {
    # ═══ 남자 보컬 50명 ═══
    '프레디머큐리': 'dramatic powerful male rock vocal with 4-octave theatrical range and perfect stage presence',
    '프레디 머큐리': 'dramatic powerful male rock vocal with 4-octave theatrical range and perfect stage presence',
    '마이클잭슨': 'rhythmic male pop vocal freely switching between falsetto and chest voice',
    '마이클 잭슨': 'rhythmic male pop vocal freely switching between falsetto and chest voice',
    '로버트플랜트': 'high-pitched screaming male rock vocal, textbook hard rock tenor',
    '로버트 플랜트': 'high-pitched screaming male rock vocal, textbook hard rock tenor',
    '액슬로즈': 'sharp nervous ultra-high screaming male rock vocal with wide range',
    '액슬 로즈': 'sharp nervous ultra-high screaming male rock vocal with wide range',
    '스티븐타일러': 'tireless iron-throated high male rock vocal with unique groove',
    '스티븐 타일러': 'tireless iron-throated high male rock vocal with unique groove',
    '엘비스프레슬리': 'charming deep baritone male vocal, the king of rock and roll',
    '엘비스 프레슬리': 'charming deep baritone male vocal, the king of rock and roll',
    '데이비드보위': 'theatrical mysterious mid-low male vocal with glam rock charisma',
    '데이비드 보위': 'theatrical mysterious mid-low male vocal with glam rock charisma',
    '프랭크시나트라': 'smooth classic baritone male crooner jazz pop vocal',
    '프랭크 시나트라': 'smooth classic baritone male crooner jazz pop vocal',
    '조지마이클': 'perfect vocal technique with appealing sweet falsetto male vocal',
    '조지 마이클': 'perfect vocal technique with appealing sweet falsetto male vocal',
    '엘튼존': 'powerful heartfelt classic pop male vocal with piano accompaniment',
    '엘튼 존': 'powerful heartfelt classic pop male vocal with piano accompaniment',
    '로이오비슨': 'dramatic operatic male vocal with soaring sorrowful high notes',
    '로이 오비슨': 'dramatic operatic male vocal with soaring sorrowful high notes',
    '존레논': 'raw rough soul-shaking male rock vocal with emotional honesty',
    '존 레논': 'raw rough soul-shaking male rock vocal with emotional honesty',
    '폴매카트니': 'versatile male vocal from soft falsetto to rock screaming',
    '폴 매카트니': 'versatile male vocal from soft falsetto to rock screaming',
    '스티비원더': 'genius rhythmic soulful male vocal with brilliant melisma technique',
    '스티비 원더': 'genius rhythmic soulful male vocal with brilliant melisma technique',
    '마빈게이': 'silky smooth sensual male Motown soul vocal',
    '마빈 게이': 'silky smooth sensual male Motown soul vocal',
    '레이찰스': 'soulful male vocal with deep pain and emotion, the godfather of soul',
    '레이 찰스': 'soulful male vocal with deep pain and emotion, the godfather of soul',
    '오티스레딩': 'heart-tearing sorrowful explosive male soul vocal',
    '오티스 레딩': 'heart-tearing sorrowful explosive male soul vocal',
    '샘쿡': 'clear smooth pure falsetto male R&B vocal',
    '샘 쿡': 'clear smooth pure falsetto male R&B vocal',
    '루이암스트롱': 'gravelly uniquely husky deep male jazz vocal, one of a kind tone',
    '루이 암스트롱': 'gravelly uniquely husky deep male jazz vocal, one of a kind tone',
    '넷킹콜': 'silky warm healing gentle male vocal like velvet',
    '넷 킹 콜': 'silky warm healing gentle male vocal like velvet',
    '프린스': 'genius sensual male vocal switching between falsetto and chest voice with incredible range',
    '루더반드로스': 'velvety smooth perfect male R&B ballad vocal with full volume',
    '루더 반드로스': 'velvety smooth perfect male R&B ballad vocal with full volume',
    '와냐모리스': 'overwhelming male R&B lead vocal with rich harmonics',
    '와냐 모리스': 'overwhelming male R&B lead vocal with rich harmonics',
    '브루노마스': 'solid high male vocal with retro and modern groove, energetic performer',
    '브루노 마스': 'solid high male vocal with retro and modern groove, energetic performer',
    '위켄드': 'dreamy sophisticated falsetto male vocal reminiscent of Michael Jackson',
    '디앤젤로': 'sticky deep neo-soul male vocal that melts the heart',
    '크리스코넬': 'most devastating powerful 4-octave male rock vocal in history, shattering intensity',
    '크리스 코넬': 'most devastating powerful 4-octave male rock vocal in history, shattering intensity',
    '체스터베닝턴': 'sorrowful falsetto transitioning to angry melodic screaming male vocal',
    '체스터 베닝턴': 'sorrowful falsetto transitioning to angry melodic screaming male vocal',
    '커트코베인': 'scratched wounded raspy male grunge vocal representing generational frustration',
    '커트 코베인': 'scratched wounded raspy male grunge vocal representing generational frustration',
    '에디베더': 'deep baritone male alternative rock vocal icon with emotional depth',
    '에디 베더': 'deep baritone male alternative rock vocal icon with emotional depth',
    '제프버클리': 'angelic fragile yet devastating falsetto male vocal with soul-shaking emotion',
    '제프 버클리': 'angelic fragile yet devastating falsetto male vocal with soul-shaking emotion',
    '톰요크': 'nervous yet beautiful dreamy falsetto male vocal, ethereal and haunting',
    '톰 요크': 'nervous yet beautiful dreamy falsetto male vocal, ethereal and haunting',
    '로니제임스디오': 'sacred powerful metal male vocal with commanding volume from small frame',
    '로니 제임스 디오': 'sacred powerful metal male vocal with commanding volume from small frame',
    '브루스디킨슨': 'explosive operatic metal male vocal like a human air raid siren',
    '브루스 디킨슨': 'explosive operatic metal male vocal like a human air raid siren',
    '코어리테일러': 'genius male vocal combining demonic growling with sweet clean singing',
    '코어리 테일러': 'genius male vocal combining demonic growling with sweet clean singing',
    '리암갤러거': 'arrogant cynical distinctive male britpop vocal that defined an era',
    '리암 갤러거': 'arrogant cynical distinctive male britpop vocal that defined an era',
    '보노': 'stadium-filling resonant male vocal with powerful message delivery',
    # 🇰🇷 대한민국 남자 보컬 거장
    '임재범': 'beast-like husky male vocal with emotional sorrow and raw power',
    '이문세': 'deep literary lyrical male ballad vocal with quiet resonance',
    '김범수': 'flawless technique male mastering every emotion perfectly',
    '박효신': 'evolved male vocal reaching divine territory from folk to pop ballad perfection',
    '임창정': 'desperately emotional high-pitched male that makes everyone cry',
    '조용필': 'the king , versatile male vocal covering rock ballad and folk',
    '김광석': 'warm folk acoustic with heartfelt lonely storytelling',
    '김현식': 'rough torn raspy male vocal pouring soul until the last breath',
    '하현우': 'stable ultra-high male vocal testing human limits with incredible range',
    '이수': 'overwhelming falsetto high male vocal dominating karaoke',
    '나얼': 'pinnacle of soul male vocal with perfect high-tone technique',
    '신해철': 'deep heavy charismatic low male vocal carrying philosophical messages',
    # ═══ 여자 보컬 50명 ═══
    '휘트니휴스턴': 'The Voice, perfect female vocal with flawless power pitch and emotion, the gold standard',
    '휘트니 휴스턴': 'The Voice, perfect female vocal with flawless power pitch and emotion, the gold standard',
    '머라이어캐리': '5-octave female vocal with dolphin whistle register and R&B melisma revolution',
    '머라이어 캐리': '5-octave female vocal with dolphin whistle register and R&B melisma revolution',
    '셀린디온': 'crystal clear yet steel-strong female belting high vocal filling stadiums',
    '셀린 디온': 'crystal clear yet steel-strong female belting high vocal filling stadiums',
    '아델': 'deep classic husky female vocal with overwhelming emotional delivery that makes the world cry',
    '비욘세': 'perfect powerful female R&B pop vocal with flawless technique even during intense choreography',
    '마돈나': 'iconic pop female vocal with unique tone that led an entire era',
    '바브라스트라이샌드': 'flawless classic female vocal mastering Broadway and pop with zero error',
    '바브라 스트라이샌드': 'flawless classic female vocal mastering Broadway and pop with zero error',
    '레이디가가': 'deep powerful female vocal consuming jazz rock and pop with incredible range',
    '레이디 가가': 'deep powerful female vocal consuming jazz rock and pop with incredible range',
    '도나서머': 'explosive female disco vocal cutting through club beats with cool power',
    '도나 서머': 'explosive female disco vocal cutting through club beats with cool power',
    '티나터너': 'explosive energy rough husky female rock and roll vocal that destroys the stage',
    '티나 터너': 'explosive energy rough husky female rock and roll vocal that destroys the stage',
    '크리스티나아길레라': 'raw powerful black-soul-based female belting vocal from small frame',
    '크리스티나 아길레라': 'raw powerful black-soul-based female belting vocal from small frame',
    '아리아나그란데': 'clear bright female pop vocal with ultra-high technique continuing the Mariah Carey legacy',
    '아리아나 그란데': 'clear bright female pop vocal with ultra-high technique continuing the Mariah Carey legacy',
    '아레사프랭클린': 'queen of soul, gospel-based explosive powerful female vocal full of holy spirit',
    '아레사 프랭클린': 'queen of soul, gospel-based explosive powerful female vocal full of holy spirit',
    '에디트피아프': 'sorrowful French chanson female vocal pouring raw life pain like a violin',
    '에디트 피아프': 'sorrowful French chanson female vocal pouring raw life pain like a violin',
    '빌리홀리데이': 'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing',
    '빌리 홀리데이': 'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing',
    '엘라피츠제럴드': 'first lady of jazz, perfect pitch rhythm and freestyle scat female vocal',
    '엘라 피츠제럴드': 'first lady of jazz, perfect pitch rhythm and freestyle scat female vocal',
    '니나시몬': 'deep heavy contralto female vocal singing the Black soul with gravitas',
    '니나 시몬': 'deep heavy contralto female vocal singing the Black soul with gravitas',
    '에이미와인하우스': 'rebellious melancholic raw retro soul jazz female vocal, one of a kind tone',
    '에이미 와인하우스': 'rebellious melancholic raw retro soul jazz female vocal, one of a kind tone',
    '에타제임스': 'powerful rough soulful female blues vocal bridging blues and soul',
    '에타 제임스': 'powerful rough soulful female blues vocal bridging blues and soul',
    '사데': 'most sophisticated calm sensual mid-low female vocal with luxury tone',
    '차카칸': 'fire-breathing piercing metallic high female funk R&B vocal',
    '차카 칸': 'fire-breathing piercing metallic high female funk R&B vocal',
    '로린힐': 'husky soulful female vocal fusing and R&B with deep charm',
    '로린 힐': 'husky soulful female vocal fusing and R&B with deep charm',
    '알리시아키스': 'husky intelligent female R&B vocal with classic piano accompaniment',
    '알리시아 키스': 'husky intelligent female R&B vocal with classic piano accompaniment',
    '에리카바두': 'unique sophisticated neo-soul queen female vocal reminiscent of Billie Holiday',
    '에리카 바두': 'unique sophisticated neo-soul queen female vocal reminiscent of Billie Holiday',
    '노라존스': 'quiet warm soothing female jazz folk vocal that heals the world',
    '노라 존스': 'quiet warm soothing female jazz folk vocal that heals the world',
    '재니스조플린': 'most destructive female rock vocal in history, blood-vessel-popping screaming wail',
    '재니스 조플린': 'most destructive female rock vocal in history, blood-vessel-popping screaming wail',
    '스티비닉스': 'mysterious witch-like rough vibrato female rock vocal with mystical charm',
    '스티비 닉스': 'mysterious witch-like rough vibrato female rock vocal with mystical charm',
    '돌로레스오리어던': 'sad sharp Irish traditional female vocal with sorrowful piercing tone',
    '돌로레스 오리어던': 'sad sharp Irish traditional female vocal with sorrowful piercing tone',
    '비요크': 'otherworldly bizarre yet beautiful avant-garde female vocal beyond human range',
    '돌리파튼': 'bouncy yet heartfelt country female vocal with unique vibrato and storytelling',
    '돌리 파튼': 'bouncy yet heartfelt country female vocal with unique vibrato and storytelling',
    '케이트부시': 'ethereal theatrical falsetto female vocal with 4th-dimensional mystical quality',
    '케이트 부시': 'ethereal theatrical falsetto female vocal with 4th-dimensional mystical quality',
    '헤일리윌리엄스': 'unwavering cool soaring high female pop-punk vocal',
    '헤일리 윌리엄스': 'unwavering cool soaring high female pop-punk vocal',
    '타랴투루넨': 'classical soprano female vocal pioneering symphonic metal genre',
    '타랴 투루넨': 'classical soprano female vocal pioneering symphonic metal genre',
    '시네이드오코너': 'ice-cold sad rebellious female vocal with raw emotional intensity',
    '시네이드 오코너': 'ice-cold sad rebellious female vocal with raw emotional intensity',
    '샤니아트웨인': 'refreshing powerful female country-pop crossover vocal',
    '샤니아 트웨인': 'refreshing powerful female country-pop crossover vocal',
    '애니레녹스': 'androgynous cold urban charismatic female vocal with cool presence',
    '애니 레녹스': 'androgynous cold urban charismatic female vocal with cool presence',
    '에이미리': 'mysterious powerful gothic female rock vocal piercing through dark orchestral sound',
    '에이미 리': 'mysterious powerful gothic female rock vocal piercing through dark orchestral sound',
    '패티스미스': 'punk rock godmother female vocal chanting poetry of resistance beyond technique',
    '패티 스미스': 'punk rock godmother female vocal chanting poetry of resistance beyond technique',
    # 🇰🇷 대한민국 여성 보컬 거장
    '소향': 'world-class 5-octave powerful with dramatic high notes and pop diva power',
    '태연': 'unique delicate representing a generation with perfect control and emotion',
    '이선희': 'explosive power from small frame, timeless clear sorrowful',
    '백지영': 'husky heartbreak-filled female OST ballad queen vocal',
    '박정현': 'fairy female vocal with perfect breath control and brilliant melisma',
    '거미': 'deep husky soulful female ballad vocal with Black soul influence',
    '이은미': 'barefoot diva, deeply appealing drawn from the depths of the heart',
    '인순이': 'explosive powerful energetic mastering soul dance and pop',
    '윤미래': 'sticky deep husky at the pinnacle',
    '아이유': 'gentle acoustic to soaring high notes, clear storytelling defining an era',
    '심수봉': 'legendary nasal sorrowful uniquely toned female vocal soaking the soul',
    '소찬휘': 'ultimate female high-note queen with blade-sharp piercing rapid vocal delivery',
    # ── 영문 이름 매핑 ──
    'Freddie Mercury': 'dramatic powerful male rock vocal with 4-octave theatrical range',
    'Michael Jackson': 'rhythmic male pop vocal freely switching between falsetto and chest voice',
    'Whitney Houston': 'The Voice, perfect female vocal with flawless power pitch and emotion',
    'Mariah Carey': '5-octave female vocal with dolphin whistle register and R&B melisma',
    'Celine Dion': 'crystal clear yet steel-strong female belting high vocal filling stadiums',
    'Adele': 'deep classic husky female vocal with overwhelming emotional delivery',
    'Beyonce': 'perfect powerful female R&B pop vocal with flawless technique',
    'Lady Gaga': 'deep powerful female vocal consuming jazz rock and pop',
    'Bruno Mars': 'solid high male vocal with retro and modern groove',
    'The Weeknd': 'dreamy sophisticated falsetto male vocal',
    'Ariana Grande': 'clear bright female pop vocal with ultra-high technique',
    'Ed Sheeran': 'warm intimate male folk pop vocal with gentle rasp',
    'Prince': 'genius sensual male vocal switching between falsetto and chest voice',
    '에드시런': 'warm intimate male folk pop vocal with gentle rasp',
    '에드 시런': 'warm intimate male folk pop vocal with gentle rasp',
    '올리비아뉴튼존': 'soft breathy warm female pop vocal with gentle tender tone',
    '올리비아 뉴튼존': 'soft breathy warm female pop vocal with gentle tender tone',
    '보니테일러': 'husky raspy female rock ballad vocal with raw dramatic emotion',
    '보니 테일러': 'husky raspy female rock ballad vocal with raw dramatic emotion',
    '나훈아': 'deep emotional dramatic with powerful vibrato',

    # ═══════════════════════════════════════════════════════
    # 🎹 팝 발라드 300인 보컬 믹서 확장 (cgo-377)
    # ═══════════════════════════════════════════════════════

    # ── Global Male: Piano-Based Ballad ──
    '빌리조엘': 'storytelling piano male vocal with warm gritty New York baritone charm',
    '빌리 조엘': 'storytelling piano male vocal with warm gritty New York baritone charm',
    '존레전드': 'smooth silky male R&B piano vocal with tender romantic warmth',
    '존 레전드': 'smooth silky male R&B piano vocal with tender romantic warmth',
    '찰리푸스': 'precise pitch-perfect male pop vocal with clean falsetto and modern production edge',
    '찰리 푸스': 'precise pitch-perfect male pop vocal with clean falsetto and modern production edge',
    '마이클부블레': 'polished swinging male crooner vocal with warm big-band jazz tone',
    '마이클 부블레': 'polished swinging male crooner vocal with warm big-band jazz tone',
    '베리매닐로우': 'theatrical sweeping male piano ballad vocal with dramatic crescendo delivery',
    '베리 매닐로우': 'theatrical sweeping male piano ballad vocal with dramatic crescendo delivery',
    '토니베넷': 'timeless elegant male jazz crooner vocal with effortless classic phrasing',
    '토니 베넷': 'timeless elegant male jazz crooner vocal with effortless classic phrasing',
    '닐세다카': 'bright cheerful male pop vocal with catchy melodic 60s piano flair',
    '닐 세다카': 'bright cheerful male pop vocal with catchy melodic 60s piano flair',
    '길버트오설리반': 'gentle wistful male vocal with delicate lyrical piano folk-pop phrasing',
    '길버트 오설리반': 'gentle wistful male vocal with delicate lyrical piano folk-pop phrasing',
    '루카스그레이엄': 'earnest narrative male pop vocal with soulful rasp and confessional tone',
    '루카스 그레이엄': 'earnest narrative male pop vocal with soulful rasp and confessional tone',
    '톰오델': 'raw aching male piano vocal that erupts from whisper to desperate wail',
    '톰 오델': 'raw aching male piano vocal that erupts from whisper to desperate wail',
    '에릭카멘': 'lush romantic male vocal blending classical piano grandeur with pop yearning',
    '에릭 카멘': 'lush romantic male vocal blending classical piano grandeur with pop yearning',
    '데이비드게이츠': 'feathery soft tender male vocal with airy gentle folk-pop delivery',
    '데이비드 게이츠': 'feathery soft tender male vocal with airy gentle folk-pop delivery',
    '샘스미스': 'trembling soulful male vocal with aching falsetto and vulnerable emotional depth',
    '샘 스미스': 'trembling soulful male vocal with aching falsetto and vulnerable emotional depth',
    '캘럼스콧': 'transparent fragile male vocal with crystalline sad tone and quiet intensity',
    '캘럼 스콧': 'transparent fragile male vocal with crystalline sad tone and quiet intensity',
    '가빈디그로우': 'gritty warm male keyboard-soul vocal with bluesy rasp and heartfelt punch',
    '가빈 디그로우': 'gritty warm male keyboard-soul vocal with bluesy rasp and heartfelt punch',
    '앤디윌리엄스': 'pristine smooth male easy-listening vocal with serene velvety mid-range tone',
    '앤디 윌리엄스': 'pristine smooth male easy-listening vocal with serene velvety mid-range tone',

    # ── Global Male: Power/Rock Ballad ──
    '마이클볼튼': 'husky powerful male belting vocal with soul-drenched rock ballad intensity',
    '마이클 볼튼': 'husky powerful male belting vocal with soul-drenched rock ballad intensity',
    '리차드막스': 'soaring male rock ballad vocal with polished tenor and guitar-driven passion',
    '리차드 막스': 'soaring male rock ballad vocal with polished tenor and guitar-driven passion',
    '스티브페리': 'legendary pure high male tenor with effortless sustained arena rock notes',
    '스티브 페리': 'legendary pure high male tenor with effortless sustained arena rock notes',
    '브라이언애덤스': 'raspy warm male rock vocal with anthemic sing-along ballad grit',
    '브라이언 애덤스': 'raspy warm male rock vocal with anthemic sing-along ballad grit',
    '피터세테라': 'crystalline soaring high male tenor with smooth Chicago soft-rock shimmer',
    '피터 세테라': 'crystalline soaring high male tenor with smooth Chicago soft-rock shimmer',
    '존본조비': 'gritty charismatic male stadium-rock vocal with fist-pumping anthem delivery',
    '존 본 조비': 'gritty charismatic male stadium-rock vocal with fist-pumping anthem delivery',
    '필콜린스': 'emotive building male vocal from restrained verse to explosive drum-driven chorus',
    '필 콜린스': 'emotive building male vocal from restrained verse to explosive drum-driven chorus',
    '조코커': 'raw convulsive gravelly male vocal wringing every note with blues agony',
    '조 코커': 'raw convulsive gravelly male vocal wringing every note with blues agony',
    '로드스튜어트': 'sandpaper-rough charming male vocal with loose swaggering rock-ballad phrasing',
    '로드 스튜어트': 'sandpaper-rough charming male vocal with loose swaggering rock-ballad phrasing',
    '제임스아서': 'rough textured male vocal with gritty emotional rock-ballad appeal and edge',
    '제임스 아서': 'rough textured male vocal with gritty emotional rock-ballad appeal and edge',
    '루이스카팔디': 'broken sobbing male vocal pouring desperate modern heartbreak with Scottish rasp',
    '루이스 카팔디': 'broken sobbing male vocal pouring desperate modern heartbreak with Scottish rasp',
    '게리무어': 'wailing blues-rock male vocal fused with crying electric guitar tone',
    '게리 무어': 'wailing blues-rock male vocal fused with crying electric guitar tone',
    '대니오도노휴': 'passionate climbing male vocal with urgent emotional pop-rock conviction',
    '대니 오도노휴': 'passionate climbing male vocal with urgent emotional pop-rock conviction',
    '크리스마틴': 'airy ethereal male falsetto vocal building to sweeping arena-rock crescendo',
    '크리스 마틴': 'airy ethereal male falsetto vocal building to sweeping arena-rock crescendo',
    '롭토마스': 'punchy dynamic male vocal with bright midrange pop-rock drive and grit',
    '롭 토마스': 'punchy dynamic male vocal with bright midrange pop-rock drive and grit',
    '에어서플라이': 'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes',
    '러셀히치콕': 'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes',
    '러셀 히치콕': 'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes',

    # ── Global Male: R&B/Soul Ballad ──
    '브라이언맥나이트': 'pristine smooth falsetto with effortless high register over lush harmonics',
    '브라이언 맥나이트': 'pristine smooth falsetto with effortless high register over lush harmonics',
    '베이비페이스': 'tender warm midrange with intimate gentle romantic R&B ballad phrasing',
    '베이비 페이스': 'tender warm midrange with intimate gentle romantic R&B ballad phrasing',
    '피보브라이슨': 'refined velvety tenor with elegant soaring romantic soul pop delivery',
    '피보 브라이슨': 'refined velvety tenor with elegant soaring romantic soul pop delivery',
    '제임스잉그램': 'luxurious deep soulful baritone with rich gospel-touched warm resonance',
    '제임스 잉그램': 'luxurious deep soulful baritone with rich gospel-touched warm resonance',
    '라이오넬리치': 'warm inviting midrange with gentle easygoing pop ballad groove feel',
    '라이오넬 리치': 'warm inviting midrange with gentle easygoing pop ballad groove feel',
    '알그린': 'airy ethereal falsetto with raw sensual gospel tension and sweet grit',
    '알 그린': 'airy ethereal falsetto with raw sensual gospel tension and sweet grit',
    '맥스웰': 'breathy dreamy falsetto layered in urban neo-soul lush atmosphere',
    '도니헤더웨이': 'deeply emotive soulful tenor with commanding raw gospel-rooted intensity',
    '도니 헤더웨이': 'deeply emotive soulful tenor with commanding raw gospel-rooted intensity',
    '테디펜더그라스': 'commanding heavy baritone with powerful sensual deep soul growl',
    '테디 펜더그라스': 'commanding heavy baritone with powerful sensual deep soul growl',
    '알제로': 'agile scatting tenor blending jazz improvisation with smooth pop finesse',
    '알 제로': 'agile scatting tenor blending jazz improvisation with smooth pop finesse',
    '알자로': 'agile scatting tenor blending jazz improvisation with smooth pop finesse',
    '시엘': 'distinctive husky baritone with dramatic soaring falsetto soul breaks',
    '빌위더스': 'honest unpretentious warm baritone with pure heartfelt soul simplicity',
    '빌 위더스': 'honest unpretentious warm baritone with pure heartfelt soul simplicity',

    # ── Global Male: Acoustic/Folk Ballad ──
    '제임스테일러': 'gentle soothing baritone with the warmest comforting folk vocal intimacy',
    '제임스 테일러': 'gentle soothing baritone with the warmest comforting folk vocal intimacy',
    '에릭클랩튼': 'weathered bluesy midrange with aching understated acoustic emotional depth',
    '에릭 클랩튼': 'weathered bluesy midrange with aching understated acoustic emotional depth',
    '존덴버': 'bright pure tenor with wholesome nature-inspired clear acoustic tone',
    '존 덴버': 'bright pure tenor with wholesome nature-inspired clear acoustic tone',
    '돈맥클린': 'clear earnest tenor with sweeping poetic folk storytelling grandeur',
    '돈 맥클린': 'clear earnest tenor with sweeping poetic folk storytelling grandeur',
    '데미안라이스': 'raw fragile tenor building from whisper to intense acoustic crescendo',
    '데미안 라이스': 'raw fragile tenor building from whisper to intense acoustic crescendo',
    '제이슨므라즈': 'easygoing sunny tenor with playful organic folk pop vocal charm',
    '제이슨 므라즈': 'easygoing sunny tenor with playful organic folk pop vocal charm',
    '잭존슨': 'laid-back mellow baritone with serene breezy minimal acoustic calm',
    '잭 존슨': 'laid-back mellow baritone with serene breezy minimal acoustic calm',
    '글렌프레이': 'smooth confident midrange with relaxed classic western folk rock ease',
    '글렌 프레이': 'smooth confident midrange with relaxed classic western folk rock ease',
    '돈헨리': 'refined slightly nasal tenor with polished introspective rock sensibility',
    '돈 헨리': 'refined slightly nasal tenor with polished introspective rock sensibility',
    '닐다이아몬드': 'resonant deep baritone with dramatic anthemic folk pop vocal projection',
    '닐 다이아몬드': 'resonant deep baritone with dramatic anthemic folk pop vocal projection',
    '캣스티븐스': 'gentle meditative tenor with searching spiritual folk vocal sincerity',
    '캣 스티븐스': 'gentle meditative tenor with searching spiritual folk vocal sincerity',
    '크리스토퍼크로스': 'feathery high tenor with breezy soft yacht rock silky smoothness',
    '크리스토퍼 크로스': 'feathery high tenor with breezy soft yacht rock silky smoothness',
    '댄포겔버그': 'delicate lyrical tenor with poetic graceful folk melodic phrasing',
    '댄 포겔버그': 'delicate lyrical tenor with poetic graceful folk melodic phrasing',
    '호세펠레시아노': 'passionate tender tenor with soulful Latin folk vibrato and longing',
    '호세 펠레시아노': 'passionate tender tenor with soulful Latin folk vibrato and longing',
    '파사신저': 'plaintive soaring falsetto with vulnerable intimate indie folk ache',
    '밴모리슨': 'gravelly soulful midrange with Celtic mystical stream-of-consciousness vocal',
    '밴 모리슨': 'gravelly soulful midrange with Celtic mystical stream-of-consciousness vocal',
    '숀멘데스': 'youthful clear tenor with earnest modern acoustic pop vulnerability',
    '숀 멘데스': 'youthful clear tenor with earnest modern acoustic pop vulnerability',

    # ── Global Male: Orchestral/Cinematic Ballad ──
    '안드레아보첼리': 'majestic operatic tenor with soaring classical purity and grand resonance',
    '안드레아 보첼리': 'majestic operatic tenor with soaring classical purity and grand resonance',
    '조쉬그로반': 'powerful rich baritone-tenor with sweeping orchestral epic vocal scale',
    '조쉬 그로반': 'powerful rich baritone-tenor with sweeping orchestral epic vocal scale',
    '훌리오이글레시아스': 'suave romantic baritone with elegant continental orchestral pop charm',
    '훌리오 이글레시아스': 'suave romantic baritone with elegant continental orchestral pop charm',
    '빙크로스비': 'velvety deep crooning baritone with effortless vintage big band warmth',
    '빙 크로스비': 'velvety deep crooning baritone with effortless vintage big band warmth',
    '페리코모': 'relaxed mellow baritone with comfortable lush string-backed vocal ease',
    '페리 코모': 'relaxed mellow baritone with comfortable lush string-backed vocal ease',
    '바비빈튼': 'tender longing falsetto carrying sorrowful romantic feel over strings',
    '바비 빈튼': 'tender longing falsetto carrying sorrowful romantic feel over strings',
    '맷먼로': 'polished suave British tenor with sophisticated cinematic ballad phrasing',
    '맷 먼로': 'polished suave British tenor with sophisticated cinematic ballad phrasing',
    '알마티노': 'warm robust tenor with grand sweeping romantic pop balladry and passion',
    '알 마티노': 'warm robust tenor with grand sweeping romantic pop balladry and passion',
    '엥겔베르트험퍼딩크': 'powerful sweeping baritone with lush romantic orchestral vocal projection',
    '엥겔베르트 험퍼딩크': 'powerful sweeping baritone with lush romantic orchestral vocal projection',
    '조니마티스': 'elegant silken high tenor with refined classical pop gentle sophistication',
    '조니 마티스': 'elegant silken high tenor with refined classical pop gentle sophistication',
    '해리벨라폰테': 'warm charismatic baritone with calypso-tinged folk orchestral humanity',
    '해리 벨라폰테': 'warm charismatic baritone with calypso-tinged folk orchestral humanity',
    '알레한드로산스': 'passionate raspy tenor with intense emotional Latin cinematic depth',
    '알레한드로 산스': 'passionate raspy tenor with intense emotional Latin cinematic depth',
    '루이스미겔': 'radiant smooth tenor with luminous Latin pop brass-backed vocal glow',
    '루이스 미겔': 'radiant smooth tenor with luminous Latin pop brass-backed vocal glow',
    '일디보': 'blended operatic tenor ensemble with lush popera orchestral harmony',
    '일 디보': 'blended operatic tenor ensemble with lush popera orchestral harmony',
    '포스트말론': 'warm melancholic midrange with genre-fluid modern cinematic melodic feel',
    '포스트 말론': 'warm melancholic midrange with genre-fluid modern cinematic melodic feel',
    '핀네이어': 'intimate whispery baritone with atmospheric mysterious cinematic texture',
    '해리스타일스': 'charismatic retro tenor with vintage seventies pop-rock warmth and flair',
    '해리 스타일스': 'charismatic retro tenor with vintage seventies pop-rock warmth and flair',
    '로비윌리엄스': 'bold theatrical baritone with brassy big band showman vocal energy',
    '로비 윌리엄스': 'bold theatrical baritone with brassy big band showman vocal energy',

    # ── Global Female: Piano-Based Ballad ──
    '캐롤킹': 'warm earthy alto with tender piano-driven storytelling delivery',
    '캐롤 킹': 'warm earthy alto with tender piano-driven storytelling delivery',
    '사라맥라클란': 'ethereal crystalline soprano with gentle breathy healing resonance',
    '사라 맥라클란': 'ethereal crystalline soprano with gentle breathy healing resonance',
    '토니브랙스턴': 'deep velvety contralto with smoldering sultry low-register warmth',
    '토니 브랙스턴': 'deep velvety contralto with smoldering sultry low-register warmth',
    '바네사칼튼': 'bright youthful soprano with bouncy melodic piano-pop energy',
    '바네사 칼튼': 'bright youthful soprano with bouncy melodic piano-pop energy',
    '피오나애플': 'raw angular mezzo with unconventional phrasing and emotional grit',
    '피오나 애플': 'raw angular mezzo with unconventional phrasing and emotional grit',
    '토리에이모스': 'intense dramatic soprano with overwhelming ethereal vibrato power',
    '토리 에이모스': 'intense dramatic soprano with overwhelming ethereal vibrato power',
    '레지나스펙터': 'quirky playful soprano with witty theatrical vocal leaps',
    '레지나 스펙터': 'quirky playful soprano with witty theatrical vocal leaps',
    '레이첼플래튼': 'transparent earnest soprano with heartfelt uplifting pop clarity',
    '레이첼 플래튼': 'transparent earnest soprano with heartfelt uplifting pop clarity',
    '라나델레이': 'dreamy breathy contralto with cinematic nostalgic melancholy tone',
    '라나 델 레이': 'dreamy breathy contralto with cinematic nostalgic melancholy tone',
    '버디': 'delicate airy soprano with tender indie falsetto fragility',
    '사라바렐리스': 'warm versatile mezzo with theatrical pop crossover phrasing',
    '사라 바렐리스': 'warm versatile mezzo with theatrical pop crossover phrasing',
    '델타굿렘': 'clear pristine soprano with polished clean piano ballad projection',
    '델타 굿렘': 'clear pristine soprano with polished clean piano ballad projection',
    '니콜크로아질': 'refined French chanteuse with classic cinematic vocal elegance',
    '니콜 크로아질': 'refined French chanteuse with classic cinematic vocal elegance',
    '디아나크롤': 'smoky low alto with intimate atmospheric jazz vocal phrasing',
    '디아나 크롤': 'smoky low alto with intimate atmospheric jazz vocal phrasing',

    # ── Global Female: Power/Rock Ballad ──
    '켈리클락슨': 'explosive powerhouse belter with massive high-register vocal force',
    '켈리 클락슨': 'explosive powerhouse belter with massive high-register vocal force',
    '앤윌슨': 'soaring rock soprano with legendary high-range stadium power',
    '앤 윌슨': 'soaring rock soprano with legendary high-range stadium power',
    '셰어': 'bold deep contralto with distinctive vibrato and rock-pop grit',
    '핑크': 'rough raspy alto with raw heartfelt rock ballad intensity',
    '데미로바토': 'piercing powerful soprano with raw high-note emotional belting',
    '데미 로바토': 'piercing powerful soprano with raw high-note emotional belting',
    '알라니스모리셋': 'angsty alternative mezzo with confessional emotional vocal rawness',
    '알라니스 모리셋': 'angsty alternative mezzo with confessional emotional vocal rawness',
    '플로렌스웰치': 'massive operatic soprano with stadium-shaking dramatic vocal volume',
    '플로렌스 웰치': 'massive operatic soprano with stadium-shaking dramatic vocal volume',
    '리오나루이스': 'crystalline soaring soprano with gradual power ballad buildup',
    '리오나 루이스': 'crystalline soaring soprano with gradual power ballad buildup',
    '에이브릴라빈': 'edgy youthful alto with emotional pop-rock ballad vulnerability',
    '에이브릴 라빈': 'edgy youthful alto with emotional pop-rock ballad vulnerability',

    # ── Global Female: R&B/Soul Ballad ──
    '조스스톤': 'gritty soulful groove vocal with British white-soul rasp',
    '조스 스톤': 'gritty soulful groove vocal with British white-soul rasp',
    '메리제이블라이즈': 'commanding alto with passionate 90s ballad grit',
    '메리 제이 블라이즈': 'commanding alto with passionate 90s ballad grit',
    '로바타플랙': 'sophisticated silky soprano with elegant understated soul phrasing',
    '로바타 플랙': 'sophisticated silky soprano with elegant understated soul phrasing',
    '솔란지': 'airy alternative R&B vocal with modern textured tension',
    '디온워윅': 'polished warm soprano with elegant 60s sophisticated pop soul tone',
    '디온 워윅': 'polished warm soprano with elegant 60s sophisticated pop soul tone',
    '인디아아리': 'healing earthy mezzo with gentle acoustic soul warmth',
    '인디아 아리': 'healing earthy mezzo with gentle acoustic soul warmth',
    '에이치이알': 'modern layered R&B alto with dense harmonic tension and depth',
    '허': 'modern layered R&B alto with dense harmonic tension and depth',
    '자스민설리반': 'husky powerful soprano with overwhelming melismatic soul technique',
    '자스민 설리반': 'husky powerful soprano with overwhelming melismatic soul technique',
    '시저': 'breathy syncopated alto with unique rhythmic modern R&B flow',
    '나탈리콜': 'polished velvety mezzo with jazz-pop R&B fusion warmth',
    '나탈리 콜': 'polished velvety mezzo with jazz-pop R&B fusion warmth',
    '글래디스나이트': 'rich commanding contralto with majestic traditional soul phrasing',
    '글래디스 나이트': 'rich commanding contralto with majestic traditional soul phrasing',

    # ── Global Female: Acoustic/Folk Ballad ──
    '조니미첼': 'sophisticated folk soprano with jazz-inflected phrasing and soaring melodic range',
    '조니 미첼': 'sophisticated folk soprano with jazz-inflected phrasing and soaring melodic range',
    '조안바에즈': 'pure crystal-clear folk soprano with gentle vibrato and unwavering clarity',
    '조안 바에즈': 'pure crystal-clear folk soprano with gentle vibrato and unwavering clarity',
    '트레이시채프먼': 'deep warm contralto folk vocal with soulful resonant storytelling tone',
    '트레이시 채프먼': 'deep warm contralto folk vocal with soulful resonant storytelling tone',
    '주디콜린스': 'transparent dewdrop-clear soprano with pristine folk articulation and purity',
    '주디 콜린스': 'transparent dewdrop-clear soprano with pristine folk articulation and purity',
    '수잔베가': 'cool understated spoken-word folk vocal with minimal acoustic intimacy',
    '수잔 베가': 'cool understated spoken-word folk vocal with minimal acoustic intimacy',
    '주얼': 'delicate yodeling folk soprano with breathy acoustic pop tenderness',
    '코린베일리래': 'warm honeyed neo-soul acoustic vocal with gentle healing softness',
    '코린 베일리 래': 'warm honeyed neo-soul acoustic vocal with gentle healing softness',
    '테일러스위프트': 'bright narrative acoustic pop vocal with heartfelt country-folk sincerity',
    '테일러 스위프트': 'bright narrative acoustic pop vocal with heartfelt country-folk sincerity',
    '콜비카레이': 'breezy light acoustic vocal with sunny relaxed beach pop warmth',
    '콜비 카레이': 'breezy light acoustic vocal with sunny relaxed beach pop warmth',
    '케이디랭': 'rich velvety contralto with breathtaking sustained ballad phrasing and control',
    '케이디 랭': 'rich velvety contralto with breathtaking sustained ballad phrasing and control',
    '피비브리저스': 'hushed melancholic indie folk vocal with haunting mournful fragility',
    '피비 브리저스': 'hushed melancholic indie folk vocal with haunting mournful fragility',
    '케이시머스그레이브스': 'smooth modern country soprano with dreamy spacious acoustic pop glow',
    '케이시 머스그레이브스': 'smooth modern country soprano with dreamy spacious acoustic pop glow',
    '카를라브루니': 'whispery intimate French chanson vocal with featherlight guitar murmur',
    '카를라 브루니': 'whispery intimate French chanson vocal with featherlight guitar murmur',
    '케이티턴스탈': 'rhythmic percussive acoustic vocal with energetic one-woman-band drive',
    '케이티 턴스탈': 'rhythmic percussive acoustic vocal with energetic one-woman-band drive',
    '에밀루해리스': 'silvery celestial country harmony soprano with aching folk purity',
    '에밀루 해리스': 'silvery celestial country harmony soprano with aching folk purity',
    '디도': 'soft dreamy ethereal vocal with understated electronic-acoustic melancholy',

    # ── Global Female: Orchestral/Cinematic Ballad ──
    '엔야': 'layered multitrack choral vocal creating vast cosmic ethereal soundscape',
    '사라브라이트만': 'soaring popera soprano with theatrical cinematic grandeur and power',
    '사라 브라이트만': 'soaring popera soprano with theatrical cinematic grandeur and power',
    '셜리바시': 'commanding dramatic diva vocal with explosive big-band cinematic delivery',
    '셜리 바시': 'commanding dramatic diva vocal with explosive big-band cinematic delivery',
    '헤일리웨스텐라': 'pristine classical crossover soprano with angelic symphonic serenity',
    '헤일리 웨스텐라': 'pristine classical crossover soprano with angelic symphonic serenity',
    '라라파비앙': 'emotionally explosive belting soprano with dramatic orchestral climax power',
    '라라 파비앙': 'emotionally explosive belting soprano with dramatic orchestral climax power',
    '레아살롱가': 'crystalline narrative soprano with warm theatrical storytelling clarity',
    '레아 살롱가': 'crystalline narrative soprano with warm theatrical storytelling clarity',
    '켈틱우먼': 'angelic Irish ensemble soprano with soaring Celtic harmony and purity',
    '클로에애그뉴': 'angelic Irish ensemble soprano with soaring Celtic harmony and purity',
    '재키에반코': 'young luminous classical crossover soprano with operatic innocence and grace',
    '재키 에반코': 'young luminous classical crossover soprano with operatic innocence and grace',
    '샤론덴아델': 'delicate symphonic metal soprano piercing through heavy orchestral layers',
    '샤론 덴 아델': 'delicate symphonic metal soprano piercing through heavy orchestral layers',
    '빌리에일리시': 'whispery intimate ASMR vocal with dark minimal cinematic atmosphere',
    '빌리 에일리시': 'whispery intimate ASMR vocal with dark minimal cinematic atmosphere',
    '아그네스오벨': 'haunting atmospheric mezzo-soprano with dreamy cinematic piano textures',
    '아그네스 오벨': 'haunting atmospheric mezzo-soprano with dreamy cinematic piano textures',
    '오로라': 'ethereal Nordic soprano with nature-inspired orchestral folk grandeur',
    '시아': 'raw explosive belting vocal tearing through massive dramatic string arrangements',

    # ── 🇰🇷 한국 남자: Piano-Based Ballad ──
    '신승훈': 'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone',
    '김동률': 'deep grand baritone with rich harmonic resonance and classical vocal control',
    '성시경': 'soft gentle tenor with intimate breathy delivery and warm melodic phrasing',
    '유희열': 'sophisticated mid-range vocal with refined phrasing and understated elegance',
    '정재형': 'lyrical light tenor with airy French-pop-influenced delicate vocal texture',
    '김광진': 'comforting warm baritone with steady timeless phrasing and soothing resonance',
    '윤건': 'smooth urban tenor with silky tone and polished melodic vocal delivery',
    '정승환': 'pure crystalline tenor with emotionally transparent and heartfelt vocal delivery',
    '박원': 'honest deep tenor with raw sincerity and quietly compelling melodic appeal',
    '곽진언': 'calm low mid-range voice with grounded emotional stability and gentle warmth',

    # ── 🇰🇷 한국 남자: Power/Rock Ballad ──
    '김경호': 'screaming high tenor with razor-sharp power and explosive sustained notes',
    '박완규': 'sharp sorrowful tenor with intense vibrato and dramatic dynamic vocal range',
    '김종서': 'pioneering rock tenor with soaring falsetto mastery and gritty edge',
    '김장훈': 'husky theatrical baritone with bold unique projection and dramatic flair',
    '윤도현': 'refreshing raspy rock tenor with commanding anthemic vocal energy',
    '정홍일': 'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit',
    '김민우': 'romantic emotional tenor with soaring rock-ballad phrasing and soft power',

    # ── 🇰🇷 한국 남자: R&B/Soul Ballad ──
    '이승철': 'transcendent tenor with flawless breath control and effortless vocal mastery',
    '바비킴': 'unique husky voice blending soul reggae groove with inflection',
    '김조한': 'smooth R&B tenor with layered harmony skill and soulful melodic texture',
    '휘성': 'addictive husky mid-range with rhythmic R&B top-line vocal precision',
    '케이윌': 'versatile clear tenor with explosive high notes and agile R&B runs',
    '환희': 'rich deep mid-low voice with velvety soul resonance and warm delivery',
    '크러쉬': 'modern breathy alternative R&B vocal with trendy melodic sensibility',

    # ── 🇰🇷 한국 남자: Acoustic/Folk Ballad ──
    '변진섭': 'earnest warm tenor with pure heartfelt delivery and classic ballad phrasing',
    '윤종신': 'natural conversational mid-range voice with realistic folk storytelling warmth',
    '유재하': 'elegant refined tenor with classically elevated harmonic vocal phrasing',
    '최백호': 'romantic aged baritone with weathered folk warmth and nostalgic resonance',
    '이한철': 'bright warm tenor with uplifting acoustic energy and positive vocal clarity',
    '권정열': 'distinctive nasal indie tenor with quirky charm and acoustic pop character',
    '10CM': 'distinctive nasal indie tenor with quirky charm and acoustic pop character',

    # ── 🇰🇷 한국 남자: Orchestral/Cinematic Ballad ──
    '조성모': 'sweeping dramatic tenor with lush symphonic phrasing and soaring crescendos',
    '이승환': 'perfectionist tenor with powerful live projection and orchestral vocal precision',
    '임영웅': 'modern healing tenor with gentle crossover appeal and wide emotional range',
    '김연우': 'flawless crystal clear tenor cutting through full orchestral arrangements cleanly',
    '정국': 'global trendy tenor with cinematic pop polish and youthful dynamic range',
    '황치열': 'powerful husky high tenor with dramatic intensity and piercing climactic notes',
    '테이': 'stable rich baritone with sweeping epic phrasing and cinematic vocal depth',
    'KCM': 'airy mixed-voice high tenor soaring with delicate falsetto over strings',

    # ── 🇰🇷 한국 여자: Piano-Based Ballad ──
    '이소라': 'literary poetic mezzo-soprano with healing resonance and uniquely textured warmth',
    '양희은': 'pure clean soprano capturing quiet depth with timeless graceful phrasing',
    '민서': 'youthful crystalline soprano with emotionally transparent pure ballad delivery',
    '린': 'natural nasal-toned mezzo wrapping melodies with tender breathy intimacy',
    '안예은': 'creative fusion soprano blending traditional color with modern tone',
    '이진아': 'inventive jazzy soprano with playful harmonic twists and whimsical vocal texture',

    # ── 🇰🇷 한국 여자: Power/Rock Ballad ──
    '서문탁': 'powerful husky alto with commanding rock intensity and fierce stage presence',
    '박기영': 'multi-genre soprano with piercing high notes and relentless vocal stamina',
    '김현정': 'refreshing bright rock soprano with crisp attack and vibrant power delivery',
    '도원경': 'pioneering fierce female rock alto with raw gritty emotional conviction',
    '마야': 'raw desperate soprano with unfiltered emotional intensity and urgent vocal power',
    '손승연': 'monster-vocal soprano with devastating power and next-generation explosive technique',
    '에일리': 'explosive belting soprano with piercing volume and electrifying high-note precision',
    '정은지': 'refreshing clear soprano with surprisingly powerful ballad projection and warmth',

    # ── 🇰🇷 한국 여자: R&B/Soul Ballad ──
    '화요비': 'authentic R&B alto with golden-era groove and smooth soulful vocal runs',
    '이영현': 'soulful mezzo-soprano with flawless R&B scale technique and rich harmony',
    '정인': 'uniquely textured alto with rare soulful groove and smoky warmth',
    '헤이즈': 'trendy urban mezzo with tension-filled chord sensibility and sultry phrasing',
    '박봄': 'globally distinctive soprano with unique nasal R&B color and emotional crack',
    '이하이': 'deep soul-laden mezzo-alto with mature tone and moody R&B depth',
    '선우정아': 'genre-bending refined alto with jazz-soul sophistication and avant-garde phrasing',
    '효린': 'rhythmic powerhouse soprano with Black-music-influenced soulful R&B agility',
    '비비': 'captivating syncopated mezzo with seductive breathy storytelling and playful tone',

    # ── 🇰🇷 한국 여자: Acoustic/Folk Ballad ──
    '볼빨간사춘기': 'unique bright indie soprano with youthful acoustic charm and catchy phrasing',
    '안지영': 'unique bright indie soprano with youthful acoustic charm and catchy phrasing',
    '장필순': 'ethereal breathtaking soprano with deep folk wisdom and transcendent vocal purity',
    '조원선': 'refined effortless alto with minimal urban folk elegance and cool restraint',
    '한희정': 'serene quiet soprano with gentle indie folk healing warmth and calm clarity',
    '치즈': 'cute warm soprano with sweet acoustic delivery and endearing soft vocal tone',
    '요조': 'bright pure indie soprano with cheerful Hongdae folk energy and lightness',
    '타루': 'sweet lyrical soprano with delicate acoustic phrasing and tender emotional color',
    '박새별': 'warm organic mezzo with earthy folk resonance and natural singer-songwriter tone',
    '스텔라장': 'minimal clear soprano with global folk sensibility and understated guitar-pop grace',

    # ── 🇰🇷 한국 여자: Orchestral/Cinematic Ballad ──
    '조수미': 'world-class soprano with soaring operatic power and pristine cinematic projection',
    '윤하': 'explosive narrative soprano with grand rock-symphonic intensity and dramatic arc',
    '다비치': 'devastating power-ballad soprano tearing through lush string arrangements emotionally',
    '이해리': 'devastating power-ballad soprano tearing through lush string arrangements emotionally',
    '서영은': 'pristine clear soprano with ethereal purity suited for epic dramatic ballads',
    '왁스': 'emotionally charged soprano with blockbuster string-ballad intensity and raw feeling',
    '임정희': 'street-style diva soprano with massive volume and bold orchestral vocal presence',
    '알리': 'massive cinematic soprano dominating choir and symphony with towering vocal force',
    # ═══════════════════════════════════════════════════════════
    # cgo-383: 힙합 남자 보컬 100명
    # ═══════════════════════════════════════════════════════════
    # ── 🥁 정통 붐뱁 & 동부 힙합 (Boom-Bap / East Coast) ──
    '노토리어스 비아이쥐': 'heavyweight commanding male vocalists with flawless flow and deep groove mastery',
    '노토리어스비아이쥐': 'heavyweight commanding male vocalists with flawless flow and deep groove mastery',
    '나스': 'poetic conscious male vocalists with precise lyrical craftsmanship and street narrative depth',
    '제이 지': 'authoritative smooth male vocalists with business-mogul swagger and effortless delivery',
    '제이지': 'authoritative smooth male vocalists with business-mogul swagger and effortless delivery',
    '라킴': 'pioneering male vocalists who defined modern rhyme schemes with meticulous cadence',
    '에미넴': 'explosive rapid-fire male vocalists with razor-sharp diction and unmatched global impact',
    '디엠엑스': 'raw gravelly male vocalists with ferocious barking energy and intense emotional delivery',
    '빅 엘': 'razor-sharp punchline male vocalists with dazzling lyrical acrobatics and tragic brilliance',
    '빅엘': 'razor-sharp punchline male vocalists with dazzling lyrical acrobatics and tragic brilliance',
    '메서드 맨': 'distinctive husky low-tone male vocalists with gritty charismatic flow and swagger',
    '메서드맨': 'distinctive husky low-tone male vocalists with gritty charismatic flow and swagger',
    '레드맨': 'funky freewheeling male vocalists with raw unfiltered energy on beats',
    '모스 뎁': 'intellectually refined male vocalists blending conscious lyricism with jazz- soul',
    '모스뎁': 'intellectually refined male vocalists blending conscious lyricism with jazz- soul',
    '탈립 퀠리': 'cerebral eloquent male vocalists elevating artistry with intricate rhyme patterns',
    '탈립퀠리': 'cerebral eloquent male vocalists elevating artistry with intricate rhyme patterns',
    '스릭 릭': 'legendary storytelling male vocalists with unique accent and theatrical narrative flow',
    '스릭릭': 'legendary storytelling male vocalists with unique accent and theatrical narrative flow',
    '엘엘 쿨 제이': 'pioneering male vocalists mastering both hardcore and smooth love-',
    '엘엘쿨제이': 'pioneering male vocalists mastering both hardcore and smooth love-',
    '빅 푸니셔': 'relentless rapid-fire male vocalists with massive projection and dominance',
    '빅푸니셔': 'relentless rapid-fire male vocalists with massive projection and dominance',
    '빅 대디 케인': 'fleet-footed male vocalists with dazzling speed and showmanship from the golden era',
    '빅대디케인': 'fleet-footed male vocalists with dazzling speed and showmanship from the golden era',
    '에이셉 라키': 'trendy stylish male vocalists layering fashion-forward aesthetics over New York',
    '에이셉라키': 'trendy stylish male vocalists layering fashion-forward aesthetics over New York',
    '크리스 크로스': 'authoritative male vocalists who defined philosophy with intellectual power',
    '크리스크로스': 'authoritative male vocalists who defined philosophy with intellectual power',
    '빅 원': 'underground gritty male vocalists with raw sensibility and street authenticity',
    '빅원': 'underground gritty male vocalists with raw sensibility and street authenticity',
    '구루': 'monotone mid-bass male vocalists fusing jazz harmonics with groove seamlessly',
    '조이 배드애스': 'modern male vocalists perfectly reviving 90s golden-era New York aesthetics',
    '조이배드애스': 'modern male vocalists perfectly reviving 90s golden-era New York aesthetics',
    # ── 🌴 웨스트 코스트 & 지펑크 (West Coast / G-Funk) ──
    '투팍 샤커': 'passionate revolutionary male vocalists with soul-stirring delivery and poetic intensity',
    '투팍샤커': 'passionate revolutionary male vocalists with soul-stirring delivery and poetic intensity',
    '투팍': 'passionate revolutionary male vocalists with soul-stirring delivery and poetic intensity',
    '스눕 독': 'silky laid-back male vocalists with signature drawl and effortless west-coast groove',
    '스눕독': 'silky laid-back male vocalists with signature drawl and effortless west-coast groove',
    '닥터 드레': 'authoritative deep male vocal who architected G-Funk sound and discovered legends',
    '닥터드레': 'authoritative deep male vocal who architected G-Funk sound and discovered legends',
    '아이스 큐브': 'aggressive hard-hitting male vocalists with menacing gangsta delivery and sharp wit',
    '아이스큐브': 'aggressive hard-hitting male vocalists with menacing gangsta delivery and sharp wit',
    '켄드릭 라마': 'virtuoso male vocalists with chameleonic vocal range and Pulitzer-level storytelling',
    '켄드릭라마': 'virtuoso male vocalists with chameleonic vocal range and Pulitzer-level storytelling',
    '더 게임': 'rugged male vocalists blending gritty tone with gangster balladry and west-coast soul',
    '더게임': 'rugged male vocalists blending gritty tone with gangster balladry and west-coast soul',
    '이지 이': 'distinctive high-pitched male vocalists gangsta rap with piercing tone',
    '이지이': 'distinctive high-pitched male vocalists gangsta rap with piercing tone',
    '네이트 독': 'melodic hook-master male vocalist who defined G-Funk with soulful singing-rap fusion',
    '네이트독': 'melodic hook-master male vocalist who defined G-Funk with soulful singing-rap fusion',
    '워렌 지': 'smooth romantic male vocalists delivering the most mellow G-Funk with laid-back flow',
    '워렌지': 'smooth romantic male vocalists delivering the most mellow G-Funk with laid-back flow',
    '엑지빗': 'powerfully raspy male vocalists with aggressive west-coast punch and commanding presence',
    '쿨리오': 'chart-dominating male vocalists with infectious melodic hooks and global crossover appeal',
    '사이프레스 힐': 'nasal high-pitched male vocalists Latin west-coast uniquely',
    '사이프레스힐': 'nasal high-pitched male vocalists Latin west-coast uniquely',
    '맥 텐': 'hard-hitting gangster male vocalists delivering textbook west-coast hardcore with authority',
    '맥텐': 'hard-hitting gangster male vocalists delivering textbook west-coast hardcore with authority',
    '디제이 퀵': 'multi-talented male vocalist-producer with refined west-coast lyricism and groove mastery',
    '디제이퀵': 'multi-talented male vocalist-producer with refined west-coast lyricism and groove mastery',
    '쿠럽': 'technically gifted male vocalists with extraordinary rhyme arrangement on west-coast beats',
    '대즈 딜린저': 'deep-voiced male vocalist-producer who powered Death Row Records golden era sound',
    '대즈딜린저': 'deep-voiced male vocalist-producer who powered Death Row Records golden era sound',
    '엠씨 아이트': 'cold atmospheric male vocalists delivering chilling street narratives with west-coast cool',
    '엠씨아이트': 'cold atmospheric male vocalists delivering chilling street narratives with west-coast cool',
    '스쿨보이 큐': 'gritty groovy modern male vocalists with aggressive delivery on contemporary west-coast',
    '스쿨보이큐': 'gritty groovy modern male vocalists with aggressive delivery on contemporary west-coast',
    '타이 달라 사인': 'versatile male vocalist seamlessly fusing R&B over laid-back west-coast beats',
    '타이달라사인': 'versatile male vocalist seamlessly fusing R&B over laid-back west-coast beats',
    '비지 본': 'lightning-fast male vocalists layering angelic melodies over rapid-fire delivery uniquely',
    '비지본': 'lightning-fast male vocalists layering angelic melodies over rapid-fire delivery uniquely',
    # ── 🔥 서던 힙합 & 트랩 (Southern / Trap) ──
    '티아이': 'commanding male vocalists who coined trap music with authoritative Atlanta swagger',
    '구찌 메인': 'foundational male vocalists who architected modern trap culture and sound',
    '구찌메인': 'foundational male vocalists who architected modern trap culture and sound',
    '릴 웨인': 'inventive genius male vocalists with unique voice and otherworldly metaphorical wordplay',
    '릴웨인': 'inventive genius male vocalists with unique voice and otherworldly metaphorical wordplay',
    '퓨처': 'autotune-wielding male vocalists who perfected modern melodic trap with hypnotic delivery',
    '트래비스 콧': 'psychedelic male vocalists fusing rock energy with stadium-shaking trap production',
    '트래비스콧': 'psychedelic male vocalists fusing rock energy with stadium-shaking trap production',
    '영 턱': 'revolutionary male vocalists who weaponized his voice as an instrument destroying conventions',
    '영턱': 'revolutionary male vocalists who weaponized his voice as an instrument destroying conventions',
    '빅 보이': 'rapid-fire southern male vocalists elevating OutKast with virtuoso technical precision',
    '빅보이': 'rapid-fire southern male vocalists elevating OutKast with virtuoso technical precision',
    '안드레 3000': 'wildly innovative southern male vocalists and genre most artistic genre-bending voice',
    '안드레3000': 'wildly innovative southern male vocalists and genre most artistic genre-bending voice',
    '루다크리스': 'hard-hitting precise male vocalists with thunderous diction and humorous southern delivery',
    '릭 로스': 'heavyweight deep-bass male vocalists dominating grandiose trap beats with boss authority',
    '릭로스': 'heavyweight deep-bass male vocalists dominating grandiose trap beats with boss authority',
    '퀘이보': 'triplet-flow male vocalists who rewrote global trap standards with infectious cadence',
    '오프셋': 'technically sharp male vocalists with rapid-fire triplet flow and aggressive punch',
    '테이크오프': 'precise surgical male vocalists with clean triplet delivery and understated excellence',
    '릴 존': 'explosive crunk-pioneer male vocalists with earth-shaking club energy and powerful vocals',
    '릴존': 'explosive crunk-pioneer male vocalists with earth-shaking club energy and powerful vocals',
    '쥬시 제이': 'underground legend male vocalists who built southern hardcore trap skeletal framework',
    '쥬시제이': 'underground legend male vocalists who built southern hardcore trap skeletal framework',
    '미스틱칼': 'thunderous military-grade male vocalists with overwhelming volume and raw southern energy',
    '투 체인즈': 'addictive punchline male vocalists with playful charisma and infectious trap mastery',
    '투체인즈': 'addictive punchline male vocalists with playful charisma and infectious trap mastery',
    '지 지': 'husky gravelly male vocalists delivering authentic Atlanta street narratives with grit',
    '지지': 'husky gravelly male vocalists delivering authentic Atlanta street narratives with grit',
    '릴 바비': 'precision-engineered modern male vocalists with relentless continuous trap flow dominance',
    '릴바비': 'precision-engineered modern male vocalists with relentless continuous trap flow dominance',
    '건나': 'silky sliding male vocalists perfecting melodic trap with fluid effortless delivery',
    '맥클모어': 'accessible narrative male vocalists layering popular storytelling over southern-style production',
    '21 새비지': 'ice-cold monotone male vocalists embodying modern dark trap with deadpan delivery',
    '21새비지': 'ice-cold monotone male vocalists embodying modern dark trap with deadpan delivery',
    # ── 💫 21세기 하이브리드 & 얼터너티브 (Alternative / Hybrid) ──
    '카니예 웨스트': 'paradigm-shifting male vocalist-producer and genre greatest sonic innovator ever',
    '카니예웨스트': 'paradigm-shifting male vocalist-producer and genre greatest sonic innovator ever',
    '드레이크': 'chart-dominating male vocalists-singer who demolished the rap-R&B boundary worldwide',
    '키드 커디': 'dreamy alternative male vocalists who implanted psychedelic rock sensibility into',
    '키드커디': 'dreamy alternative male vocalists who implanted psychedelic rock sensibility into',
    '릴 우지 버트': 'emo-rock infused male vocalists fusing emotional intensity with rapid hi-hat trap',
    '릴우지버트': 'emo-rock infused male vocalists fusing emotional intensity with rapid hi-hat trap',
    '주스 월드': 'heart-wrenching melodic male vocalists who epitomized emo-rap with devastating melodies',
    '주스월드': 'heart-wrenching melodic male vocalists who epitomized emo-rap with devastating melodies',
    '릴 피프': 'punk-rock crossover male vocalists who perfectly blended hardcore punk with trap',
    '릴피프': 'punk-rock crossover male vocalists who perfectly blended hardcore punk with trap',
    '타일러 더 크리에이터': 'eccentric brilliant male vocalists commanding neo-soul alternative with vision',
    '타일러더크리에이터': 'eccentric brilliant male vocalists commanding neo-soul alternative with vision',
    '에이셉 퍼그': 'energetic flashy male vocalists with wild hybrid flow over hardcore club beats',
    '에이셉퍼그': 'energetic flashy male vocalists with wild hybrid flow over hardcore club beats',
    '차일디시 감비노': 'multi-talented male vocalists-actor delivering socially charged alternative masterfully',
    '차일디시감비노': 'multi-talented male vocalists-actor delivering socially charged alternative masterfully',
    '포스트 말론': 'genre-blending male vocalist fusing rock, country, and trap with healing timbre',
    '포스트말론': 'genre-blending male vocalist fusing rock, country, and trap with healing timbre',
    '엑스엑스엑스텐타시온': 'versatile raw male vocalists spanning distorted lo-fi beats to tender acoustic rap',
    '플레이보이 카티': 'hypnotic baby-voice male vocalists commanding rave-trap with addictive minimalist flow',
    '플레이보이카티': 'hypnotic baby-voice male vocalists commanding rave-trap with addictive minimalist flow',
    '맥 밀러': 'soulful jazzy male vocalists weaving indie rock and jazz into beautifully',
    '맥밀러': 'soulful jazzy male vocalists weaving indie rock and jazz into beautifully',
    '로직': 'rapid-fire technical male vocalists balancing speed with accessible pop-ballad sensibility',
    '지이지': 'polished male vocalists with clean jazz-synthpop beats and sophisticated hybrid delivery',
    '와이클리프 장': 'Caribbean-fusion male vocalists crossing reggae, Latin, and with 90s mastery',
    '와이클리프장': 'Caribbean-fusion male vocalists crossing reggae, Latin, and with 90s mastery',
    '비오비': 'pop-rock acoustic male vocalists who conquered Billboard with accessible crossover sound',
    '자 룰': 'husky passionate male vocalists love-rap duets with R&B singers globally',
    '자룰': 'husky passionate male vocalists love-rap duets with R&B singers globally',
    '플로 라이다': 'EDM-fused male vocalists dominating global clubs with electronic dance-rap energy',
    '플로라이다': 'EDM-fused male vocalists dominating global clubs with electronic dance-rap energy',
    '위즈 칼리파': 'mellow melodic male vocalists with addictive hooks and relaxed hybrid ballad delivery',
    '위즈칼리파': 'mellow melodic male vocalists with addictive hooks and relaxed hybrid ballad delivery',
    # ── 🇬🇧 글로벌 영미권 & 그라임/드릴 (UK / Drill / Global) ──
    '팝 스모크': 'thunderous deep-cave male vocalists who exploded Brooklyn',
    '팝스모크': 'thunderous deep-cave male vocalists who exploded Brooklyn',
    '스켑타': 'razor-sharp UK grime male vocalists representing London streets with global authority',
    '스톰지': 'powerful UK national male vocalists fusing grime with classic soul harmonics brilliantly',
    '센트럴 씨': 'trend-setting male vocalists dominating global shorts and reels effortlessly',
    '센트럴씨': 'trend-setting male vocalists dominating global shorts and reels effortlessly',
    '제이훕': 'Afrobeat- fusion male vocalists Afroswing genre with infectious energy',
    '데이브': 'genius lyricist UK male vocalists delivering profound narratives over piano-driven beats',
    '긱스': 'slow heavyweight UK underground male vocalists with iconic deep bass flow delivery',
    '헤디 원': 'precisely polished male vocalists with the most refined rhythmic mastery',
    '헤디원': 'precisely polished male vocalists with the most refined rhythmic mastery',
    '디지 래스칼': 'revolutionary UK grime male vocalists who first brought grime to mainstream success',
    '디지래스칼': 'revolutionary UK grime male vocalists who first brought grime to mainstream success',
    '크로프트': 'wordplay-brilliant duo male vocalists with pop-friendly beat chemistry',
    '케난': 'Somali-born male vocalists fusing African traditional rhythms with uplifting spirit',
    '티제이': 'hard-hitting slide- male vocalists with powerful 808 bass-riding technique',
    '토니 레인즈': 'explosive Canadian male vocalists-singer with 80s-90s sampling and dynamic versatility',
    '토니레인즈': 'explosive Canadian male vocalists-singer with 80s-90s sampling and dynamic versatility',
    '나브': 'atmospheric Canadian male vocalists crafting dreamy synth-trap soundscapes with soft delivery',
    '아웃로우즈': 'commanding male vocalists carrying west-coast and with authority',
    '빅 주': 'heavyweight drum-and-bass male vocalists with signature UK grime flow mastery',
    '빅주': 'heavyweight drum-and-bass male vocalists with signature UK grime flow mastery',
    '아론 스미스': 'electronic-trap crossover male vocalists blending synths with global hybrid',
    '아론스미스': 'electronic-trap crossover male vocalists blending synths with global hybrid',
    '제이 콜': 'wise philosophical male vocalists layering classic soul with modern narratives',
    '제이콜': 'wise philosophical male vocalists layering classic soul with modern narratives',
    # ═══════════════════════════════════════════════════════════
    # cgo-383: 힙합 여자 보컬 100명
    # ═══════════════════════════════════════════════════════════
    # ── 🥁 정통 붐뱁 & 올드스쿨 전설 (Boom-Bap / Old-School) ──
    '미시 에일리엇': 'innovative female vocalist-producer with revolutionary visual and sonic mastery',
    '미시에일리엇': 'innovative female vocalist-producer with revolutionary visual and sonic mastery',
    '퀸 라티파': 'commanding female vocalists who elevated with social messages and artistic gravitas',
    '퀸라티파': 'commanding female vocalists who elevated with social messages and artistic gravitas',
    '엠씨 라이트': 'pioneering female vocalists who achieved the first solo gold album for women',
    '엠씨라이트': 'pioneering female vocalists who achieved the first solo gold album for women',
    '솔트': 'legendary 80s female vocalists who spearheaded mainstream with infectious energy',
    '페파': 'charismatic bold female vocalists who transformed genetics with playful authority',
    '록샌 섕테': 'battle-rap legend female vocalists with unmatched improvisational freestyle brilliance',
    '록샌섕테': 'battle-rap legend female vocalists with unmatched improvisational freestyle brilliance',
    '백시 미터': 'understated monotone female vocalists dominating 90s Philadelphia underground',
    '백시미터': 'understated monotone female vocalists dominating 90s Philadelphia underground',
    '모니 러브': 'witty transatlantic female vocalists with clever flow who defined an era gracefully',
    '모니러브': 'witty transatlantic female vocalists with clever flow who defined an era gracefully',
    '폭시 브라운': 'fierce sharp-tongued New York female vocalists with aggressive delivery',
    '폭시브라운': 'fierce sharp-tongued New York female vocalists with aggressive delivery',
    '라 비아': 'heavyweight hardcore female vocalists with solid powerful projection and grit',
    '라비아': 'heavyweight hardcore female vocalists with solid powerful projection and grit',
    '레이디 오브 레이지': 'explosive hardcore female vocalists who shredded 90s Death Row beats with fury',
    '레이디오브레이지': 'explosive hardcore female vocalists who shredded 90s Death Row beats with fury',
    '이브': 'fierce female vocalists from Ruff Ryders dominating 2000s with aggressive powerful flow',
    '리사 로페즈': 'legendary high-tone female vocalists with rhythmic agility and iconic vocal presence',
    '리사로페즈': 'legendary high-tone female vocalists with rhythmic agility and iconic vocal presence',
    '챰 브레이클리': 'intense 90s New York hardcore female vocalists with striking lyrical craftsmanship',
    '챰브레이클리': 'intense 90s New York hardcore female vocalists with striking lyrical craftsmanship',
    '진 그레이': 'underground technical female vocalists with the most intricate rhyme architecture',
    '진그레이': 'underground technical female vocalists with the most intricate rhyme architecture',
    '트리나': 'Miami hardcore female vocalists who built southern rap foundation with bold delivery',
    '갱스타 부': 'legendary Three 6 Mafia female vocalists ruling southern underground with fierce authority',
    '갱스타부': 'legendary Three 6 Mafia female vocalists ruling southern underground with fierce authority',
    '레이디 럭': 'skilled battle-rap female vocalists who shook early 2000s New York underground scene',
    '레이디럭': 'skilled battle-rap female vocalists who shook early 2000s New York underground scene',
    '랩소디': 'modern female vocalists praised by legends for supreme lyrical wisdom and craft',
    # ── 👑 빌보드 지배자 & 하드코어 퀸 (Hardcore & Billboard Queens) ──
    '니키 미나즈': 'rapid-fire versatile female vocalists with unmatched diction and genre-defining delivery',
    '니키미나즈': 'rapid-fire versatile female vocalists with unmatched diction and genre-defining delivery',
    '카디 비': 'raw powerful female vocalists conquering Billboard with aggressive unfiltered street energy',
    '카디비': 'raw powerful female vocalists conquering Billboard with aggressive unfiltered street energy',
    '메간 디 스탈리온': 'relentless southern female vocalists with heavyweight flow pounding hard-hitting beats',
    '메간디스탈리온': 'relentless southern female vocalists with heavyweight flow pounding hard-hitting beats',
    '릴 킴': 'trailblazing New York female vocalists who set fashion and lyrical standards for women',
    '릴킴': 'trailblazing New York female vocalists who set fashion and lyrical standards for women',
    '라토': 'polished Atlanta trap female vocalists with sophisticated hooks and confident swagger',
    '글로릴라': 'deep-voiced southern female vocalists with heavy 808 impact and raw visceral power',
    '아이스 스파이스': 'Bronx female vocalists who conquered global shorts with irresistible trend-setting',
    '아이스스파이스': 'Bronx female vocalists who conquered global shorts with irresistible trend-setting',
    '글로리아 마르티네즈': 'Latin Afro-beat female vocalists who shook Grammy stages with hardcore fusion vocals',
    '글로리아마르티네즈': 'Latin Afro-beat female vocalists who shook Grammy stages with hardcore fusion vocals',
    '레미 마': 'Terror Squad pride female vocalists with authentic New York hardcore dignity',
    '레미마': 'Terror Squad pride female vocalists with authentic New York hardcore dignity',
    '아잘리아 뱅크스': 'technically brilliant female vocalists who perfectly crosses house-EDM with hardcore rap',
    '아잘리아뱅크스': 'technically brilliant female vocalists who perfectly crosses house-EDM with hardcore rap',
    '이고 아잘리아': 'Australian-born female vocalists who topped Billboard charts with crossover hits',
    '이고아잘리아': 'Australian-born female vocalists who topped Billboard charts with crossover hits',
    '제이티': 'fierce Miami trap duo female vocalists with unrestrained lyrics and infectious groove',
    '융 마이애미': 'bouncy southern trap female vocalists optimized for hi-hat-driven hardcore delivery',
    '융마이애미': 'bouncy southern trap female vocalists optimized for hi-hat-driven hardcore delivery',
    '빅 보스 벨라': 'viral hook-machine female vocalists crafting addictive trap hooks for social media',
    '빅보스벨라': 'viral hook-machine female vocalists crafting addictive trap hooks for social media',
    '소이티': 'glamorous west-coast female vocalists delivering lifestyle bars over stylish sampled beats',
    '도치': 'next-generation hardcore female vocalists excelling at rap, singing, and stage performance',
    '플로 밀리': 'sharp high-tone female vocalists with clever off-beat flow and chart-penetrating diction',
    '플로밀리': 'sharp high-tone female vocalists with clever off-beat flow and chart-penetrating diction',
    '티아코라': 'anime-aesthetic female vocalists blending quirky visuals with unique trap beat choices',
    '티에라 왁': 'inventive creative female vocalists who stunned music world with one-minute track concepts',
    '티에라왁': 'inventive creative female vocalists who stunned music world with one-minute track concepts',
    '코이 르레이': 'light floating female vocalists with melodic singing-rap over 808 trap beats uniquely',
    '코이르레이': 'light floating female vocalists with melodic singing-rap over 808 trap beats uniquely',
    # ── 🔥 트랩 & 멜로딕 힙합 (Trap & Melodic Drill) ──
    '도자 캣': 'genre-crossing female artist seamlessly switching between pop, R&B, and hardcore trap',
    '도자캣': 'genre-crossing female artist seamlessly switching between pop, R&B, and hardcore trap',
    '리틀 심즈': 'globally acclaimed UK female vocalists with orchestral trap narratives and depth',
    '리틀심즈': 'globally acclaimed UK female vocalists with orchestral trap narratives and depth',
    '아쿠아 나루': 'poetic jazz-harmony female vocalists fusing intricate trap beats with literary grace',
    '아쿠아나루': 'poetic jazz-harmony female vocalists fusing intricate trap beats with literary grace',
    '캄린': '90s G-Funk revivalist female vocalists reinterpreting west-coast vibes through modern trap',
    '리코 네스티': 'punk-rage female vocalists pioneering trap-metal by fusing rock fury with hardcore beats',
    '리코네스티': 'punk-rage female vocalists pioneering trap-metal by fusing rock fury with hardcore beats',
    '영 마': 'Brooklyn deep-voiced female vocalists dominating trap beats with cold mid-bass authority',
    '영마': 'Brooklyn deep-voiced female vocalists dominating trap beats with cold mid-bass authority',
    '노네임': 'whisper-soft literary female vocalists floating poetically over jazz and lo-fi trap beats',
    '덱 로프': 'gentle melodic trap female vocalists smooth singing-rap trends in 2010s',
    '덱로프': 'gentle melodic trap female vocalists smooth singing-rap trends in 2010s',
    '비비머타': 'sharp organic indie-trap female vocalists creating the rawest underground groove patterns',
    '샤이걸': 'UK club hyperpop female vocalists crossing electronic beats with melodic trap seamlessly',
    '아르마니 시저': 'Griselda Records queen female vocalists moving between and dark trap mastery',
    '아르마니시저': 'Griselda Records queen female vocalists moving between and dark trap mastery',
    '레이디 레셔': 'ultra-fast UK grime female vocalists with blazing speed over trap and beats',
    '레이디레셔': 'ultra-fast UK grime female vocalists with blazing speed over trap and beats',
    '스테플론 돈': 'dancehall-reggae female vocalists fusing Caribbean rhythms with trap globally',
    '스테플론돈': 'dancehall-reggae female vocalists fusing Caribbean rhythms with trap globally',
    '쉔시아': 'Caribbean-flavored female vocalists who effortlessly rides trendy American trap production',
    '프린세스 노키아': 'NYC underground queen female vocalists embodying alternative trap with raw authenticity',
    '프린세스노키아': 'NYC underground queen female vocalists embodying alternative trap with raw authenticity',
    '레일라': 'dark European female vocalists commanding heavy trap beats with ominous presence',
    '글로벌 믹스 래퍼': 'Eastern-melodic female vocalists crossing Asian tonality with dark trap production',
    '글로벌믹스래퍼': 'Eastern-melodic female vocalists crossing Asian tonality with dark trap production',
    '엠아이에이': 'revolutionary female vocalists fusing third-world percussion with electronic trap radically',
    '산티골드': 'new-wave rock female vocalists who shattered boundaries between rock and hybrid trap',
    '나오': 'sophisticated tension-chord female vocalist with distinctive falsetto over refined trap beats',
    # ── 🇬🇧 글로벌 영미권 & 그라임/드릴 (UK / Drill / Global) ──
    '에니': 'London-born healing female vocalists with jazz- warmth over refined beats',
    '아이라 스타': 'Afrobeat- hybrid female vocalists setting global rhythm trends with infectious energy',
    '아이라스타': 'Afrobeat- hybrid female vocalists setting global rhythm trends with infectious energy',
    '이브스 투모어': 'radically alternative female vocalists fusing avant-garde sound with trap production',
    '이브스투모어': 'radically alternative female vocalists fusing avant-garde sound with trap production',
    '토미 제네시스': 'Canadian dark-aesthetic female vocalists with unique fetish-rap style over trap beats',
    '토미제네시스': 'Canadian dark-aesthetic female vocalists with unique fetish-rap style over trap beats',
    '수가 티': 'historic west-coast crew female vocalists with distinctive original flow and presence',
    '수가티': 'historic west-coast crew female vocalists with distinctive original flow and presence',
    '미즈 다이너마이트': 'pioneering UK grime-garage female vocalists who opened mainstream doors in 2000s',
    '미즈다이너마이트': 'pioneering UK grime-garage female vocalists who opened mainstream doors in 2000s',
    '나바': 'microtonal Arab-maqam female vocalists crossing Middle-Eastern melody with beats',
    '엠씨 멜로디': 'Dutch female vocalists who captivated all of Europe with classic flow',
    '엠씨멜로디': 'Dutch female vocalists who captivated all of Europe with classic flow',
    '니나 디아즈': 'Latin rock- crossover female vocalists with hybrid trap energy and raw power',
    '니나디아즈': 'Latin rock- crossover female vocalists with hybrid trap energy and raw power',
    '디암스': 'France greatest-selling female vocalists with epic narrative storytelling and authority',
    '제니': 'globally verified female vocalists with trendy English trap flow and stage power',
    '드리지': 'Chicago hardcore female vocalists with precise punchlines and aggressive delivery',
    '차이나': 'Philadelphia dark-cloud trap female vocalists with haunting atmospheric lo-fi mastery',
    '스노우 더 프로덕트': 'world-class speed-rap female vocalists switching effortlessly between English and Spanish',
    '스노우더프로덕트': 'world-class speed-rap female vocalists switching effortlessly between English and Spanish',
    '가비': 'Latin reggaeton- crossover female vocalists connecting Caribbean beats with US trap',
    '소피아 블랙': 'R&B-infused female vocalists riding 808 glide bass with smooth vocal elegance',
    '소피아블랙': 'R&B-infused female vocalists riding 808 glide bass with smooth vocal elegance',
    '비비 부렐리': 'hit-songwriter female vocalists with raw soulful trap vocals and creative genius',
    '비비부렐리': 'hit-songwriter female vocalists with raw soulful trap vocals and creative genius',
    '하비아 마이티': 'Polaris-winning Canadian female vocalists with hardcore mastery and intelligence',
    '하비아마이티': 'Polaris-winning Canadian female vocalists with hardcore mastery and intelligence',
    '칼리 우치스': 'dreamy Latin-pop female vocalists weaving ethereal harmonics with southern trap groove',
    '칼리우치스': 'dreamy Latin-pop female vocalists weaving ethereal harmonics with southern trap groove',
    '티나셰': 'lethal off-beat female vocalists-singer delivering devastating flow over syncopated hi-hats',
    # ── 🇰🇷 대한민국 최고의 여성 래퍼 (K-HipHop Queens) ──
    '씨엘': '2NE1 hardcore female vocalists Billboard and global fashion-music fusion',
    '제시': 'explosive raspy female vocalists with southern trap energy and stage-commanding power',
    '치타': 'razor-precise female vocalists devouring and trap with lethal punchlines',
    '이영지': 'deep baritone-grade female vocalists bombing modern and trap beats powerfully',
    '미란이': 'addictive melodic female vocalists comforting audiences with catchy singing-rap hooks',
    '신스': 'hardcore female vocalists filling beats with intense life-story narratives',
    '키디비': 'technically versatile female vocalists freely riding R&B harmonics and rhymes',
    '길미': 'original all-rounder female vocalists with rapid-fire delivery and powerful singing',
    '캠보': 'emerging female vocalists commanding heavy 808 beats with bold presence',
    '전소연': 'genius producing female vocalists who shatters idol limits with trap mastery',
    '유빈': 'charming mid-low female vocalists with attractive husky tone on beats',
    '지민': 'distinctive ultra-high female vocalists crafting ear-catching trap hooks with precision',
    '카디': 'underground female vocalists crossing hardcore rock sound with trap production',
    '엑시': 'competition-bred female vocalists with solid vocal foundation and rhythmic precision',
    '문별': 'bold thick-toned female vocalists anchoring songs with signature mid-low delivery',
    '최예나': 'pop-punk female vocalists harmoniously mixing cute trap flow with bubbly energy',
    '리사': 'global Billboard-hitting female vocalists with Thai-international swagger and trap queen energy',


    # ═══════════════════════════════════════════════════════════
    # cgo-384: 트로트 남자 보컬 50명
    # ═══════════════════════════════════════════════════════════
    # ── 🎵 전통 트로트 개척자 & 전설 ──
    '남인수': 'crystalline pure-toned male tenor revered as the emperor of classic',
    '고복수': 'plaintive gentle soothing homesick hearts with simple heartfelt melody',
    '백년설': 'rich earthy male baritone comforting working-class souls with rustic warmth',
    '현인': 'pioneering with signature vibrato who opened popular music',
    '박재홍': 'powerful open-throated male singer belting folk sorrows with piercing clarity',
    '진방남': 'sorrowful bending-note master with deeply mournful delivery',
    '이인권': 'warm low-register evoking hometown nostalgia with gentle phrasing',
    '도미': 'sophisticated mid-century bridging modern melody with traditional roots',
    '배호': 'immortal deep baritone who elevated music with noble dignity',
    '한복남': 'humorous witty who popularized comedic storytelling with catchy groove',
    # ── 👑 트로트 황금기 & 양대산맥 레전드 ──
    '남진': 'Elvis-inspired charismatic with stage magnetism',
    '최희준': 'elegant baritone singing life melancholy with refined literary grace',
    '태진아': 'hook-driven addictive dominating with catchy refrains and deep emotion',
    '송대관': 'quintessentially optimistic with earthy rustic warmth and joy',
    '설운도': 'genius singer-songwriter blending samba and twist rhythms creatively',
    '현철': 'uniquely flavored with signature nasal bending-note technique mastery',
    '조항조': 'mournful mid-bass commanding orchestral-scale grand ballad narratives',
    # ── ⚡ 파워 락·댄스 & 뉴웨이브 ──
    '강진': 'rhythmic groove master who electrified all generations with one hit',
    '박현빈': 'classically trained powerful high-note who launched power- era',
    '신유': 'sweet romantic tenor with handsome appeal and lyrical sensitivity',
    '진성': 'explosive raspy with gut-wrenching sorrow and raw emotional power',
    '박상철': 'brass-backed powerhouse with commanding stage energy',
    '영탁': 'rhythmic all-rounder with powerful diction and stage-breaking energy',
    '장민호': 'refined groovy with idol-trained polish and solid vocal technique',
    '이찬원': 'traditional bending-note technician with earthy fermented-bean voice',
    '김호중': 'operatic tenor completing orchestral-scale power with massive volume',
    '김수찬': 'flashy showman male new-wave with infectious entertainment energy',
    # ── 🪕 감성 서정 & 포크 융합 ──
    '김희재': 'polished hybrid blending precise choreography with sweet light tenor',
    '오승근': 'folk-rooted gentle comforting the nation with plain warm delivery',
    '진시몬': 'folk-ballad optimized with sweet sentimental melodic craftsmanship',
    '나태주': 'clear steady male pop- hiding deep lyricism behind flashy performance',
    '배일호': 'earthy rustic combining rural folk sentiment with tradition',
    '김용필': 'dignified low-tone who sings like reciting poetry with gravitas',
    '안성훈': 'pristine clean high-note male healing-ballad with flawless precision',
    '박지현': 'bright energetic radiating vitality with open airy tenor delivery',
    '최수호': 'minimalist acoustic with deep resonance on simple folk melodies',
    # ── 🎻 국악 크로스오버 & 시네마틱 ──
    '민수현': 'gentle refined delivering traditional depth with elegant composure',
    '최재명': 'pansori-infused cinematic with elaborate melodic architecture and grand projection',
    '정동원': 'prodigy mastering saxophone to orchestra with epic narrative depth',
    '손태진': 'classical crossover harmonizing operatic power with grand orchestral scale',
    '최우진': 'stable soaring high-note riding grand traditional melodies',
    '박서진': 'percussion-performing with deeply sorrowful han-infused vocal power',
    '강태관': 'pansori-based showing textbook traditional crossover with profound depth',
    '고영열': 'master-architect radically mixing pansori, piano and harmonics',
    '조명섭': 'bel-canto creating cinematic time-slip narratives with unique timbre',
    '영광': 'rugged bending-note cutting through grand horn and string ensembles',
    '남승민': 'cinematic riding large string sections with emotional stability and depth',

    # ═══════════════════════════════════════════════════════════
    # cgo-384: 트로트 여자 보컬 50명
    # ═══════════════════════════════════════════════════════════
    # ── 🎵 전통 트로트 여류 전설 ──
    '황금심': 'crystalline nightingale who dominated early classic',
    '이난영': 'legendary nasal-melody who comforted a colonized nation with sorrow',
    '심연옥': 'deep resonant who tenderly soothed wartime refugees with warmth',
    '박재란': 'brilliant nightingale celebrated as the golden voice of the 50s-60s',
    '백설희': 'hauntingly beautiful who captured han in exquisite melody',
    '박애경': 'legendary harmony showcasing textbook traditional duet vocal mastery',
    '김향미': 'rustic mid-low bending-note anchoring legendary harmony foundations',
    '지화자': 'stable powerhouse with the most reliable pentatonic vocal delivery',
    '안다성': 'elegant 60s layering sophisticated arrangements over traditional melody',
    '이미자': 'the living goddess of elegy with immortal bending-note vocal mythology',
    # ── 👑 트로트 황금기 여제들 ──
    '하춘화': 'textbook with decades of live performance forging unshakeable technique',
    '김연자': '- queen female vocalist who conquered both Japan and Korea with explosive power',
    '김수희': 'powerful pansori-toned who made the whole nation cry and laugh',
    '주현미': 'pharmacist-turned with crystalline falsetto and delicate high bending',
    '문희옥': 'textbook traditional with the most flavorful classic bending delivery',
    '현숙': 'positive upbeat rhythmic with infectious joy',
    '최진희': 'pop-ballad crossover perfectly blending Western harmony with sorrow',
    '방실이': 'powerhouse big-voiced energetic showmanship',
    '한혜진': 'husky deep mid-low adding mature depth to',
    '한복희': 'stage-dominating with addictive groove and magnetic crowd control',
    # ── ⚡ 파워 댄스 & 뉴웨이브 ──
    '장윤정': 'genre-reshaping female queen who single-handedly revived for a new generation',
    '홍진영': 'cute nasally charming female electronic with modern pop appeal',
    '김혜연': 'powerful venue-shaking commanding massive event stages',
    '서지오': 'vocalists-style maintaining rock-solid technique through choreography',
    '은가은': 'musical-theater trained female power- with soaring high-note stage presence',
    '황우림': 'idol-trained groovy female new-wave incorporating Latin rhythms creatively',
    '별사랑': 'all-range female spanning deep bass to soaring high notes',
    '허찬미': 'idol-crossover with trendy beat-riding ability and sharp performance',
    '요요미': 'cute bright dominating highway-groove beats with adorable charm',
    '강혜연': 'girl-group trained hiding solid traditional vocal power behind charm',
    # ── 🪕 감성 서정 & 포크 융합 ──
    '홍자': 'thick soulful showing peak sorrowful delivery with gomtang warmth',
    '우연이': 'folk-rock gentle comforting the nation with plain heartfelt warmth',
    '금잔디': 'highway queen with tender yet smooth voice soothing working hearts',
    '정다경': 'dancer-trained graceful with clear deep lyrical vocal delivery',
    '김나희': 'crystal-clear healing who shattered comedian-to-singer stereotypes',
    '강예슬': 'pure refreshing providing emotional calm with angelic bright tone',
    '마리아': 'first foreign champion female vocalist who mastered bending-note technique',
    '김다현': 'young prodigy narrating deep life stories with mature emotional arc',
    '김태연': 'pansori-master young melting fierce traditional soul into acoustic folk',
    '윤태화': 'solid expressive carrying traditional lyrical depth with steady power',
    # ── 🎻 국악 크로스오버 & 시네마틱 ──
    '송가인': 'Miss champion female vocalist with overwhelming pansori-based power shattering han',
    '양지은': 'pansori-certified female crossover singing miracles over grand orchestrations',
    '홍지윤': 'doll-faced hiding explosive pansori-scaled cinematic high-note power',
    '김의영': 'spicy capsaicin-sharp with traditional bending and pansori flair',
    '전유진': 'prodigious genius female cinematic effortlessly riding symphonic waves',
    '오유진': 'gayageum-playing female hybrid bridging traditional music and',
    '최향': 'rich-volume female cinematic standing firm within grand horn ensembles',
    '풍금': 'refined who distills deep traditional han into cinematic film-scale delivery',
    '신미래': 'dreamy atmospheric reinterpreting 30s-40s with ethereal tone',
    '나영': 'explosive next-generation female cinematic with overwhelming pansori projection',

    # Total: 97 artists, 97 entries

    # ── cgo-386: EDM 200명 보컬리스트 ──
    # Male Progressive/House
    '아비치': 'melodic progressive house, bright euphoric synth leads, emotional festival anthem drops',
    '마틴 개릭스': 'arena electro house, explosive big room drops, soaring festival synth anthems',
    '마틴개릭스': 'arena electro house, explosive big room drops, soaring festival synth anthems',
    '제드': 'pop-electronic fusion, polished harmonic structures, crystalline synth production',
    '다프트 펑크 토마': 'french touch house, funky analog synth bass, robotic vocoder filtered disco',
    '다프트펑크토마': 'french touch house, funky analog synth bass, robotic vocoder filtered disco',
    '다프트 펑크 기마누엘': 'retro disco house, warm analog synth pads, groovy filtered funk loops',
    '다프트펑크기마누엘': 'retro disco house, warm analog synth pads, groovy filtered funk loops',
    '하드웰': 'powerful big room house, massive kick-driven drops, stadium-scale synth stabs',
    '악스웰': 'swedish progressive house, uplifting chord progressions, euphoric melodic builds',
    '세바스티앙 인그로소': 'stadium anthem house, grand-scale melodic builds, driving progressive drops',
    '세바스티앙인그로소': 'stadium anthem house, grand-scale melodic builds, driving progressive drops',
    '스티브 안젤로': 'heavy rhythmic house, deep bass-driven grooves, powerful percussive energy',
    '스티브안젤로': 'heavy rhythmic house, deep bass-driven grooves, powerful percussive energy',
    '카고': 'tropical house, warm sunny synth pads, relaxed melodic plucks and piano chords',
    '돈 디아블로': 'future house, sleek futuristic synth design, bouncy bass-driven grooves',
    '돈디아블로': 'future house, sleek futuristic synth design, bouncy bass-driven grooves',
    '앤드류 타가트': 'pop-EDM crossover, catchy vocal-driven hooks, radio-friendly dance production',
    '앤드류타가트': 'pop-EDM crossover, catchy vocal-driven hooks, radio-friendly dance production',
    '알레소': 'emotional progressive house, heart-stirring chord progressions, euphoric climactic drops',
    '닉 로메로': 'powerful kick-driven house, sharp synth arpeggios, protocol-style progressive drops',
    '닉로메로': 'powerful kick-driven house, sharp synth arpeggios, protocol-style progressive drops',
    '데이비드 게타': 'mainstream pop-dance, polished radio EDM, uplifting festival-ready anthems',
    '데이비드게타': 'mainstream pop-dance, polished radio EDM, uplifting festival-ready anthems',
    '캘빈 해리스': 'disco-house funk, chart-topping dance production, groove-heavy synth bass',
    '캘빈해리스': 'disco-house funk, chart-topping dance production, groove-heavy synth bass',
    '페데 레 그란드': 'classic club house, solid four-on-the-floor grooves, standard dancefloor energy',
    '페데레그란드': 'classic club house, solid four-on-the-floor grooves, standard dancefloor energy',
    '케스케이드': 'dreamy atmospheric house, lush vocal pads, smooth emotional dance melodies',
    '데드마우스': 'minimal progressive techno, precise mathematical synth patterns, cerebral electronic',
    '에릭 프리즈': 'hypnotic progressive tech house, immersive light-and-sound art, deep layered builds',
    '에릭프리즈': 'hypnotic progressive tech house, immersive light-and-sound art, deep layered builds',
    # Male Trance/Anthems
    '아민 반 뷰렌': 'uplifting vocal trance, soaring 138 BPM builds, euphoric emotional anthem climaxes',
    '아민반뷰렌': 'uplifting vocal trance, soaring 138 BPM builds, euphoric emotional anthem climaxes',
    '티에스토': 'all-genre electronic master, stadium trance anthems transitioning to pop dance mastery',
    '폴 반 다이크': 'classic German trance, deep emotional 90s synth leads, pure uplifting energy',
    '폴반다이크': 'classic German trance, deep emotional 90s synth leads, pure uplifting energy',
    '페리 코스텐': 'melodic euro trance, grand orchestral builds, sweeping cinematic progressions',
    '페리코스텐': 'melodic euro trance, grand orchestral builds, sweeping cinematic progressions',
    '조노 그랜트': 'emotional vocal trance, heart-purifying chord sequences, spiritual group trance',
    '조노그랜트': 'emotional vocal trance, heart-purifying chord sequences, spiritual group trance',
    '토니 맥기네스': 'soothing trance harmony, gentle heartbeat-calming melodic progression mastery',
    '토니맥기네스': 'soothing trance harmony, gentle heartbeat-calming melodic progression mastery',
    '파보 실야마키': 'cinematic trance soundscape, epic orchestral live performance, layered harmonic genius',
    '파보실야마키': 'cinematic trance soundscape, epic orchestral live performance, layered harmonic genius',
    '다쉬 베를린': 'anthem trance, heart-wrenching melodies, powerful emotional vocal trance hooks',
    '다쉬베를린': 'anthem trance, heart-wrenching melodies, powerful emotional vocal trance hooks',
    '마르쿠스 슐츠': 'dark progressive trance, heavy deep atmospheres, powerful brooding builds',
    '마르쿠스슐츠': 'dark progressive trance, heavy deep atmospheres, powerful brooding builds',
    '알리 앤 필라': 'uplifting 138 BPM trance, Egyptian-inspired melodies, pure euphoric energy',
    '알리앤필라': 'uplifting 138 BPM trance, Egyptian-inspired melodies, pure euphoric energy',
    '빌럼 반 하네험': 'trance-big room hybrid, explosive rave energy, powerful harmonic combinations',
    '빌럼반하네험': 'trance-big room hybrid, explosive rave energy, powerful harmonic combinations',
    '바르트 반 데르 빌스트': 'stadium rave beats, ultra-powerful synth hooks, shaking big room energy',
    '바르트반데르빌스트': 'stadium rave beats, ultra-powerful synth hooks, shaking big room energy',
    '가레스 에메리': 'emotional healing trance, angelic vocal melodies, soothing uplifting atmospheres',
    '가레스에메리': 'emotional healing trance, angelic vocal melodies, soothing uplifting atmospheres',
    '코스믹 게이트': 'hard trance grooves, distinctive bounce patterns, energetic dancefloor drive',
    '코스믹게이트': 'hard trance grooves, distinctive bounce patterns, energetic dancefloor drive',
    '앤드류 바이어': 'cinematic electronic art, classical-influenced precise sound design, intricate layers',
    '앤드류바이어': 'cinematic electronic art, classical-influenced precise sound design, intricate layers',
    '디미트리 베가스': 'big room anthem, massive crowd-engaging drops, festival main stage energy',
    '디미트리베가스': 'big room anthem, massive crowd-engaging drops, festival main stage energy',
    '라이크 마이크': 'Belgian festival vocal performance, hype crowd energy, powerful show presence',
    '라이크마이크': 'Belgian festival vocal performance, hype crowd energy, powerful show presence',
    '티미 트럼펫': 'live trumpet hybrid, real brass instrument with hardstyle beats, explosive power',
    '티미트럼펫': 'live trumpet hybrid, real brass instrument with hardstyle beats, explosive power',
    '빈 나이': 'psychedelic trance chants, mystical Israeli psytrance, hypnotic tribal rhythms',
    '빈나이': 'psychedelic trance chants, mystical Israeli psytrance, hypnotic tribal rhythms',
    '케이쉬머': 'Indian orchestral big room, traditional instrument fusion, epic cinematic EDM',
    # Male Bass/Dubstep
    '스크릴렉스': 'aggressive brostep, heavy wobble bass, glitchy distorted synths, chaotic drops',
    '롭 스와이어': 'destructive electro-dubstep, razor-sharp bass design, Knife Party intensity',
    '롭스와이어': 'destructive electro-dubstep, razor-sharp bass design, Knife Party intensity',
    '엑시전': 'ultra-heavy bass dubstep, thunderous sub-bass, dinosaur-weight drops',
    '일레늄': 'melodic future bass, emotional cinematic builds, heartfelt synth-driven drops',
    '세이드 더 스카이': 'warm melodic bass, healing emotional synths, gentle future bass atmospheres',
    '세이드더스카이': 'warm melodic bass, healing emotional synths, gentle future bass atmospheres',
    '산 홀로': 'bright celestial future bass, live electric guitar layers, uplifting airy synths',
    '산홀로': 'bright celestial future bass, live electric guitar layers, uplifting airy synths',
    '마시멜로': 'pop future bass, catchy melodic hooks, radio-friendly bouncy synth drops',
    '가레스 맥그릴런': 'drum and bass rock fusion, live band DnB energy, Pendulum intensity',
    '가레스맥그릴런': 'drum and bass rock fusion, live band DnB energy, Pendulum intensity',
    '서브 포커스': 'rapid-fire DnB, sleek melodic drum and bass, 174 BPM precision beats',
    '서브포커스': 'rapid-fire DnB, sleek melodic drum and bass, 174 BPM precision beats',
    '네츠키': 'liquid funk DnB, bright refreshing drum and bass melodies, smooth flowing energy',
    '디플로': 'dancehall-bass fusion, Caribbean rhythms with heavy bass, Major Lazer energy',
    '알엘 그라임': 'heavy 808 trap EDM, deep sub-bass with flashy synths, powerful trap drops',
    '알엘그라임': 'heavy 808 trap EDM, deep sub-bass with flashy synths, powerful trap drops',
    '플로스트라다무스': 'trap EDM pioneer, swagger-heavy beats with electronic drops',
    '젯츠 데드': 'old-school dubstep, dark bass, Toronto underground bass music',
    '젯츠데드': 'old-school dubstep, dark bass, Toronto underground bass music',
    '네로': 'cinematic synthwave dubstep, London bass orchestra, dramatic electronic scores',
    '플룸': 'experimental glitch bass, future-forward alternative electronic, deconstructed beats',
    '슬러시': 'chiptune dubstep, cute 8-bit game sounds with heavy bass drops, playful energy',
    '보그레': 'raw hardcore dubstep, intense aggressive bass, provocative high-energy drops',
    '루이 더 차일드': 'soft minimal future pop bass, feel-good bouncy melodies, bright happy vibes',
    '루이더차일드': 'soft minimal future pop bass, feel-good bouncy melodies, bright happy vibes',
    '나이트메어': 'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy',
    # Male Hardstyle/Hardcore
    '헤드헌터즈': 'legendary hardstyle, melodic euphoric kicks, genre-defining reverse bass anthems',
    '와일드스타일즈': 'golden-era hardstyle, summer anthem melodies, popularizing euphoric hardstyle',
    '브레넌 하트': 'cinematic emotional hardstyle, stadium singalong anthems, epic reverse bass',
    '브레넌하트': 'cinematic emotional hardstyle, stadium singalong anthems, epic reverse bass',
    '디블록 앤 스테판': 'euro dance hardstyle harmony, perfectly balanced euphoric kicks and melodies',
    '디블록앤스테판': 'euro dance hardstyle harmony, perfectly balanced euphoric kicks and melodies',
    '코원': 'Belgian hardstyle representative, powerful kick drums, main stage energy',
    '자톡스': 'Italian minor-key hardstyle, raw powerful kicks, dramatic dark melodies',
    '다 트위카즈': 'fun remix hardstyle, Disney OST remixes, playful energetic hard kicks',
    '다트위카즈': 'fun remix hardstyle, Disney OST remixes, playful energetic hard kicks',
    '서브 제로 프로젝트': 'modern psy kicks, innovative synth loops, cutting-edge hardstyle sound design',
    '서브제로프로젝트': 'modern psy kicks, innovative synth loops, cutting-edge hardstyle sound design',
    '란디': 'rawstyle master, devastating low kicks, shocking remix power, dark intensity',
    '건즈 포 하이어': 'underground dark rave, intense brooding hardstyle, heavy atmospheric drops',
    '건즈포하이어': 'underground dark rave, intense brooding hardstyle, heavy atmospheric drops',
    '앵거피스트': 'hardcore gabber, 170 BPM distorted kicks, relentless violent energy',
    '쇼텍': 'classic hardstyle foundation, transitioning to pop dance, versatile production',
    '아타로스': 'transparent beautiful pop-melodic hardstyle, crystal-clear emotional kicks',
    '사이코 펑크즈': 'raw funky hardstyle, powerful energetic drives, punchy festival kicks',
    '사이코펑크즈': 'raw funky hardstyle, powerful energetic drives, punchy festival kicks',
    '노이즈컨트롤러즈': 'sound engineering perfection, cleanest hardstyle kicks, precise production',
    '퓨처 노이즈': 'philosophical narrative hardstyle, poetic cinematic sad stories, artistic depth',
    '퓨처노이즈': 'philosophical narrative hardstyle, poetic cinematic sad stories, artistic depth',
    '워페이스': 'hardcore raw kicks, metal-infused bass, heart-pounding intense drops',
    '세파': 'frenchcore piano genius, harpsichord classical harmony on 200 BPM kicks',
    '닥터 피콕': 'frenchcore godfather, festival-speed 200 BPM rave, relentless kick energy',
    '닥터피콕': 'frenchcore godfather, festival-speed 200 BPM rave, relentless kick energy',
    # Male Techno/Synthwave
    '리치 호틴': 'minimal techno architect, precise mathematical minimalism, surgical sound design',
    '리치호틴': 'minimal techno architect, precise mathematical minimalism, surgical sound design',
    '칼 콕스': 'legendary techno-house, warm groovy 4-on-the-floor, decades of dance mastery',
    '칼콕스': 'legendary techno-house, warm groovy 4-on-the-floor, decades of dance mastery',
    '스벤 바스': 'Frankfurt techno rave, legendary nightlife culture, driving hard techno energy',
    '스벤바스': 'Frankfurt techno rave, legendary nightlife culture, driving hard techno energy',
    '보리스 브레차': 'high-tech minimal, masked genius, innovative minimal techno groove patterns',
    '보리스브레차': 'high-tech minimal, masked genius, innovative minimal techno groove patterns',
    '제프 밀스': 'Detroit techno pioneer, ultra-fast turntable wizardry, futuristic machine rhythm',
    '제프밀스': 'Detroit techno pioneer, ultra-fast turntable wizardry, futuristic machine rhythm',
    '마이클 비비': 'sleek tech house groove, polished club-standard beats, modern dancefloor command',
    '마이클비비': 'sleek tech house groove, polished club-standard beats, modern dancefloor command',
    '솔로몬': 'deep house hypnosis, addictive disco harmony grooves, mesmerizing melodic depth',
    '블랙 커피': 'afro house, African percussion rhythms, deep soulful warm bass grooves',
    '블랙커피': 'afro house, African percussion rhythms, deep soulful warm bass grooves',
    '마세오 플렉스': 'dark cosmic electronic house, dreamy space-age atmospheres, deep grooves',
    '마세오플렉스': 'dark cosmic electronic house, dreamy space-age atmospheres, deep grooves',
    '피셔': 'funky tech house, infectious groovy 4-on-the-floor kicks, party-starting energy',
    '카빈스키': '80s neon retrowave synthwave, cinematic analog synths, Drive soundtrack vibes',
    '타일러 라일': 'neon synthwave, analog nostalgia, modern retro-futuristic electronic atmosphere',
    '타일러라일': 'neon synthwave, analog nostalgia, modern retro-futuristic electronic atmosphere',
    '게사펠슈타인': 'dark industrial techno-electro, brooding heavy bass, cinematic darkness',
    '페르투르바토르': 'cyberpunk synthwave, dystopian dark electronic, hardcore retro intensity',
    '존 서미트': 'trendsetting tech house, modern pop-EDM crossover, dominant club grooves',
    '존서미트': 'trendsetting tech house, modern pop-EDM crossover, dominant club grooves',
    # Female Techno/Underground
    '샬롯 드 위트': 'dark techno queen, powerful industrial kicks, commanding stage presence',
    '샬롯드위트': 'dark techno queen, powerful industrial kicks, commanding stage presence',
    '아멜리 렌': 'fast hard techno, pounding minimal drums, intense rave energy',
    '아멜리렌': 'fast hard techno, pounding minimal drums, intense rave energy',
    '니나 크라비츠': 'acid techno art, old-school analog acid bass, hypnotic Russian electronic',
    '니나크라비츠': 'acid techno art, old-school analog acid bass, hypnotic Russian electronic',
    '페기 구': 'global tech house-pop, catchy dance grooves, trendsetting K-EDM style',
    '페기구': 'global tech house-pop, catchy dance grooves, trendsetting K-EDM style',
    '토키몬스타': 'alternative techno beat-making, Grammy-nominated -American electronic genius',
    '앨리슨 원더랜드': 'cello-based hardcore trap-techno, classically trained aggressive bass queen',
    '앨리슨원더랜드': 'cello-based hardcore trap-techno, classically trained aggressive bass queen',
    '레즈': 'space mom hypnotic downtempo, eerie minimal techno, dark mysterious atmospheres',
    '블론디시': 'spiritual groovy afro-tech house, healing festival vibes, earthy dance energy',
    '아냐 슈나이더': 'Berlin underground techno matriarch, foundational German electronic scene',
    '아냐슈나이더': 'Berlin underground techno matriarch, foundational German electronic scene',
    '엘렌 알리엔': 'Berlin techno legend, electro-hard rave crossover, decades of innovation',
    '엘렌알리엔': 'Berlin techno legend, electro-hard rave crossover, decades of innovation',
    '마야 제인 콜스': 'deep dark deep house-minimal techno, subtle genius UK electronic production',
    '마야제인콜스': 'deep dark deep house-minimal techno, subtle genius UK electronic production',
    '한나 원츠': 'heavy groovy bass house, powerful UK dancefloor driving beats',
    '한나원츠': 'heavy groovy bass house, powerful UK dancefloor driving beats',
    '샘 디바인': 'classic club-standard house, Defected label queen, pure dancefloor energy',
    '샘디바인': 'classic club-standard house, Defected label queen, pure dancefloor energy',
    '니콜 무다버': 'hard techno first lady, thundering powerful drum hits, commanding beats',
    '니콜무다버': 'hard techno first lady, thundering powerful drum hits, commanding beats',
    '미스 키틴': 'French electroclash legend, 2000s electro vocal pioneer, iconic synth voice',
    '미스키틴': 'French electroclash legend, 2000s electro vocal pioneer, iconic synth voice',
    '모니카 크루즈': 'German first-generation techno, precise orderly beat structures, pure techno',
    '모니카크루즈': 'German first-generation techno, precise orderly beat structures, pure techno',
    '헤이디': 'energetic jacking house-techno, worldwide dance-inducing grooves, raw energy',
    '마그다': 'precise minimalist techno architecture, Richie Hawtin protege, surgical beats',
    '아니아 브루스터': 'sleek tech house grooves, polished feminine electronic production',
    '아니아브루스터': 'sleek tech house grooves, polished feminine electronic production',
    '클로에': 'French avant-garde techno, artistic experimental electronic soundscapes',
    # Female Progressive/House
    '올리비아 너보': 'Australian twin melody master, powerful vocal-driven progressive house',
    '올리비아너보': 'Australian twin melody master, powerful vocal-driven progressive house',
    '미미 너보': 'festival main stage powerhouse, energetic show performance, NERVO twin duo',
    '미미너보': 'festival main stage powerhouse, energetic show performance, NERVO twin duo',
    '자한 유스프': 'power pop-rock EDM vocal, aggressive electro house energy, Krewella force',
    '자한유스프': 'power pop-rock EDM vocal, aggressive electro house energy, Krewella force',
    '야스민 유스프': 'Eastern melodic scales with hard dance beats, Krewella creative genius',
    '야스민유스프': 'Eastern melodic scales with hard dance beats, Krewella creative genius',
    '안나': 'Brazilian progressive tech house, precise synth arpeggios, sophisticated grooves',
    '자스민 톰슨': 'angelic healing tropical house vocal, celestial pure tone, Kygo collaboration',
    '자스민톰슨': 'angelic healing tropical house vocal, celestial pure tone, Kygo collaboration',
    '노라 엔 퓨어': 'indie dance deep house, classical piano harmonies, healing organic melodies',
    '노라엔퓨어': 'indie dance deep house, classical piano harmonies, healing organic melodies',
    '이다 엔버그': 'Swedish minimal progressive house, elegant melodic sophistication',
    '이다엔버그': 'Swedish minimal progressive house, elegant melodic sophistication',
    '소피 하울리 벨트': 'Brazilian folk-house hybrid, Portuguese vocal raps, funky dance fusion',
    '소피하울리벨트': 'Brazilian folk-house hybrid, Portuguese vocal raps, funky dance fusion',
    '크리스티나 아길레라': 'powerful pop diva on house beats, explosive 7th-9th tension harmonies',
    '크리스티나아길레라': 'powerful pop diva on house beats, explosive 7th-9th tension harmonies',
    '엘피 지오비': 'jazz pianist house producer, live keyboard piano house grooves',
    '엘피지오비': 'jazz pianist house producer, live keyboard piano house grooves',
    '블론드웨어': 'future house synth pads, flashy feminine electronic production',
    '베키 힐': 'UK house-DnB hit vocal, chart-dominating dance vocal powerhouse',
    '베키힐': 'UK house-DnB hit vocal, chart-dominating dance vocal powerhouse',
    '조지아': '80s retro synthpop-London house, multi-instrument genius, boundary-breaking',
    '엘리 굴딩': 'ethereal high-register EDM vocal, iconic pop-electronic collaboration voice',
    '엘리굴딩': 'ethereal high-register EDM vocal, iconic pop-electronic collaboration voice',
    '폭스스': 'Grammy-winning EDM topline vocal, Zedd Clarity vocalist, pure melodic clarity',
    '헤일리 윌리엄스': 'power rock ballad vocal on EDM beats, explosive high notes, Paramore energy',
    '헤일리윌리엄스': 'power rock ballad vocal on EDM beats, explosive high notes, Paramore energy',
    '엠마 휴이트': 'trance vocal queen, Armin and Gareth Emery favorite, emotional vocal trance',
    '엠마휴이트': 'trance vocal queen, Armin and Gareth Emery favorite, emotional vocal trance',
    '헤더 노바': 'dreamy atmospheric electronic vocal, deep lingering emotional tone',
    '헤더노바': 'dreamy atmospheric electronic vocal, deep lingering emotional tone',
    '뢰이크솝 보컬': 'Nordic crystal-clear electronic vocal, cold beautiful Scandinavian soundscape',
    '뢰이크솝보컬': 'Nordic crystal-clear electronic vocal, cold beautiful Scandinavian soundscape',
    # Female Bass/Future Pop
    '클로지': 'organic world-music glitch bass, French experimental tribal dubstep fusion',
    '위프트 크림': 'dark trap with cinematic dubstep, aggressive bass queen',
    '위프트크림': 'dark trap with cinematic dubstep, aggressive bass queen',
    '미자': 'genre-destroying hybrid, dubstep-DnB-house crossover, Skrillex-endorsed innovator',
    '할리엔': 'celestial melodic bass vocal, Illenium and Armin featured angelic voice',
    '바시': 'powerful soul vocal on EDM beats, David Guetta and Tiesto featured powerhouse',
    '루엘': 'cinematic dubstep orchestral vocal, movie-score dramatic narrative queen',
    '라이츠': 'self-playing synth bass, melodic future bass golden toplines, creative genius',
    '켈라니': 'smooth R&B singing over 808 glide bass trap EDM, silky vocal flow',
    '알루나 프란시스': 'pop-future bass fairy, AlunaGeorge vocal pixie, light sparkling tone',
    '알루나프란시스': 'pop-future bass fairy, AlunaGeorge vocal pixie, light sparkling tone',
    '우피': 'retro 8-bit electro-trap pioneer, funky chiptune rebel, original genre bender',
    '에이바 맥스': 'dance anthem powerhouse, Tiesto collaboration, high-energy pop-EDM vocal',
    '에이바맥스': 'dance anthem powerhouse, Tiesto collaboration, high-energy pop-EDM vocal',
    '자라 라슨': 'precise rhythmic Swedish diva, Clean Bandit featured explosive vocal power',
    '자라라슨': 'precise rhythmic Swedish diva, Clean Bandit featured explosive vocal power',
    '듀아 리파': 'sticky house groove vocal, disco-rock crossover, dominant versatile tone',
    '듀아리파': 'sticky house groove vocal, disco-rock crossover, dominant versatile tone',
    '비비 렉사': 'global EDM hit vocal number one, David Guetta featured chart destroyer',
    '비비렉사': 'global EDM hit vocal number one, David Guetta featured chart destroyer',
    '알레시아 카라': 'perfect pitch hybrid vocal, Zedd Stay vocalist, precise rhythmic delivery',
    '알레시아카라': 'perfect pitch hybrid vocal, Zedd Stay vocalist, precise rhythmic delivery',
    '테일러 스위프트': 'arena-scale future bass buildup vocal, grand pop-EDM anthem arrangements',
    '테일러스위프트': 'arena-scale future bass buildup vocal, grand pop-EDM anthem arrangements',
    '로린': 'Eurovision cinematic electro queen, grand synthpop symphonic vocal power',
    '아니타': 'Latin Caribbean EDM fusion, Brazilian rhythmic bass drop energy',
    '비숍 브릭스': 'powerful gritty vocal on hardcore bass, chest-voice distortion ballad power',
    '비숍브릭스': 'powerful gritty vocal on hardcore bass, chest-voice distortion ballad power',
    # Female Hardstyle/Hardcore
    '미스 K8': 'hardcore gabber queen, 170 BPM destructive dominance, Ukrainian rave power',
    '미스K8': 'hardcore gabber queen, 170 BPM destructive dominance, Ukrainian rave power',
    '아니메': 'Italian hardcore rave bass master, raw aggressive gabber energy',
    '코르사코프': 'Netherlands hardcore techno legend, golden-era rave matriarch',
    '레이디 다나': 'early gabber-hardstyle pioneer, 2000s Dutch rave scene legend',
    '레이디다나': 'early gabber-hardstyle pioneer, 2000s Dutch rave scene legend',
    '스테파니': 'Italian rawstyle specialist, powerful festival stage reverse-bass kicks',
    '만디': 'Belgian pop-harmonic hardstyle, accessible euphoric kick melodies, rising star',
    '말루 트레베호': 'Latin samba-hardcore crossover, exotic rhythm on rave beat fusion',
    '말루트레베호': 'Latin samba-hardcore crossover, exotic rhythm on rave beat fusion',
    # Female K-EDM
    '디제이 미아': 'powerful K-club electro house, club scene dominating big room beats',
    '디제이미아': 'powerful K-club electro house, club scene dominating big room beats',
    '디제이 바나': 'sophisticated progressive house toplines, techno scene technician',
    '디제이바나': 'sophisticated progressive house toplines, techno scene technician',
    '수라': 'trendy neo-dance K-EDM hybrid, flashy showmanship, pan-Asian festival vocal',
    '디제이 소다': 'hybrid future pop bass, global SNS pioneer, Asian festival market leader',
    '디제이소다': 'hybrid future pop bass, global SNS pioneer, Asian festival market leader',
    '디제이 나리': 'solid four-on-the-floor club synths, polished club groove master',
    '디제이나리': 'solid four-on-the-floor club synths, polished club groove master',
    '안예은': 'traditional gugak-techno fusion, unprecedented sonic cultural crossover',
    '선우정아': 'jazz-pop-electronic boundary-breaking, sophisticated harmonic backend vocal',
    '림 킴': 'mysterious Eastern pentatonic dark trap bass, revolutionary K-electronic fusion',
    '림킴': 'mysterious Eastern pentatonic dark trap bass, revolutionary K-electronic fusion',
    '전소연': 'boundary-breaking, direct EDM house-techno drop production genius',
    '씨엘': 'Daft Punk-style french touch house, iconic global trap powerhouse',
    '제니': 'trendy electronic pop bass, Coachella main stage K-EDM pop fairy vocal',
    '빌리 에일리시': 'dreamy retro synthwave, cinematic industrial techno, whispery dark vocal',
    '빌리에일리시': 'dreamy retro synthwave, cinematic industrial techno, whispery dark vocal',
    # ═══ 팝록(Pop Rock) 남녀 200인 보컬 — 신규 160명 (cgo-409) ═══
    # ── 남성 팝록: 레전드 & 클래식 ──
    '브라이언아담스': 'bold arena-shaking male tenor with soaring anthemic pop rock delivery and triumphant energy',
    '브라이언 아담스': 'bold arena-shaking male tenor with soaring anthemic pop rock delivery and triumphant energy',
    '톰페티': 'steady rootsy male vocal with twangy heartland rock phrasing and honest everyman conviction',
    '톰 페티': 'steady rootsy male vocal with twangy heartland rock phrasing and honest everyman conviction',
    '릭스프링필드': 'clean bright male pop rock vocal with power hooks and radio-friendly arena sparkle',
    '릭 스프링필드': 'clean bright male pop rock vocal with power hooks and radio-friendly arena sparkle',
    '케니로긴스': 'dynamic driving male vocal with cinematic adventure energy and infectious pop rock groove',
    '케니 로긴스': 'dynamic driving male vocal with cinematic adventure energy and infectious pop rock groove',
    '존웨이트': 'yearning soaring male tenor with arena-sized melodic rock emotion and dramatic phrasing',
    '존 웨이트': 'yearning soaring male tenor with arena-sized melodic rock emotion and dramatic phrasing',
    '리처드막스': 'smooth polished male tenor with guitar-driven emotive pop rock craftsmanship',
    '리처드 막스': 'smooth polished male tenor with guitar-driven emotive pop rock craftsmanship',
    '제리래퍼티': 'mellow refined male vocal with jazzy sophistication and understated classic pop rock warmth',
    '제리 래퍼티': 'mellow refined male vocal with jazzy sophistication and understated classic pop rock warmth',
    '리오세이어': 'buoyant cheerful male falsetto with breezy light-hearted pop rock charm',
    '리오 세이어': 'buoyant cheerful male falsetto with breezy light-hearted pop rock charm',
    '크리스디버그': 'theatrical sweeping male vocal with romantic cinematic pop rock storytelling',
    '크리스 디 버그': 'theatrical sweeping male vocal with romantic cinematic pop rock storytelling',
    # ── 남성 팝록: 밴드 프런트맨 ──
    '애덤리바인': 'agile funky male tenor with slick pop groove and effortless high-register rock energy',
    '애덤 리바인': 'agile funky male tenor with slick pop groove and effortless high-register rock energy',
    '라이언테더': 'precision-tuned male pop rock vocal with anthemic build-ups and crystal-clear melodic control',
    '라이언 테더': 'precision-tuned male pop rock vocal with anthemic build-ups and crystal-clear melodic control',
    '조니레즈닉': 'tender emotionally charged male vocal with aching alternative pop rock vulnerability',
    '조니 레즈닉': 'tender emotionally charged male vocal with aching alternative pop rock vulnerability',
    '톰채플린': 'soaring piano-driven male tenor with expansive britpop rock melodies and pure tone',
    '톰 채플린': 'soaring piano-driven male tenor with expansive britpop rock melodies and pure tone',
    '브랜든플라워스': 'dramatic synth-rock male baritone with retro new wave charisma and arena presence',
    '브랜든 플라워스': 'dramatic synth-rock male baritone with retro new wave charisma and arena presence',
    '게리라이트바디': 'gentle atmospheric male vocal with introspective britpop warmth and quiet anthemic power',
    '게리 라이트바디': 'gentle atmospheric male vocal with introspective britpop warmth and quiet anthemic power',
    '프란히리': 'warm understated male vocal with folk-tinged pop rock sincerity and relaxed charm',
    '프란 히리': 'warm understated male vocal with folk-tinged pop rock sincerity and relaxed charm',
    '켈리존스': 'husky raspy male rock vocal with sharp guitar riff energy and raw welsh grit',
    '켈리 존스': 'husky raspy male rock vocal with sharp guitar riff energy and raw welsh grit',
    '스테판젠킨스': 'bright catchy male vocal with buzzing pop rock hooks and effortless melodic bounce',
    '스테판 젠킨스': 'bright catchy male vocal with buzzing pop rock hooks and effortless melodic bounce',
    '존온드라식': 'contemplative emotive male piano-rock vocal with heartfelt lyrical depth',
    '존 온드라식': 'contemplative emotive male piano-rock vocal with heartfelt lyrical depth',
    '아이작슬레이드': 'yearning vulnerable male vocal with soaring piano-driven alt rock sincerity',
    '아이작 슬레이드': 'yearning vulnerable male vocal with soaring piano-driven alt rock sincerity',
    '댄레이놀즈': 'thundering percussive male vocal with stadium-sized anthemic pop rock intensity',
    '댄 레이놀즈': 'thundering percussive male vocal with stadium-sized anthemic pop rock intensity',
    '패트모나한': 'easygoing charismatic male vocal with bright pop rock hooks and radio-friendly warmth',
    '패트 모나한': 'easygoing charismatic male vocal with bright pop rock hooks and radio-friendly warmth',
    '개빈로스데일': 'gravelly textured male vocal with alt rock edge mellowing into melodic pop rock',
    '개빈 로스데일': 'gravelly textured male vocal with alt rock edge mellowing into melodic pop rock',
    '루크헤밍스': 'youthful clear male tenor with pop punk energy evolving into mature pop rock',
    '루크 헤밍스': 'youthful clear male tenor with pop punk energy evolving into mature pop rock',
    '알렉스밴드': 'rich resonant male baritone with emotional post-grunge pop rock grandeur',
    '알렉스 밴드': 'rich resonant male baritone with emotional post-grunge pop rock grandeur',
    '데이비드쿡': 'powerful gritty male vocal with post-grunge pop rock conviction and stadium delivery',
    '데이비드 쿡': 'powerful gritty male vocal with post-grunge pop rock conviction and stadium delivery',
    '다미아노다비드': 'electrifying rebellious male vocal with Italian glam rock swagger and raw pop rock fire',
    '다미아노 다비드': 'electrifying rebellious male vocal with Italian glam rock swagger and raw pop rock fire',
    # ── 남성 팝록: 90~2000년대 모던 & 얼터너티브 ──
    '존메이어': 'silky blues-inflected male vocal with masterful guitar tone and laid-back pop rock finesse',
    '존 메이어': 'silky blues-inflected male vocal with masterful guitar tone and laid-back pop rock finesse',
    '개빈디그로우': 'soulful piano-rock male vocal with strong midrange and earnest pop rock delivery',
    '개빈 디그로우': 'soulful piano-rock male vocal with strong midrange and earnest pop rock delivery',
    '매튜스위트': 'jangly guitar-heavy male power pop vocal with 90s indie rock sweetness',
    '매튜 스위트': 'jangly guitar-heavy male power pop vocal with 90s indie rock sweetness',
    '피트욘': 'understated cool male vocal with mellow indie rock vibe and subtle melodic pull',
    '피트 욘': 'understated cool male vocal with mellow indie rock vibe and subtle melodic pull',
    '맷네이선슨': 'upbeat infectious male pop rock vocal with catchy hooks and charming storytelling',
    '맷 네이선슨': 'upbeat infectious male pop rock vocal with catchy hooks and charming storytelling',
    '데이비드그레이': 'moody atmospheric male vocal with folk rock undertones and britpop emotional depth',
    '데이비드 그레이': 'moody atmospheric male vocal with folk rock undertones and britpop emotional depth',
    '라이언아담스': 'prolific alt-country male vocal with roots rock grit and pop melody instinct',
    '라이언 아담스': 'prolific alt-country male vocal with roots rock grit and pop melody instinct',
    '던칸쉐이크': 'cerebral dreamy male vocal with delicate acoustic pop rock textures',
    '던칸 쉐이크': 'cerebral dreamy male vocal with delicate acoustic pop rock textures',
    '대니얼파우터': 'bright piano-pop male vocal with infectious hook-writing and upbeat pop rock charm',
    '대니얼 파우터': 'bright piano-pop male vocal with infectious hook-writing and upbeat pop rock charm',
    '벤폴즈': 'percussive piano-smashing male vocal with witty power pop rock energy and raw enthusiasm',
    '벤 폴즈': 'percussive piano-smashing male vocal with witty power pop rock energy and raw enthusiasm',
    '존맥러글린': 'polished keyboard-driven male vocal with soulful pop rock warmth and crafted melodies',
    '존 맥러글린': 'polished keyboard-driven male vocal with soulful pop rock warmth and crafted melodies',
    '하우이데이': 'layered acoustic male vocal building into sweeping pop rock grandeur',
    '하우이 데이': 'layered acoustic male vocal building into sweeping pop rock grandeur',
    '마크브루사드': 'southern-fried soulful male vocal with blues-rock grit and pop accessibility',
    '마크 브루사드': 'southern-fried soulful male vocal with blues-rock grit and pop accessibility',
    '에드윈매케인': 'warm earnest male vocal with gentle acoustic pop rock and heartfelt ballad delivery',
    '에드윈 매케인': 'warm earnest male vocal with gentle acoustic pop rock and heartfelt ballad delivery',
    '맷스캐넬': 'melodic focused male vocal with clean pop rock hooks and 90s alternative charm',
    '맷 스캐넬': 'melodic focused male vocal with clean pop rock hooks and 90s alternative charm',
    '케빈카도간': 'bright chiming male guitar-driven vocal with crystalline pop rock tone',
    '케빈 카도간': 'bright chiming male guitar-driven vocal with crystalline pop rock tone',
    '댄윌슨': 'masterful songwriting male vocal with understated power pop melodic genius',
    '댄 윌슨': 'masterful songwriting male vocal with understated power pop melodic genius',
    '토니스칼조': 'exuberant driving male pop rock vocal with 90s radio-rock catchiness',
    '토니 스칼조': 'exuberant driving male pop rock vocal with 90s radio-rock catchiness',
    '탈바흐만': 'clean soaring male tenor with shimmering guitar pop rock brightness',
    '탈 바흐만': 'clean soaring male tenor with shimmering guitar pop rock brightness',
    '잭안토노프': 'retro 80s-inspired male vocal with synth-laden pop rock production genius',
    '잭 안토노프': 'retro 80s-inspired male vocal with synth-laden pop rock production genius',
    # ── 남성 팝록: 어쿠스틱 & 포크 ──
    '제이슨무라즈': 'breezy nimble male vocal with acoustic guitar virtuosity and feel-good pop rock sunshine',
    '제이슨 무라즈': 'breezy nimble male vocal with acoustic guitar virtuosity and feel-good pop rock sunshine',
    '제임스블런트': 'plaintive delicate male tenor with aching acoustic pop rock vulnerability',
    '제임스 블런트': 'plaintive delicate male tenor with aching acoustic pop rock vulnerability',
    '제임스모리슨': 'husky textured male vocal with raw folk-rock warmth and pop accessibility',
    '제임스 모리슨': 'husky textured male vocal with raw folk-rock warmth and pop accessibility',
    '뉴턴포크너': 'percussive fingerstyle male vocal with rhythmic acoustic pop rock dexterity',
    '뉴턴 포크너': 'percussive fingerstyle male vocal with rhythmic acoustic pop rock dexterity',
    '파올로누티니': 'vintage gravelly male vocal with scottish soul-rock swagger and pop rock magnetism',
    '파올로 누티니': 'vintage gravelly male vocal with scottish soul-rock swagger and pop rock magnetism',
    '마이로우': 'gentle warm male vocal with European acoustic pop rock charm and melodic simplicity',
    '조지에즈라': 'deep rich male baritone with cheerful folk pop rock bounce and youthful energy',
    '조지 에즈라': 'deep rich male baritone with cheerful folk pop rock bounce and youthful energy',
    '밴스조이': 'bright ukulele-driven male vocal with sunny indie pop rock freshness',
    '밴스 조이': 'bright ukulele-driven male vocal with sunny indie pop rock freshness',
    '패신저': 'intimate acoustic male storyteller with folk pop rock singalong anthems',
    '벤하워드': 'intricate fingerpicking male vocal with brooding modern folk rock depth',
    '벤 하워드': 'intricate fingerpicking male vocal with brooding modern folk rock depth',
    '제이크버그': 'young raw male vocal with classic rockabilly swagger and folk pop rock edge',
    '제이크 버그': 'young raw male vocal with classic rockabilly swagger and folk pop rock edge',
    '브렛데넌': 'warm friendly male vocal with gentle folk pop rock melodies and positive vibe',
    '브렛 데넌': 'warm friendly male vocal with gentle folk pop rock melodies and positive vibe',
    '조슈아라딘': 'whispery soft male vocal with intimate acoustic pop rock tenderness',
    '조슈아 라딘': 'whispery soft male vocal with intimate acoustic pop rock tenderness',
    '맷코스타': 'retro 60s-flavored male vocal with pure folk pop rock nostalgia',
    '맷 코스타': 'retro 60s-flavored male vocal with pure folk pop rock nostalgia',
    '레이라몬테인': 'gravelly weathered male vocal with deep acoustic vintage pop rock soul',
    '레이 라몬테인': 'gravelly weathered male vocal with deep acoustic vintage pop rock soul',
    '그레이엄콜튼': 'crisp acoustic male vocal with TV-friendly indie pop rock polish',
    '그레이엄 콜튼': 'crisp acoustic male vocal with TV-friendly indie pop rock polish',
    '알렉시머독': 'minimal haunting male vocal with sparse acoustic pop rock emotional resonance',
    '알렉시 머독': 'minimal haunting male vocal with sparse acoustic pop rock emotional resonance',
    # ── 남성 팝록: 차세대 스타 ──
    '션멘데스': 'powerful clear male tenor with acoustic-electric pop rock urgency and youthful passion',
    '션 멘데스': 'powerful clear male tenor with acoustic-electric pop rock urgency and youthful passion',
    '호지어': 'towering blues-rock male vocal with gospel-tinged pop rock grandeur and poetic depth',
    '나일호란': 'warm folksy male vocal with 80s-inspired pop rock charm and easygoing charisma',
    '나일 호란': 'warm folksy male vocal with 80s-inspired pop rock charm and easygoing charisma',
    '샘펜더': 'springsteen-inspired male vocal with electric guitar anthems and working-class pop rock fire',
    '샘 펜더': 'springsteen-inspired male vocal with electric guitar anthems and working-class pop rock fire',
    '노아칸': 'earnest folk-rock male vocal with modern pop sensibility and confessional storytelling',
    '노아 칸': 'earnest folk-rock male vocal with modern pop sensibility and confessional storytelling',
    '벤슨분': 'explosive dramatic male vocal with thundering piano and volcanic pop rock crescendos',
    '벤슨 분': 'explosive dramatic male vocal with thundering piano and volcanic pop rock crescendos',
    '코난그레이': 'glam-influenced male vocal with retro 80s pop rock flair and emotional range',
    '코난 그레이': 'glam-influenced male vocal with retro 80s pop rock flair and emotional range',
    '스티븐산체스': 'vintage crooner male vocal with 50s rockabilly romance and timeless pop rock elegance',
    '스티븐 산체스': 'vintage crooner male vocal with 50s rockabilly romance and timeless pop rock elegance',
    '딘루이스': 'crisp emotive male vocal with guitar-forward Australian pop rock sincerity',
    '딘 루이스': 'crisp emotive male vocal with guitar-forward Australian pop rock sincerity',
    '칼럼스콧': 'delicate soaring male vocal with heartrending pop rock ballad vulnerability',
    '칼럼 스콧': 'delicate soaring male vocal with heartrending pop rock ballad vulnerability',
    '폴클라인': 'dreamy synth-pop male vocal with guitar-laced romantic pop rock atmosphere',
    '폴 클라인': 'dreamy synth-pop male vocal with guitar-laced romantic pop rock atmosphere',
    '시안두크로': 'passionate orchestral male vocal with sweeping Irish pop rock drama',
    '시안 두크로': 'passionate orchestral male vocal with sweeping Irish pop rock drama',
    '톰워커': 'gritty resonant male vocal with heavy rock beat-driven British pop rock power',
    '톰 워커': 'gritty resonant male vocal with heavy rock beat-driven British pop rock power',
    '제임스베이': 'bluesy electric guitar male vocal with fedora-era modern pop rock authenticity',
    '제임스 베이': 'bluesy electric guitar male vocal with fedora-era modern pop rock authenticity',
    '딜런바스': 'energetic guitar-forward male vocal with surging indie pop rock momentum',
    '딜런 바스': 'energetic guitar-forward male vocal with surging indie pop rock momentum',
    '엘리야휴슨': 'atmospheric post-punk male vocal with U2-lineage pop rock grandeur and modern edge',
    '엘리야 휴슨': 'atmospheric post-punk male vocal with U2-lineage pop rock grandeur and modern edge',
    '데클란맥케나': 'quirky inventive male vocal with eccentric British indie pop rock character',
    '데클란 맥케나': 'quirky inventive male vocal with eccentric British indie pop rock character',
    '잭코크레인': 'fierce direct male vocal with raw Scottish guitar pop rock intensity',
    '잭 코크레인': 'fierce direct male vocal with raw Scottish guitar pop rock intensity',
    # ── 여성 팝록: 클래식 파이오니어 ──
    '패트베네타': 'powerhouse belting female vocal with 80s arena pop rock dominance and fierce edge',
    '패트 베네타': 'powerhouse belting female vocal with 80s arena pop rock dominance and fierce edge',
    '크리시하인드': 'cool detached female vocal with new wave pop rock attitude and razor-sharp phrasing',
    '크리시 하인드': 'cool detached female vocal with new wave pop rock attitude and razor-sharp phrasing',
    '신디로퍼': 'quirky effervescent female vocal with distinctive vibrato and colorful 80s pop rock personality',
    '신디 로퍼': 'quirky effervescent female vocal with distinctive vibrato and colorful 80s pop rock personality',
    '보니타일러': 'thunderous husky female vocal with dramatic rock power ballad grandeur',
    '보니 타일러': 'thunderous husky female vocal with dramatic rock power ballad grandeur',
    '린다론스태드': 'versatile crystal-clear female vocal spanning country and pop rock with effortless range',
    '린다 론스태드': 'versatile crystal-clear female vocal spanning country and pop rock with effortless range',
    '낸시윌슨': 'refined guitar-driven female vocal with acoustic-to-electric pop rock mastery',
    '낸시 윌슨': 'refined guitar-driven female vocal with acoustic-to-electric pop rock mastery',
    '조안제트': 'raw rebellious female vocal with punk-edged hard pop rock defiance',
    '조안 제트': 'raw rebellious female vocal with punk-edged hard pop rock defiance',
    '데비해리': 'cool blonde new wave female vocal with catchy pop rock hooks and punk attitude',
    '데비 해리': 'cool blonde new wave female vocal with catchy pop rock hooks and punk attitude',
    '멜리사에더리지': 'fierce gravelly female vocal with passionate acoustic-electric pop rock conviction',
    '멜리사 에더리지': 'fierce gravelly female vocal with passionate acoustic-electric pop rock conviction',
    '김카네스': 'smoky distinctive female vocal with synth-pop rock mystery and 80s cool',
    '김 카네스': 'smoky distinctive female vocal with synth-pop rock mystery and 80s cool',
    '벨린다칼라일': 'sweet bright female vocal with effervescent 80s pop rock sparkle',
    '벨린다 칼라일': 'sweet bright female vocal with effervescent 80s pop rock sparkle',
    '로라브래니건': 'powerful soaring female vocal with dramatic melodic pop rock intensity',
    '로라 브래니건': 'powerful soaring female vocal with dramatic melodic pop rock intensity',
    '주스뉴튼': 'clean crisp female vocal with acoustic pop rock polish and country-tinged warmth',
    '주스 뉴튼': 'clean crisp female vocal with acoustic pop rock polish and country-tinged warmth',
    '린디로스': 'saxophone-playing female vocal with sophisticated pop rock and jazz-inflected elegance',
    '린디 로스': 'saxophone-playing female vocal with sophisticated pop rock and jazz-inflected elegance',
    '마사데이비스': 'sultry new wave female vocal with moody melodic pop rock intrigue',
    '마사 데이비스': 'sultry new wave female vocal with moody melodic pop rock intrigue',
    # ── 여성 팝록: 90년대 아이콘 ──
    '셰릴크로우': 'warm rootsy female vocal with laid-back guitar-driven pop rock swagger and authenticity',
    '셰릴 크로우': 'warm rootsy female vocal with laid-back guitar-driven pop rock swagger and authenticity',
    '그웬스테파니': 'bold ska-punk female vocal with No Doubt pop rock charisma and style icon presence',
    '그웬 스테파니': 'bold ska-punk female vocal with No Doubt pop rock charisma and style icon presence',
    '셜리맨슨': 'seductive cutting female vocal with industrial-tinged modern pop rock danger',
    '셜리 맨슨': 'seductive cutting female vocal with industrial-tinged modern pop rock danger',
    '나탈리임브룰리아': 'fresh clear female vocal with late-90s modern pop rock vulnerability and charm',
    '나탈리 임브룰리아': 'fresh clear female vocal with late-90s modern pop rock vulnerability and charm',
    '리즈페어': 'raw confessional female vocal with indie attitude and commercial pop rock crossover guts',
    '리즈 페어': 'raw confessional female vocal with indie attitude and commercial pop rock crossover guts',
    '조안오스본': 'soulful bluesy female vocal with weighty philosophical pop rock depth',
    '조안 오스본': 'soulful bluesy female vocal with weighty philosophical pop rock depth',
    '메러디스브룩스': 'fierce electric guitar-driven female vocal with unapologetic 90s pop rock attitude',
    '메러디스 브룩스': 'fierce electric guitar-driven female vocal with unapologetic 90s pop rock attitude',
    '폴라콜': 'intelligent layered female vocal with piano-driven art pop rock sophistication',
    '폴라 콜': 'intelligent layered female vocal with piano-driven art pop rock sophistication',
    '숀콜빈': 'graceful acoustic female vocal with Grammy-winning folk pop rock craftsmanship',
    '숀 콜빈': 'graceful acoustic female vocal with Grammy-winning folk pop rock craftsmanship',
    '샤린스피테리': 'sleek blues-flavored female vocal with British modern pop rock sophistication',
    '샤린 스피테리': 'sleek blues-flavored female vocal with British modern pop rock sophistication',
    '소피홉킨스': 'lilting rhythmic female vocal with catchy 90s pop rock melodies',
    '소피 홉킨스': 'lilting rhythmic female vocal with catchy 90s pop rock melodies',
    '리사로브': 'sweet bespectacled female vocal with acoustic pop rock innocence and melodic precision',
    '리사 로브': 'sweet bespectacled female vocal with acoustic pop rock innocence and melodic precision',
    '니나페르손': 'sugary Swedish female vocal with sophisticated pop rock elegance and subtle darkness',
    '니나 페르손': 'sugary Swedish female vocal with sophisticated pop rock elegance and subtle darkness',
    '리내시': 'angelic ethereal female vocal with delicate pop rock sweetness and dreamy charm',
    '리 내시': 'angelic ethereal female vocal with delicate pop rock sweetness and dreamy charm',
    '쉘리풀': 'vibrant energetic female vocal with 90s British guitar pop rock fizz',
    '쉘리 풀': 'vibrant energetic female vocal with 90s British guitar pop rock fizz',
    # ── 여성 팝록: 2000년대 팝 펑크 & 록 ──
    '미셸브랜치': 'clear ringing female vocal with acoustic-electric 2000s pop rock purity',
    '미셸 브랜치': 'clear ringing female vocal with acoustic-electric 2000s pop rock purity',
    '애슐리심슨': 'edgy teen-pop female vocal with punk-leaning pop rock attitude',
    '애슐리 심슨': 'edgy teen-pop female vocal with punk-leaning pop rock attitude',
    '페페돕슨': 'fierce punchy female vocal with Canadian pop rock fire and punk energy',
    '페페 돕슨': 'fierce punchy female vocal with Canadian pop rock fire and punk energy',
    'KT턴스톨': 'loop-station acoustic female vocal with percussive pop rock innovation and earthy power',
    'KT 턴스톨': 'loop-station acoustic female vocal with percussive pop rock innovation and earthy power',
    '에이미맥도널드': 'strong Scottish female vocal with driving guitar-based pop rock and folk edge',
    '에이미 맥도널드': 'strong Scottish female vocal with driving guitar-based pop rock and folk edge',
    '케이티화이트': 'catchy minimal female vocal with indie guitar pop rock hooks and playful energy',
    '케이티 화이트': 'catchy minimal female vocal with indie guitar pop rock hooks and playful energy',
    '안나날릭': 'tender emotive female vocal with piano-guitar pop rock sensitivity',
    '안나 날릭': 'tender emotive female vocal with piano-guitar pop rock sensitivity',
    '케이트보겔': 'polished piano-acoustic female vocal with reliable modern pop rock craftsmanship',
    '케이트 보겔': 'polished piano-acoustic female vocal with reliable modern pop rock craftsmanship',
    '마리온레이븐': 'fierce alternative female vocal with M2M origins evolving into edgy pop rock',
    '마리온 레이븐': 'fierce alternative female vocal with M2M origins evolving into edgy pop rock',
    '마리트라르센': 'delicate charming female vocal with Nordic acoustic pop rock sweetness',
    '마리트 라르센': 'delicate charming female vocal with Nordic acoustic pop rock sweetness',
    '스카이스윗남': 'bouncy teen-punk female vocal with colorful pop rock playfulness',
    '스카이 스윗남': 'bouncy teen-punk female vocal with colorful pop rock playfulness',
    '타샤레이에빈': 'guitar-slinging female vocal with all-girl band pop rock punch',
    '타샤레이 에빈': 'guitar-slinging female vocal with all-girl band pop rock punch',
    '제시카오리글리아소': 'twin-powered female vocal with Aussie electric pop rock intensity',
    '제시카 오리글리아소': 'twin-powered female vocal with Aussie electric pop rock intensity',
    # ── 여성 팝록: 어쿠스틱 & 인디 ──
    '파이스트': 'minimalist elegant female vocal with Canadian indie pop rock refinement and restraint',
    '파이 스트': 'minimalist elegant female vocal with Canadian indie pop rock refinement and restraint',
    '잉그리드마이클슨': 'charming quirky female vocal with ukulele-flavored acoustic pop rock warmth',
    '잉그리드 마이클슨': 'charming quirky female vocal with ukulele-flavored acoustic pop rock warmth',
    '어파인프렌지': 'crystalline piano-driven female vocal with organic pop rock delicacy',
    '어 파인 프렌지': 'crystalline piano-driven female vocal with organic pop rock delicacy',
    '렌카': 'bright bubbly Aussie female vocal with whimsical indie pop rock cheerfulness',
    '가브리엘아플린': 'translucent folk-pop female vocal with British acoustic pop rock purity',
    '가브리엘 아플린': 'translucent folk-pop female vocal with British acoustic pop rock purity',
    '로라마링': 'intellectual deep female vocal with sophisticated British folk rock artistry',
    '로라 마링': 'intellectual deep female vocal with sophisticated British folk rock artistry',
    '요한나소더버그': 'harmonious Swedish folk-pop female vocal with sisterly pop rock warmth',
    '요한나 소더버그': 'harmonious Swedish folk-pop female vocal with sisterly pop rock warmth',
    '레이첼야마가타': 'wistful emotional female vocal with indie rock ballad intimacy and raw beauty',
    '레이첼 야마가타': 'wistful emotional female vocal with indie rock ballad intimacy and raw beauty',
    '프리실라안': 'whispery gentle female vocal with minimal acoustic folk pop rock tenderness',
    '프리실라 안': 'whispery gentle female vocal with minimal acoustic folk pop rock tenderness',
    '캣파워': 'lo-fi smoky female vocal with vintage indie pop rock melancholy',
    '네코케이스': 'powerful resonant female vocal with americana-indie pop rock boldness',
    '네코 케이스': 'powerful resonant female vocal with americana-indie pop rock boldness',
    '샤론반이튼': 'expansive epic female vocal with heavy guitar indie pop rock gravitas',
    '샤론 반 이튼': 'expansive epic female vocal with heavy guitar indie pop rock gravitas',
    '엔젤올센': 'retro-laced emotive female vocal with cinematic indie pop rock drama',
    '엔젤 올센': 'retro-laced emotive female vocal with cinematic indie pop rock drama',
    '루시대커스': 'literary thoughtful female vocal with firm indie pop rock guitar riffs',
    '루시 대커스': 'literary thoughtful female vocal with firm indie pop rock guitar riffs',
    '줄리엔베이커': 'raw minimalist female vocal with devastating electric guitar pop rock honesty',
    '줄리엔 베이커': 'raw minimalist female vocal with devastating electric guitar pop rock honesty',
    # ── 여성 팝록: 현재 글로벌 스타 ──
    '올리비아로드리고': 'explosive Gen-Z female vocal reviving 90s alt and 2000s pop punk with fierce pop rock energy',
    '올리비아 로드리고': 'explosive Gen-Z female vocal reviving 90s alt and 2000s pop punk with fierce pop rock energy',
    '빌리아이리시': 'whisper-dark minimal female vocal with alternative pop rock cool and hypnotic intimacy',
    '빌리 아이리시': 'whisper-dark minimal female vocal with alternative pop rock cool and hypnotic intimacy',
    '매기로저스': 'organic folk-rock female vocal with stadium pop rock power and earthy groove',
    '매기 로저스': 'organic folk-rock female vocal with stadium pop rock power and earthy groove',
    '클라이로': 'soft lo-fi bedroom female vocal with hazy indie pop rock charm',
    '비바두비': 'jangly 90s-revival female vocal with nostalgic guitar pop rock sparkle',
    '홀리험버스톤': 'atmospheric British female vocal with moody modern guitar pop rock depth',
    '홀리 험버스톤': 'atmospheric British female vocal with moody modern guitar pop rock depth',
    '메이지피터스': 'catchy upbeat female vocal with punchy British pop rock hooks and storytelling flair',
    '메이지 피터스': 'catchy upbeat female vocal with punchy British pop rock hooks and storytelling flair',
    '그리프': 'innovative synth-forward British female vocal with modern pop rock production artistry',
    '채플론': 'glam 80s synth-rock female vocal with theatrical pop rock spectacle and icon energy',
    '채플 론': 'glam 80s synth-rock female vocal with theatrical pop rock spectacle and icon energy',
    '플레처': 'direct punchy female vocal with electric guitar-driven modern pop rock edge',
    '케이티개빈': 'sleek new wave female vocal with synth-pop rock sophistication and magnetic presence',
    '케이티 개빈': 'sleek new wave female vocal with synth-pop rock sophistication and magnetic presence',
    '조던밀러': 'cool guitar-riffing Canadian female vocal with classic pop rock swagger',
    '조던 밀러': 'cool guitar-riffing Canadian female vocal with classic pop rock swagger',
    '린지조던': 'prodigious lo-fi female vocal with raw indie pop rock guitar genius',
    '린지 조던': 'prodigious lo-fi female vocal with raw indie pop rock guitar genius',
    '소피앨리슨': 'nostalgic 90s-tinged female vocal with dreamy melodic indie pop rock',
    '소피 앨리슨': 'nostalgic 90s-tinged female vocal with dreamy melodic indie pop rock',
    '멜리나두테르테': 'multi-instrumental female vocal with lush dream pop rock textures',
    '멜리나 두테르테': 'multi-instrumental female vocal with lush dream pop rock textures',
    '브레이든레마스터스': 'youthful collaborative male vocal with bright modern indie pop rock sensibility',
    '브레이든 레마스터스': 'youthful collaborative male vocal with bright modern indie pop rock sensibility',
    '아비게일모리스': 'baroque art-pop female vocal with grandiose orchestral pop rock ambition',
    '아비게일 모리스': 'baroque art-pop female vocal with grandiose orchestral pop rock ambition',
    '사미아': 'candid witty female vocal with raw emotive indie pop rock storytelling',
    '지플립': 'powerhouse drumming Australian female vocal with punk-charged pop rock ferocity',
    '난나브린디스힐마르스도티르': 'ethereal Nordic female vocal with sweeping folk pop rock grandeur and Icelandic mystique',
    '난나 브린디스 힐마르스도티르': 'ethereal Nordic female vocal with sweeping folk pop rock grandeur and Icelandic mystique',
}


# ═══ CGO 보컬 믹서 1,000 프리셋 (cgo-390) ═══
# cat: A=강화형 B=충돌형 C=크로스 D=시대믹스
# tag: 사용자에게 표시되는 특성 라벨 (아티스트명 비노출)
# w: 3인 가중치 [%,%,%]
# prompt: Suno API에 전달되는 합성 보컬 프롬프트
VOICE_MIX: Dict[str, dict] = {
    'VM0001': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, gravelly uniquely husky deep male jazz vocal, one, heavy gravelly vocal'},
    'VM0002': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, laid-back mellow baritone, polished swinging male'},
    'VM0003': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, sticky deep husky female , immortal deep baritone male'},
    'VM0004': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, sticky deep husky female , smoky low alto'},
    'VM0005': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'husky deep mid-low adding mature depth, gravelly uniquely husky deep male jazz vocal, one, pop-future bass fairy'},
    'VM0006': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, sorrowful bending-note master, bold deep contralto'},
    'VM0007': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, charming deep baritone, velvety deep crooning'},
    'VM0008': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'immortal deep baritone who elevated music with noble dignity, mellow melodic male vocalists with addictive hooks and, heavy gravelly vocal'},
    'VM0009': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, mellow melodic male vocalists with addictive hooks and, polished swinging male'},
    'VM0010': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'dark European female vocalists commanding heavy trap, storytelling piano male vocal with warm gritty, slow heavyweight UK underground'},
    'VM0011': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, deep baritone male alternative rock vocal, brass-backed powerhouse male'},
    'VM0012': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'luxurious deep soulful baritone with rich gospel-touched warm resonance, dark European female vocalists commanding heavy trap, barefoot diva, deeply appealing'},
    'VM0013': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, heavyweight drum-and-bass male, intimate whispery'},
    'VM0014': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'deep heavy contralto female vocal singing the, deep baritone male alternative rock vocal, commanding'},
    'VM0015': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'barefoot diva, deeply appealing drawn from the depths of the heart, commanding , storytelling piano male'},
    'VM0016': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, brass-backed powerhouse with, heavyweight drum-and-bass male'},
    'VM0017': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'deep-voiced southern female vocalists with heavy 808 impact and raw visceral power, slow heavyweight UK underground, deep velvety'},
    'VM0018': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'husky theatrical baritone with bold unique projection and dramatic flair, most sophisticated calm sensual mid-low female, relaxed mellow baritone'},
    'VM0019': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, velvety deep crooning, clear steady male'},
    'VM0020': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'husky theatrical baritone with bold unique projection and dramatic flair, pop-future bass fairy, bold deep contralto'},
    'VM0021': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep-voiced male vocalist-producer who powered Death Row Records golden era sound, deep heavy charismatic low male vocal, commanding'},
    'VM0022': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavyweight commanding male vocalists with flawless flow and deep groove mastery, bold thick-toned female vocalists anchoring songs, husky theatrical baritone'},
    'VM0023': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, heavyweight commanding male, rustic mid-low bending-note'},
    'VM0024': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'brass-backed powerhouse with commanding stage energy, resonant deep baritone with dramatic anthemic, honest deep tenor'},
    'VM0025': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'whispery intimate ASMR vocal with dark, pop-future bass fairy, AlunaGeorge vocal pixie, refined'},
    'VM0026': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'immortal deep baritone who elevated music with noble dignity, bold thick-toned female vocalists anchoring songs, heavyweight drum-and-bass male'},
    'VM0027': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'laid-back mellow baritone with serene breezy minimal acoustic calm, all-range female spanning deep bass, rustic mid-low bending-note'},
    'VM0028': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'deep literary lyrical male ballad vocal with quiet resonance, timeless elegant male, suave romantic baritone'},
    'VM0029': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, immortal deep baritone male, deep heavy contralto'},
    'VM0030': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'dark European female vocalists commanding heavy trap, clear steady male pop- hiding deep, young prodigy female'},
    'VM0031': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, theatrical mysterious mid-low, timeless elegant male'},
    'VM0032': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'refined who distills deep traditional han into cinematic film-scale delivery, husky theatrical baritone with bold unique, mellow melodic male vocalists'},
    'VM0033': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, elegant baritone male , velvety deep crooning'},
    'VM0034': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'charming deep baritone male vocal, the king of rock and roll, honest deep tenor with raw sincerity and, husky deep mid-low female'},
    'VM0035': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'husky deep mid-low adding mature depth, polished swinging male, stable rich baritone'},
    'VM0036': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'husky deep mid-low adding mature, bold theatrical baritone with brassy big, suave romantic baritone'},
    'VM0037': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, heavy dubstep-trap hybrid, dark aggressive bass, young prodigy female'},
    'VM0038': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep-voiced male vocalist-producer who powered Death Row Records golden era sound, percussion-performing with deeply, smooth classic'},
    'VM0039': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavyweight drum-and-bass male vocalists with signature UK grime flow mastery, deep soul-laden mezzo-alto with mature tone, commanding dramatic diva'},
    'VM0040': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, barefoot diva, deeply appealing drawn from, percussion-performing male'},
    'VM0041': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'refined who distills deep traditional han into cinematic film-scale delivery, commanding heavy baritone with powerful sensual, dignified low-tone male'},
    'VM0042': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'relaxed mellow baritone with comfortable lush string-backed vocal ease, honest deep tenor with raw sincerity and, young prodigy female'},
    'VM0043': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'immortal deep baritone who elevated music with noble dignity, smoky low alto, heavy dubstep-trap hybrid'},
    'VM0044': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'bold theatrical baritone with brassy big band showman vocal energy, deep soul-laden mezzo-alto, bold thick-toned'},
    'VM0045': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep-voiced southern female vocalists with heavy 808 impact and raw visceral power, hook-driven addictive male , dancer-trained graceful female'},
    'VM0046': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'luxurious deep soulful baritone with rich gospel-touched warm resonance, whispery intimate ASMR vocal with dark, bold deep contralto'},
    'VM0047': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'mellow melodic male vocalists with addictive hooks and relaxed hybrid ballad delivery, smooth classic baritone male crooner, dreamy atmospheric'},
    'VM0048': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, charming deep baritone, most sophisticated calm'},
    'VM0049': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, commanding alto with passionate, deep heavy contralto'},
    'VM0050': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'heavyweight deep-bass male vocalists dominating grandiose trap beats with boss authority, most sophisticated calm sensual mid-low female, all-range female technician'},
    'VM0051': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, percussion-performing with deeply, smoky low alto'},
    'VM0052': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'commanding heavy baritone with powerful sensual deep soul growl, dignified low-tone male , whispery intimate ASMR'},
    'VM0053': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'rich earthy male baritone comforting working-class souls with rustic warmth, intimate whispery baritone with atmospheric, storytelling piano male'},
    'VM0054': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'elegant baritone singing life melancholy with refined literary grace, slow heavyweight UK underground, immortal deep baritone male'},
    'VM0055': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'clear steady male pop- hiding deep lyricism behind flashy performance, deep literary lyrical male ballad, storytelling piano male'},
    'VM0056': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'bold thick-toned female vocalists anchoring songs with signature mid-low delivery, mellow melodic male vocalists with addictive hooks and, polished swinging male'},
    'VM0057': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, refined who distills deep traditional, mournful mid-bass male'},
    'VM0058': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'immortal deep baritone who elevated music with noble dignity, radiant smooth tenor, heavy gravelly vocal'},
    'VM0059': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, most sophisticated calm sensual mid-low female, hook-driven addictive male'},
    'VM0060': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, deep literary lyrical male ballad, heavy dubstep-trap hybrid'},
    'VM0061': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'sticky deep husky at the pinnacle, commanding dramatic diva, relaxed mellow baritone'},
    'VM0062': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'percussion-performing with deeply sorrowful han-infused vocal power, minimalist acoustic male , smoky low alto'},
    'VM0063': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, relaxed mellow baritone with comfortable lush, dark European female'},
    'VM0064': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'R&B-infused female vocalists riding 808 glide bass , honest deep tenor with raw sincerity and, brass-backed powerhouse male'},
    'VM0065': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'immortal deep baritone who elevated music with noble dignity, pop-future bass fairy, AlunaGeorge vocal pixie, percussion-performing male'},
    'VM0066': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, romantic aged baritone with weathered folk, smoky low alto'},
    'VM0067': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, deep heavy charismatic low male vocal, heavyweight commanding male'},
    'VM0068': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep literary lyrical male ballad vocal with quiet resonance, minimalist acoustic with deep resonance, husky deep mid-low female'},
    'VM0069': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, deep heavy charismatic low male vocal, all-range female technician'},
    'VM0070': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, deep heavy contralto, commanding dramatic diva'},
    'VM0071': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'luxurious deep soulful baritone with rich gospel-touched warm resonance, whispery intimate ASMR vocal with dark, deep baritone male'},
    'VM0072': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'barefoot diva, deeply appealing drawn from, sorrowful bending-note master , dancer-trained graceful female'},
    'VM0073': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, deep velvety contralto with smoldering, heavyweight commanding male'},
    'VM0074': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, emerging female, commanding heavy baritone'},
    'VM0075': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, brass-backed powerhouse with, heavyweight commanding male'},
    'VM0076': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, brass-backed powerhouse with, storytelling piano male'},
    'VM0077': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'refined who distills deep traditional, smoky low alto with intimate atmospheric, husky deep mid-low female'},
    'VM0078': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'commanding heavy baritone with powerful sensual deep soul growl, thunderous deep-cave male vocalists who exploded Brooklyn , deep heavy contralto'},
    'VM0079': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep-voiced southern female vocalists with heavy 808 impact and raw visceral power, rich deep mid-low, velvety deep crooning'},
    'VM0080': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, stable rich baritone with sweeping epic phrasing, young prodigy female'},
    'VM0081': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, stable rich baritone with sweeping epic phrasing, polished swinging male'},
    'VM0082': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, timeless elegant male, polished swinging male'},
    'VM0083': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'dreamy atmospheric electronic vocal, deep, suave romantic baritone with elegant continental, husky theatrical baritone'},
    'VM0084': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep velvety contralto with smoldering sultry low-register warmth, mellow melodic male vocalists with addictive hooks and, honest deep tenor'},
    'VM0085': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'percussion-performing with deeply sorrowful han-infused vocal power, deep grand baritone with rich harmonic resonance, deep literary lyrical'},
    'VM0086': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'immortal deep baritone who elevated music with noble dignity, heavy gravelly vocal with raw hard-rock intensity, deep baritone male'},
    'VM0087': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'heavyweight deep-bass male vocalists dominating grandiose trap beats with boss authority, commanding heavy baritone with powerful sensual, honest deep tenor'},
    'VM0088': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'hook-driven addictive dominating with catchy refrains and deep emotion, dreamy atmospheric electronic vocal, deep, immortal deep baritone male'},
    'VM0089': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, timeless elegant male, most sophisticated calm'},
    'VM0090': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'suave romantic baritone with elegant continental, heavy dubstep-trap hybrid, dark aggressive bass, sticky deep neo-soul'},
    'VM0091': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, resonant deep baritone with dramatic anthemic, elegant baritone male'},
    'VM0092': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, deep soul-laden mezzo-alto with mature tone, heavyweight drum-and-bass male'},
    'VM0093': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'slow heavyweight UK underground male vocalists with iconic deep bass flow delivery, deep velvety, whispery intimate ASMR'},
    'VM0094': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, slow heavyweight UK underground, pop-future bass fairy'},
    'VM0095': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, commanding dramatic diva vocal with explosive, velvety deep crooning'},
    'VM0096': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'rich earthy male baritone comforting working-class souls with rustic warmth, gravelly uniquely husky deep, warm charismatic'},
    'VM0097': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'heavyweight deep-bass male vocalists dominating grandiose trap, rich earthy male baritone comforting working-class, whispery intimate ASMR'},
    'VM0098': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, polished swinging male crooner vocal with, dark European female'},
    'VM0099': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'dignified low-tone who sings like reciting poetry with gravitas, bold deep contralto, bold theatrical baritone'},
    'VM0100': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'thunderous deep-cave male vocalists who exploded Brooklyn , deep soul-laden mezzo-alto with mature tone, commanding'},
    'VM0101': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one, intimate whispery baritone with atmospheric, bold thick-toned'},
    'VM0102': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, deep heavy contralto, thunderous deep-cave male vocalists'},
    'VM0103': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'all-range female spanning deep bass, bold thick-toned female vocalists anchoring songs, refined'},
    'VM0104': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, timeless elegant male jazz crooner vocal, honest deep tenor'},
    'VM0105': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, sticky deep neo-soul male vocal that, polished swinging male'},
    'VM0106': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'elegant baritone singing life melancholy with refined literary grace, thunderous deep-cave male vocalists, heavyweight commanding male'},
    'VM0107': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'all-range female spanning deep bass to soaring high notes, timeless elegant male jazz crooner vocal, R&B-infused female vocalists riding'},
    'VM0108': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, deep grand baritone, deep baritone male'},
    'VM0109': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'elegant baritone singing life melancholy with refined literary grace, charming deep baritone male vocal, the king, dancer-trained graceful female'},
    'VM0110': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'bold theatrical baritone with brassy big band showman vocal energy, heavyweight drum-and-bass male vocalists with signature, elegant baritone male'},
    'VM0111': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavyweight deep-bass male vocalists dominating grandiose trap beats with boss authority, heavyweight commanding male, luxurious deep soulful'},
    'VM0112': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, husky deep mid-low female, deep soul-laden mezzo-alto'},
    'VM0113': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'rich earthy male baritone comforting working-class souls with rustic warmth, deep-voiced male vocalist-producer, bold thick-toned'},
    'VM0114': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, dreamy atmospheric electronic vocal, deep, commanding heavy baritone'},
    'VM0115': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'sorrowful bending-note master with deeply mournful delivery, dark European female vocalists commanding heavy trap, warm charismatic'},
    'VM0116': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, bold thick-toned female vocalists anchoring songs, dreamy atmospheric'},
    'VM0117': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, rich deep mid-low, deep velvety'},
    'VM0118': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep-voiced male vocalist-producer who powered Death Row Records golden era sound, clear steady male, hook-driven addictive male'},
    'VM0119': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, all-range female technician, dignified low-tone male'},
    'VM0120': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'relaxed mellow baritone with comfortable lush string-backed vocal ease, barefoot diva, deeply appealing drawn from, rich deep mid-low'},
    'VM0121': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavyweight deep-bass male vocalists dominating grandiose trap beats with boss authority, sticky deep husky, commanding dramatic diva'},
    'VM0122': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'deep soul-laden mezzo-alto with mature tone and moody R&B depth, bold deep contralto with distinctive vibrato, bold theatrical baritone'},
    'VM0123': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, deep baritone male, whispery intimate ASMR'},
    'VM0124': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'most sophisticated calm sensual mid-low female vocal with luxury tone, brass-backed powerhouse male, sorrowful bending-note master'},
    'VM0125': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, R&B-infused female vocalists riding, rustic mid-low bending-note'},
    'VM0126': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, resonant deep baritone with dramatic anthemic, charming deep baritone'},
    'VM0127': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'immortal deep baritone who elevated music with noble dignity, bold deep contralto, warm charismatic'},
    'VM0128': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'dancer-trained graceful with clear deep lyrical vocal delivery, clear steady male pop- hiding deep, deep literary lyrical'},
    'VM0129': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, luxurious deep soulful baritone with rich, brass-backed powerhouse male'},
    'VM0130': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'clear steady male pop- hiding deep lyricism behind flashy performance, young prodigy narrating deep life, pop-future bass fairy'},
    'VM0131': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'heavyweight commanding male vocalists with flawless flow and deep groove mastery, hook-driven addictive male , husky deep mid-low female'},
    'VM0132': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, mournful mid-bass male, velvety deep crooning'},
    'VM0133': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'dignified low-tone who sings like reciting poetry with gravitas, mournful mid-bass commanding, honest deep tenor'},
    'VM0134': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, dancer-trained graceful female, percussion-performing male'},
    'VM0135': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, heavy gravelly vocal with raw hard-rock intensity, hook-driven addictive male'},
    'VM0136': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'rich deep mid-low voice with velvety soul, commanding dramatic diva vocal with explosive, husky theatrical baritone'},
    'VM0137': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'heavyweight commanding male vocalists with flawless flow and deep groove mastery, heavy dubstep-trap hybrid, dark aggressive bass, barefoot diva, deeply appealing'},
    'VM0138': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, commanding , R&B-infused female vocalists riding'},
    'VM0139': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, smoky low alto, deep heavy contralto'},
    'VM0140': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'rich earthy male baritone comforting working-class souls with rustic warmth, heavyweight deep-bass male, heavy dubstep-trap hybrid'},
    'VM0141': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'sorrowful bending-note master with deeply mournful delivery, deep heavy charismatic, deep velvety'},
    'VM0142': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'R&B-infused female vocalists riding 808 glide bass with smooth vocal elegance, smooth classic, dark European female'},
    'VM0143': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'charming deep baritone male vocal, the king of rock and roll, slow heavyweight UK underground, stable rich baritone'},
    'VM0144': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'sorrowful bending-note master with deeply mournful delivery, refined who distills deep traditional, heavy dubstep-trap hybrid'},
    'VM0145': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, percussion-performing male , husky theatrical baritone'},
    'VM0146': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'percussion-performing with deeply, gravelly uniquely husky deep male jazz vocal, one, R&B-infused female vocalists riding'},
    'VM0147': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'bold thick-toned female vocalists anchoring songs with signature mid-low delivery, relaxed mellow baritone with comfortable lush, clear steady male'},
    'VM0148': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'refined who distills deep traditional han into cinematic film-scale delivery, gravelly uniquely husky deep male jazz vocal, one, emerging female'},
    'VM0149': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, barefoot diva, deeply appealing drawn from, immortal deep baritone male'},
    'VM0150': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'clear steady male pop- hiding deep, radiant smooth tenor with luminous Latin, deep literary lyrical'},
    'VM0151': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'dark European female vocalists commanding heavy trap beats with ominous presence, deep-voiced male vocalist-producer, smoky low alto'},
    'VM0152': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'deep-voiced male vocalist-producer who powered Death Row, mellow melodic male vocalists with addictive hooks and, dark European female'},
    'VM0153': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'sorrowful bending-note master with deeply mournful delivery, deep grand baritone, barefoot diva, deeply appealing'},
    'VM0154': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'smooth classic baritone male crooner, heavy dubstep-trap hybrid, dark aggressive bass, deep literary lyrical'},
    'VM0155': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, deep heavy contralto female vocal singing the, deep baritone male'},
    'VM0156': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, bold theatrical baritone with brassy big, smooth classic'},
    'VM0157': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, commanding dramatic diva, commanding heavy baritone'},
    'VM0158': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, smoky low alto, clear steady male'},
    'VM0159': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, deep-voiced southern female vocalists with heavy 808 impact, relaxed mellow baritone'},
    'VM0160': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, velvety deep crooning baritone with effortless, deep soul-laden mezzo-alto'},
    'VM0161': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, deep soul-laden mezzo-alto, dark European female'},
    'VM0162': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'bold theatrical baritone with brassy big band showman vocal energy, most sophisticated calm sensual mid-low female, whispery intimate ASMR'},
    'VM0163': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'stable rich baritone with sweeping epic phrasing, luxurious deep soulful baritone with rich, deep baritone male'},
    'VM0164': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'refined who distills deep traditional han into cinematic film-scale delivery, immortal deep baritone who elevated, deep-voiced southern female vocalists'},
    'VM0165': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'husky deep mid-low adding mature depth, mellow melodic male vocalists, theatrical mysterious mid-low'},
    'VM0166': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'percussion-performing with deeply, polished swinging male crooner vocal with, elegant baritone male'},
    'VM0167': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'brass-backed powerhouse with commanding stage energy, suave romantic baritone with elegant continental, elegant baritone male'},
    'VM0168': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'dancer-trained graceful with clear deep lyrical vocal delivery, husky theatrical baritone, refined'},
    'VM0169': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'clear steady male pop- hiding deep, elegant baritone singing life melancholy, husky theatrical baritone'},
    'VM0170': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, sticky deep husky, heavy gravelly vocal'},
    'VM0171': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'whispery intimate ASMR vocal with dark, clear steady male pop- hiding deep, dark European female'},
    'VM0172': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, rustic mid-low bending-note , deep-voiced male vocalist-producer'},
    'VM0173': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'percussion-performing with deeply, smoky low alto with intimate atmospheric, slow heavyweight UK underground'},
    'VM0174': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'mellow melodic male vocalists with addictive hooks and relaxed hybrid ballad delivery, heavy gravelly vocal with raw hard-rock intensity, pop-future bass fairy'},
    'VM0175': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'dancer-trained graceful with clear deep lyrical vocal delivery, intimate whispery baritone with atmospheric, barefoot diva, deeply appealing'},
    'VM0176': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'rich earthy male baritone comforting working-class souls with rustic warmth, percussion-performing with deeply, pop-future bass fairy'},
    'VM0177': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'dignified low-tone who sings like reciting poetry with gravitas, percussion-performing with deeply, mournful mid-bass male'},
    'VM0178': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'sticky deep husky at the pinnacle, romantic aged baritone, bold deep contralto'},
    'VM0179': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, deep soul-laden mezzo-alto, deep-voiced southern female vocalists'},
    'VM0180': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, young prodigy female , thunderous deep-cave male vocalists'},
    'VM0181': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, heavyweight deep-bass male, slow heavyweight UK underground'},
    'VM0182': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'brass-backed powerhouse with, husky theatrical baritone with bold unique, deep soul-laden mezzo-alto'},
    'VM0183': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'commanding alto with passionate, luxurious deep soulful baritone with rich, intimate whispery'},
    'VM0184': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'luxurious deep soulful baritone with rich gospel-touched warm resonance, refined who distills deep traditional, radiant smooth tenor'},
    'VM0185': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, minimalist acoustic male , heavy gravelly vocal'},
    'VM0186': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'young prodigy narrating deep life stories with mature emotional arc, rich deep mid-low voice with velvety soul, storytelling piano male'},
    'VM0187': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, commanding alto with passionate, heavyweight drum-and-bass male'},
    'VM0188': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, theatrical mysterious mid-low male vocal with, refined'},
    'VM0189': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, deep baritone male alternative rock vocal, deep literary lyrical'},
    'VM0190': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, commanding heavy baritone with powerful sensual, percussion-performing male'},
    'VM0191': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, rich deep mid-low, dignified low-tone male'},
    'VM0192': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'hook-driven addictive dominating with catchy refrains and deep emotion, mellow melodic male vocalists with addictive hooks and, radiant smooth tenor'},
    'VM0193': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, minimalist acoustic with deep resonance, timeless elegant male'},
    'VM0194': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'bold thick-toned female vocalists anchoring songs with signature mid-low delivery, percussion-performing with deeply, most sophisticated calm'},
    'VM0195': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep literary lyrical male ballad vocal with quiet resonance, suave romantic baritone with elegant continental, stable rich baritone'},
    'VM0196': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, luxurious deep soulful, pop-future bass fairy'},
    'VM0197': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, dancer-trained graceful female, dark European female'},
    'VM0198': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'hook-driven addictive dominating with catchy, stable rich baritone with sweeping epic phrasing, immortal deep baritone male'},
    'VM0199': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, storytelling piano male, deep velvety'},
    'VM0200': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, honest deep tenor with raw sincerity and, barefoot diva, deeply appealing'},
    'VM0201': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, warm charismatic baritone with calypso-tinged, dreamy atmospheric'},
    'VM0202': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, charming deep baritone, rich earthy male'},
    'VM0203': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'dignified low-tone who sings like reciting poetry with gravitas, romantic aged baritone with weathered folk, sticky deep neo-soul'},
    'VM0204': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, brass-backed powerhouse male, most sophisticated calm'},
    'VM0205': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'commanding alto with passionate 90s ballad grit, honest deep tenor, bold theatrical baritone'},
    'VM0206': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, heavy dubstep-trap hybrid, bold deep contralto'},
    'VM0207': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, husky theatrical baritone with bold unique, resonant deep baritone'},
    'VM0208': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, relaxed mellow baritone with comfortable lush, deep soul-laden mezzo-alto'},
    'VM0209': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, hook-driven addictive male , luxurious deep soulful'},
    'VM0210': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, bold thick-toned female vocalists anchoring songs, husky theatrical baritone'},
    'VM0211': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, rustic mid-low bending-note , hook-driven addictive male'},
    'VM0212': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, elegant baritone singing life melancholy, deep-voiced male vocalist-producer'},
    'VM0213': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, gravelly uniquely husky deep male jazz vocal, one, sticky deep husky'},
    'VM0214': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'smooth classic baritone male crooner jazz pop vocal, husky theatrical baritone with bold unique, deep baritone male'},
    'VM0215': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'dignified low-tone who sings like, dancer-trained graceful with clear, hook-driven addictive male'},
    'VM0216': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, elegant baritone singing life melancholy, percussion-performing male'},
    'VM0217': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'dignified low-tone who sings like, thunderous deep-cave male vocalists who exploded Brooklyn , rich earthy male'},
    'VM0218': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'deep-voiced male vocalist-producer who powered Death Row, polished swinging male crooner vocal with, velvety deep crooning'},
    'VM0219': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'rich earthy male baritone comforting working-class souls with rustic warmth, rustic mid-low bending-note, rich deep mid-low'},
    'VM0220': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'bold thick-toned female vocalists anchoring songs with signature mid-low delivery, dreamy atmospheric, sticky deep neo-soul'},
    'VM0221': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'sticky deep husky female , heavyweight commanding male vocalists with flawless flow, deep grand baritone'},
    'VM0222': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, bold thick-toned female vocalists anchoring songs, all-range female technician'},
    'VM0223': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, clear steady male pop- hiding deep, rich earthy male'},
    'VM0224': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, dancer-trained graceful with clear, heavyweight drum-and-bass male'},
    'VM0225': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'rustic mid-low bending-note anchoring legendary harmony foundations, hook-driven addictive dominating with catchy, deep literary lyrical'},
    'VM0226': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'husky theatrical baritone with bold unique, dreamy atmospheric electronic vocal, deep, commanding heavy baritone'},
    'VM0227': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'hook-driven addictive dominating with catchy refrains and deep emotion, percussion-performing with deeply, luxurious deep soulful'},
    'VM0228': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, velvety deep crooning baritone with effortless, deep soul-laden mezzo-alto'},
    'VM0229': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, dreamy atmospheric electronic vocal, deep, rich earthy male'},
    'VM0230': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, timeless elegant male jazz crooner vocal, clear steady male'},
    'VM0231': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, rustic mid-low bending-note , romantic aged baritone'},
    'VM0232': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'husky theatrical baritone with bold unique projection and dramatic flair, sorrowful bending-note master, minimalist acoustic male'},
    'VM0233': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, brass-backed powerhouse with, dreamy atmospheric'},
    'VM0234': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'R&B-infused female vocalists riding 808 glide bass with smooth vocal elegance, intimate whispery, all-range female technician'},
    'VM0235': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'laid-back mellow baritone with serene breezy minimal acoustic calm, deep heavy contralto, heavy gravelly vocal'},
    'VM0236': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'mellow melodic male vocalists with addictive hooks and relaxed hybrid ballad delivery, husky deep mid-low adding mature, husky theatrical baritone'},
    'VM0237': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, stable rich baritone with sweeping epic phrasing, percussion-performing male'},
    'VM0238': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'mournful mid-bass commanding, refined who distills deep traditional, minimalist acoustic male'},
    'VM0239': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'heavyweight deep-bass male vocalists dominating grandiose trap beats with boss authority, dreamy atmospheric, heavy dubstep-trap hybrid'},
    'VM0240': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'brass-backed powerhouse with commanding stage energy, deep-voiced male vocalist-producer, intimate whispery'},
    'VM0241': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, R&B-infused female vocalists riding, commanding'},
    'VM0242': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, heavyweight commanding male, dignified low-tone male'},
    'VM0243': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'dark European female vocalists commanding heavy trap beats with ominous presence, R&B-infused female vocalists riding, suave romantic baritone'},
    'VM0244': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'sticky deep neo-soul male vocal that melts the heart, commanding heavy baritone with powerful sensual, husky deep mid-low female'},
    'VM0245': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavyweight deep-bass male vocalists dominating grandiose trap beats with boss authority, storytelling piano male vocal with warm gritty, hook-driven addictive male'},
    'VM0246': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'young prodigy narrating deep life stories with mature emotional arc, brass-backed powerhouse male, theatrical mysterious mid-low'},
    'VM0247': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'resonant deep baritone with dramatic anthemic folk pop vocal projection, rich earthy male baritone comforting working-class, suave romantic baritone'},
    'VM0248': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'dignified low-tone who sings like reciting poetry with gravitas, honest deep tenor with raw sincerity and, mellow melodic male vocalists'},
    'VM0249': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'emerging female vocalists commanding heavy 808 beats with bold presence, barefoot diva, deeply appealing, heavyweight deep-bass male'},
    'VM0250': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'refined who distills deep traditional han into cinematic film-scale delivery, dark European female, heavyweight commanding male'},
    'VM0251': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'percussion-performing with deeply sorrowful han-infused vocal power, beast-like husky male vocal with emotional, globally verified female'},
    'VM0252': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'soaring rock soprano with legendary high-range stadium power, angsty alternative mezzo with confessional, warm low-register male'},
    'VM0253': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, most destructive female rock vocal in, elegant 60s female'},
    'VM0254': {'cat':'B','tag':'중저음 매력','w':[40, 40, 20],'prompt':'stable rich baritone with sweeping epic phrasing, genius male vocal combining demonic growling, authoritative smooth male'},
    'VM0255': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, ethereal crystalline soprano with gentle, commanding male vocalists'},
    'VM0256': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'resonant deep baritone with dramatic anthemic folk pop vocal projection, plaintive soaring falsetto with vulnerable intimate, master-architect male'},
    'VM0257': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, raw convulsive gravelly male vocal wringing every, commanding male vocalists'},
    'VM0258': {'cat':'B','tag':'압도적 고음','w':[50, 30, 20],'prompt':'perfectionist tenor with powerful live projection and orchestral vocal precision, refreshing bright rock soprano with crisp attack, charming mid-low female'},
    'VM0259': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'resonant deep baritone with dramatic anthemic folk pop vocal projection, gentle meditative tenor, idol-trained groovy female'},
    'VM0260': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'brass-backed powerhouse with commanding stage energy, rich commanding contralto with majestic, cute warm soprano with'},
    'VM0261': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'thick soulful showing peak sorrowful delivery with gomtang warmth, warm earthy alto with tender, genius producing female'},
    'VM0262': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'crystal clear yet steel-strong female belting high vocal filling stadiums, elegant 60s layering sophisticated, clear pristine soprano'},
    'VM0263': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'immortal deep baritone who elevated, modern layered R&B alto with dense, pristine classical'},
    'VM0264': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'global trendy tenor with cinematic pop polish and youthful dynamic range, spicy capsaicin-sharp with traditional, warm intimate male'},
    'VM0265': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'slow heavyweight UK underground male vocalists with iconic deep bass flow delivery, massive operatic soprano with stadium-shaking, raw desperate soprano'},
    'VM0266': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'earthy rustic combining rural folk sentiment with tradition, tender longing falsetto carrying sorrowful romantic, explosive female disco'},
    'VM0267': {'cat':'B','tag':'허스키 감성','w':[70, 20, 10],'prompt':'the king , versatile male vocal covering rock ballad and folk, creative fusion soprano, idol-crossover female'},
    'VM0268': {'cat':'B','tag':'허스키 감성','w':[50, 30, 20],'prompt':'modern breathy alternative R&B vocal with trendy melodic sensibility, gentle refined delivering traditional, smooth R&B singing over'},
    'VM0269': {'cat':'B','tag':'압도적 고음','w':[50, 30, 20],'prompt':'romantic emotional tenor with soaring rock-ballad phrasing and soft power, pristine smooth falsetto with effortless high, Latin reggaeton- crossover'},
    'VM0270': {'cat':'B','tag':'압도적 고음','w':[60, 20, 20],'prompt':'majestic operatic tenor with soaring classical purity and grand resonance, bright pure indie, dreamy atmospheric house'},
    'VM0271': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, refreshing clear soprano, authoritative male vocalists'},
    'VM0272': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, trendy urban mezzo, Grammy-winning EDM topline'},
    'VM0273': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'heavyweight drum-and-bass male vocalists with signature, explosive power from small frame, timeless clear, unique delicate female'},
    'VM0274': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'sorrowful bending-note master with deeply mournful delivery, sandpaper-rough charming male vocal with loose, punchy dynamic male'},
    'VM0275': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'pioneering with signature vibrato who, pure crystalline tenor with emotionally transparent, inventive creative female vocalists'},
    'VM0276': {'cat':'B','tag':'허스키 감성','w':[70, 20, 10],'prompt':'gritty warm male keyboard-soul vocal with bluesy rasp and heartfelt punch, honest unpretentious warm, mysterious Eastern pentatonic'},
    'VM0277': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'raw desperate soprano with unfiltered emotional intensity and urgent vocal power, warm low-register evoking hometown, revolutionary UK grime male'},
    'VM0278': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'most sophisticated calm sensual mid-low female vocal with luxury tone, ultimate female, rough raspy alto'},
    'VM0279': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'emerging female vocalists commanding heavy 808 beats with bold presence, pioneering , trendy stylish male'},
    'VM0280': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, dance anthem, warm intimate male'},
    'VM0281': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, explosive raspy male , crystal-clear healing female'},
    'VM0282': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, edgy youthful alto with emotional, angelic Irish ensemble'},
    'VM0283': {'cat':'B','tag':'압도적 고음','w':[60, 20, 20],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, inventive jazzy soprano, underground legend male'},
    'VM0284': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'clear steady male pop- hiding deep lyricism behind flashy performance, husky powerful, ethereal breathtaking soprano'},
    'VM0285': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'laid-back mellow baritone with serene breezy, stadium-filling resonant male vocal with, genius sensual male vocal'},
    'VM0286': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, fire-breathing piercing metallic high female, gritty soulful'},
    'VM0287': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'powerful husky high tenor with dramatic intensity and piercing climactic notes, textbook traditional female , polished velvety'},
    'VM0288': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, rugged bending-note male , radically alternative female'},
    'VM0289': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, bel-canto creating cinematic time-slip, explosive raspy female vocalists'},
    'VM0290': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'gayageum-playing female hybrid bridging traditional, inventive jazzy soprano with playful harmonic twists, fierce female vocalists from'},
    'VM0291': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'deep-voiced southern female vocalists with heavy 808 impact and raw visceral power, clear earnest tenor, soft dreamy'},
    'VM0292': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'versatile clear tenor with explosive high notes, unique sophisticated neo-soul queen female vocal, sweet lyrical soprano'},
    'VM0293': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'transcendent tenor with flawless breath control, nervous yet beautiful dreamy falsetto male, witty transatlantic female vocalists'},
    'VM0294': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'mysterious powerful gothic female rock vocal piercing through dark orchestral sound, textbook traditional female , unique delicate female'},
    'VM0295': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'dramatic operatic male vocal with soaring, the king , versatile male vocal, crystalline nightingale female'},
    'VM0296': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'world-class 5-octave powerful with dramatic high notes and pop diva power, gravelly soulful midrange with Celtic, warm folk acoustic'},
    'VM0297': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'clear earnest tenor with sweeping poetic folk storytelling grandeur, edgy youthful alto with emotional, delicate airy'},
    'VM0298': {'cat':'B','tag':'허스키 감성','w':[50, 30, 20],'prompt':'raw rough soul-shaking male rock vocal with emotional honesty, tender longing falsetto carrying sorrowful romantic, hard-hitting gangster male'},
    'VM0299': {'cat':'B','tag':'압도적 고음','w':[60, 30, 10],'prompt':'majestic operatic tenor with soaring classical purity and grand resonance, pharmacist-turned with crystalline falsetto, commanding male vocalists'},
    'VM0300': {'cat':'B','tag':'중저음 매력','w':[40, 40, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one, deep husky soulful female ballad vocal, rhythmic male pop'},
    'VM0301': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'gritty soulful groove vocal with British white-soul rasp, pure clean soprano capturing quiet depth, trend-setting'},
    'VM0302': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'warm robust tenor with grand sweeping romantic pop balladry and passion, rough torn raspy, soft breathy warm'},
    'VM0303': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'young prodigy narrating deep life stories with mature emotional arc, classical soprano, flawless classic female'},
    'VM0304': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'heavyweight drum-and-bass male vocalists with signature UK grime flow mastery, doll-faced female , bouncy yet heartfelt'},
    'VM0305': {'cat':'B','tag':'압도적 고음','w':[60, 20, 20],'prompt':'raw fragile tenor building from whisper to intense acoustic crescendo, inventive jazzy soprano, authoritative smooth male'},
    'VM0306': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'soulful mezzo-soprano with flawless R&B scale technique and rich harmony, pristine classical crossover soprano with, dreamy Latin-pop female'},
    'VM0307': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'minimalist acoustic with deep resonance on simple folk melodies, multi-genre soprano with, smooth modern country'},
    'VM0308': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'percussion-performing with deeply sorrowful han-infused vocal power, raw powerful female vocalists conquering Billboard with, quiet warm soothing'},
    'VM0309': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'resonant deep baritone with dramatic anthemic, flawless classic female vocal mastering Broadway and, soft breathy warm'},
    'VM0310': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, pure crystal-clear folk soprano with gentle, historic west-coast crew'},
    'VM0311': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'brass-backed powerhouse with commanding stage energy, cinematic dubstep, creative fusion soprano'},
    'VM0312': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, clear earnest tenor, silvery celestial country'},
    'VM0313': {'cat':'B','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'world-class soprano with soaring operatic power and pristine cinematic projection, devastating power-ballad soprano, crystal-clear healing female'},
    'VM0314': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'sorrowful bending-note master with deeply mournful delivery, fire-breathing piercing, globally distinctive soprano'},
    'VM0315': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, genius singer-songwriter male, bright cheerful male'},
    'VM0316': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'deep heavy contralto female vocal singing the Black soul with gravitas, queen of soul, gospel-based explosive powerful female vocal, rebellious melancholic raw retro'},
    'VM0317': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, distinctive nasal indie tenor with quirky charm, deep husky soulful'},
    'VM0318': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'pop-future bass fairy, AlunaGeorge vocal pixie, light sparkling tone, lyrical light tenor with airy French-pop-influenced, cute warm soprano with'},
    'VM0319': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'slow heavyweight UK underground male vocalists with iconic, powerfully raspy male vocalists with aggressive west-coast, rough torn raspy'},
    'VM0320': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'soaring popera soprano with theatrical cinematic grandeur and power, polished warm soprano with elegant 60s, modern male vocalists'},
    'VM0321': {'cat':'B','tag':'압도적 고음','w':[70, 20, 10],'prompt':'agile scatting tenor blending jazz improvisation with smooth pop finesse, bright cheerful male, razor-precise female'},
    'VM0322': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'sad sharp Irish traditional female vocal with sorrowful piercing tone, underground gritty male vocalists with raw , emotional healing trance'},
    'VM0323': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, sophisticated mid-range vocal with refined phrasing, flawless crystal clear'},
    'VM0324': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'legendary nasal-melody who comforted a colonized nation with sorrow, emotional healing trance, angelic vocal melodies, NYC underground queen'},
    'VM0325': {'cat':'B','tag':'압도적 고음','w':[60, 20, 20],'prompt':'heart-tearing sorrowful explosive male soul vocal, creative fusion soprano, razor-precise female'},
    'VM0326': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, broken sobbing male, bright pure tenor'},
    'VM0327': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'trendy urban mezzo with tension-filled chord sensibility and sultry phrasing, warm versatile mezzo with theatrical, explosive Canadian male'},
    'VM0328': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'ultimate female high-note queen with blade-sharp piercing rapid vocal delivery, pansori-infused cinematic with elaborate melodic, warm folk acoustic'},
    'VM0329': {'cat':'B','tag':'중저음 매력','w':[50, 40, 10],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, husky soulful female vocal fusing and, cute bright female'},
    'VM0330': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'sweeping dramatic tenor with lush symphonic, rugged bending-note cutting through grand, pristine classical'},
    'VM0331': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'distinctive nasal indie tenor with quirky charm, trendy urban mezzo with tension-filled chord, massive cinematic soprano'},
    'VM0332': {'cat':'B','tag':'허스키 감성','w':[50, 30, 20],'prompt':'rebellious melancholic raw retro soul jazz female vocal, one of a kind tone, angelic fragile yet devastating falsetto male, technically gifted male'},
    'VM0333': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'the king , versatile male vocal, bright pure indie soprano with cheerful Hongdae, pioneering male vocalists'},
    'VM0334': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'immortal deep baritone who elevated music with noble dignity, - queen female vocalist, elegant 60s female'},
    'VM0335': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, androgynous cold urban, clear pristine soprano'},
    'VM0336': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, - queen female vocalist, pure crystalline tenor'},
    'VM0337': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, refined French chanteuse with classic, delicate yodeling folk'},
    'VM0338': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'charming deep baritone male vocal, the king of rock and roll, refined velvety tenor with elegant soaring, bright youthful'},
    'VM0339': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'deep classic husky female vocal with, folk-rooted gentle comforting the nation, dreamy alternative male'},
    'VM0340': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'luxurious deep soulful baritone with rich gospel-touched warm resonance, genius rhythmic soulful, haunting atmospheric'},
    'VM0341': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'warm charismatic baritone with calypso-tinged folk orchestral humanity, explosive power from small frame, timeless clear, explosive raspy male'},
    'VM0342': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'commanding alto with passionate 90s ballad grit, trance vocal queen, NYC underground queen'},
    'VM0343': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'rugged male vocalists blending gritty tone with gangster balladry and west-coast soul, quiet warm soothing, witty transatlantic female vocalists'},
    'VM0344': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'elegant baritone singing life melancholy with refined literary grace, powerful heartfelt classic pop male vocal, pristine classical'},
    'VM0345': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'globally distinctive soprano with unique nasal R&B color and emotional crack, flawless crystal clear tenor cutting through, legendary Three 6 Mafia'},
    'VM0346': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'sorrowful bending-note master with deeply mournful delivery, piercing powerful, androgynous cold urban'},
    'VM0347': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep-voiced male vocalist-producer who powered Death Row Records golden era sound, clear bright female pop vocal, pharmacist-turned female'},
    'VM0348': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'heavyweight drum-and-bass male vocalists with signature UK grime flow mastery, clear bright female pop vocal with ultra-high technique, hauntingly beautiful female'},
    'VM0349': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'hard-hitting slide- male vocalists with powerful 808 bass-riding technique, pansori-infused cinematic with elaborate melodic, pure crystalline tenor'},
    'VM0350': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'deep soul-laden mezzo-alto with mature tone, screaming high tenor with razor-sharp power, refined French'},
    'VM0351': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'raw convulsive gravelly male vocal wringing every note with blues agony, quiet warm soothing, globally acclaimed UK'},
    'VM0352': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'young prodigy narrating deep life, Miss champion female vocalist with overwhelming pansori-based, airy ethereal male'},
    'VM0353': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'resonant deep baritone with dramatic anthemic folk pop vocal projection, gritty soulful groove vocal with, warm intimate male'},
    'VM0354': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'bold theatrical baritone with brassy big band showman vocal energy, sandpaper-rough charming male vocal with loose, creative fusion soprano'},
    'VM0355': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, transparent fragile male, power pop-rock EDM'},
    'VM0356': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'clear steady male pop- hiding deep lyricism behind flashy performance, original all-rounder , rhythmic powerhouse'},
    'VM0357': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, legendary harmony female, warm folk acoustic'},
    'VM0358': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'stable rich baritone with sweeping epic phrasing, blended operatic tenor ensemble with lush, deep classic husky female'},
    'VM0359': {'cat':'B','tag':'압도적 고음','w':[70, 20, 10],'prompt':'crystal clear yet steel-strong female belting high vocal filling stadiums, intense dramatic, Terror Squad pride female'},
    'VM0360': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'deep heavy charismatic low male vocal, distinctive nasal indie tenor with quirky charm, pure refreshing female'},
    'VM0361': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'sticky deep husky at the pinnacle, solid expressive female , transparent dewdrop-clear soprano'},
    'VM0362': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'explosive operatic metal male vocal like a human air raid siren, layered multitrack choral vocal creating vast, global EDM hit'},
    'VM0363': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, rhythmic powerhouse, microtonal Arab-maqam female'},
    'VM0364': {'cat':'B','tag':'압도적 고음','w':[60, 30, 10],'prompt':'The Voice, perfect female vocal with flawless power pitch and emotion, nervous yet beautiful dreamy falsetto male, polished Atlanta trap'},
    'VM0365': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'emotionally charged soprano with blockbuster string-ballad intensity and raw feeling, street-style diva soprano with massive volume and, relentless rapid-fire male'},
    'VM0366': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'warm charismatic baritone with calypso-tinged folk orchestral humanity, flawless technique male, paradigm-shifting male vocalist-producer'},
    'VM0367': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'hook-driven addictive dominating with catchy, anthem trance, heart-wrenching melodies, powerful emotional, lush romantic male'},
    'VM0368': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'rich deep mid-low voice with velvety soul, traditional bending-note technician , bright narrative acoustic'},
    'VM0369': {'cat':'B','tag':'중저음 매력','w':[40, 40, 20],'prompt':'deep literary lyrical male ballad, devastating power-ballad soprano tearing through lush, funky freewheeling male'},
    'VM0370': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'pop-future bass fairy, AlunaGeorge vocal pixie, global trendy tenor with cinematic pop polish, warm earthy'},
    'VM0371': {'cat':'B','tag':'중저음 매력','w':[50, 40, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, devastating power-ballad soprano tearing through lush, pioneering male vocalists'},
    'VM0372': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'resonant deep baritone with dramatic anthemic folk pop vocal projection, underground gritty male, husky passionate male vocalists'},
    'VM0373': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'emerging female vocalists commanding heavy 808 beats with bold presence, deep husky soulful, historic west-coast crew'},
    'VM0374': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'deep soul-laden mezzo-alto with mature tone, sweeping dramatic tenor with lush symphonic, unique bright indie'},
    'VM0375': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'fairy female vocal with perfect breath control and brilliant melisma, cute warm soprano with, global Billboard-hitting female'},
    'VM0376': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'young prodigy narrating deep life stories with mature emotional arc, husky soulful female, lethal off-beat female'},
    'VM0377': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'all-range female spanning deep bass, versatile clear tenor with explosive high notes, velvety smooth perfect'},
    'VM0378': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, raw angular mezzo, warm versatile'},
    'VM0379': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'easygoing sunny tenor with playful organic folk pop vocal charm, deep classic husky female vocal with, bright cheerful male'},
    'VM0380': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, cinematic dubstep orchestral vocal, movie-score, pristine clean high-note'},
    'VM0381': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'beast-like husky male vocal with emotional sorrow and raw power, serene quiet soprano with gentle indie folk healing, revolutionary UK grime male'},
    'VM0382': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'sweeping dramatic tenor with lush symphonic phrasing and soaring crescendos, raw aching male piano vocal that erupts from, gentle wistful male'},
    'VM0383': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'gritty warm male keyboard-soul vocal with bluesy rasp and heartfelt punch, massive cinematic soprano dominating choir and symphony, rapid-fire versatile female'},
    'VM0384': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'heavyweight drum-and-bass male vocalists with signature UK grime flow mastery, feathery high tenor with breezy soft, elegant 60s female'},
    'VM0385': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'mournful mid-bass commanding orchestral-scale grand ballad narratives, edgy youthful alto with emotional, calm low mid-range'},
    'VM0386': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'dignified low-tone who sings like reciting poetry with gravitas, deeply emotive soulful, polished velvety'},
    'VM0387': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'heavyweight hardcore female vocalists with solid, crystalline narrative soprano with warm, French electroclash legend'},
    'VM0388': {'cat':'B','tag':'허스키 감성','w':[70, 20, 10],'prompt':'explosive raspy with gut-wrenching sorrow and raw emotional power, ethereal Nordic, whisper-soft literary female vocalists'},
    'VM0389': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'emotive building male vocal from restrained verse, deep classic husky female vocal with, celestial melodic bass'},
    'VM0390': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'minimalist acoustic with deep resonance on simple folk melodies, distinctive nasal indie tenor with quirky charm, natural conversational mid-range'},
    'VM0391': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'dancer-trained graceful with clear deep lyrical vocal delivery, husky heartbreak-filled female OST, versatile raw male vocalists'},
    'VM0392': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'trembling soulful male vocal with aching falsetto and vulnerable emotional depth, bright pure tenor with wholesome nature-inspired, polished Atlanta trap'},
    'VM0393': {'cat':'B','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'hard-hitting slide- male vocalists with powerful 808 bass-riding technique, husky soulful female, pristine smooth falsetto'},
    'VM0394': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, bel-canto male , angelic fragile yet'},
    'VM0395': {'cat':'B','tag':'압도적 고음','w':[60, 30, 10],'prompt':'distinctive nasal indie tenor with quirky charm and acoustic pop character, airy ethereal falsetto with raw sensual gospel, charismatic bold female'},
    'VM0396': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, genius male vocal combining demonic growling, quirky playful'},
    'VM0397': {'cat':'B','tag':'허스키 감성','w':[50, 30, 20],'prompt':'devastating power-ballad soprano tearing through lush string arrangements emotionally, deep resonant who tenderly soothed, hard-hitting gangster male'},
    'VM0398': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, bright energetic radiating vitality with, minimal clear soprano'},
    'VM0399': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'first foreign champion female vocalist who mastered bending-note technique, layered multitrack choral vocal creating vast, France greatest-selling female'},
    'VM0400': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'smooth classic baritone male crooner jazz pop vocal, powerful rich baritone-tenor with sweeping orchestral, sky-high angelic male'},
    'VM0401': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'Belgian festival vocal performance, hype crowd, smooth modern country soprano with dreamy, passionate revolutionary male'},
    'VM0402': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'husky powerful soprano with overwhelming melismatic soul technique, husky heartbreak-filled female OST, angelic fragile yet'},
    'VM0403': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'brass-backed powerhouse with commanding stage energy, The Voice, perfect female, ice-cold sad rebellious'},
    'VM0404': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'dance anthem powerhouse, Tiesto collaboration, high-energy pop-EDM vocal, clear smooth pure falsetto, world-class speed-rap female'},
    'VM0405': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'delicate lyrical tenor with poetic graceful, street-style diva soprano with massive volume and, nasal high-pitched male'},
    'VM0406': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'charming deep baritone male vocal, the king of rock and roll, romantic emotional tenor with soaring rock-ballad, sophisticated silky'},
    'VM0407': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, raw powerful black-soul-based female belting vocal, hit-songwriter female vocalists'},
    'VM0408': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'powerful gritty vocal on hardcore bass, chest-voice distortion ballad power, ethereal Nordic soprano with nature-inspired, Grammy-winning EDM topline'},
    'VM0409': {'cat':'B','tag':'압도적 고음','w':[60, 30, 10],'prompt':'dramatic operatic male vocal with soaring sorrowful high notes, transparent dewdrop-clear soprano with pristine folk, Australian-born female vocalists'},
    'VM0410': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'R&B-infused female vocalists riding 808 glide bass with smooth vocal elegance, genius singer-songwriter male, chart-dominating male vocalists-singer'},
    'VM0411': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'slow heavyweight UK underground male vocalists with iconic deep bass flow delivery, 5-octave female vocal with dolphin whistle register, clear pristine soprano'},
    'VM0412': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, devastating power-ballad soprano tearing through lush, smooth romantic male vocalists'},
    'VM0413': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, devastating power-ballad soprano, highway queen female'},
    'VM0414': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'rhythmic all-rounder with powerful diction and stage-breaking energy, pansori-certified female crossover singing, refreshing clear soprano'},
    'VM0415': {'cat':'B','tag':'허스키 감성','w':[70, 20, 10],'prompt':'legendary nasal sorrowful uniquely toned female vocal soaking the soul, warm folk acoustic, flashy showman male'},
    'VM0416': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, emotive building male vocal from restrained verse, dreamy sophisticated falsetto'},
    'VM0417': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'heavyweight commanding male vocalists with flawless flow and deep groove mastery, soaring popera soprano, soft gentle tenor'},
    'VM0418': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'pansori-master young melting fierce traditional soul into acoustic folk, honest unpretentious warm, natural nasal-toned mezzo'},
    'VM0419': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'soaring rock soprano with legendary high-range stadium power, husky gravelly male vocalists delivering authentic Atlanta, Nordic crystal-clear'},
    'VM0420': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'refined effortless alto with minimal urban folk, bright pure tenor with wholesome nature-inspired, aggressive hard-hitting male'},
    'VM0421': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'bold thick-toned female vocalists anchoring songs, heavyweight hardcore female vocalists with solid, refined groovy male'},
    'VM0422': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'musical-theater trained female power- with soaring high-note stage presence, emotional vocal trance, heart-purifying chord sequences, polished velvety'},
    'VM0423': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'barefoot diva, deeply appealing drawn from the depths of the heart, cinematic , ethereal theatrical falsetto'},
    'VM0424': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'soaring male rock ballad vocal with polished, earnest warm tenor with pure heartfelt delivery, rapid-fire technical male'},
    'VM0425': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'barefoot diva, deeply appealing drawn from, warm classic tenor with smooth legato phrasing, serene quiet soprano with'},
    'VM0426': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'mournful mid-bass commanding orchestral-scale grand ballad narratives, legendary nasal-melody female , feathery soft tender'},
    'VM0427': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'thunderous deep-cave male vocalists who exploded Brooklyn , original all-rounder , flawless technique male'},
    'VM0428': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'rugged male vocalists blending gritty tone with gangster, inventive jazzy soprano with playful harmonic twists, modern female vocalists'},
    'VM0429': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, saddest tone in jazz, hardcore'},
    'VM0430': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'heavyweight drum-and-bass male vocalists with signature UK grime flow mastery, doll-faced hiding explosive, lush romantic male'},
    'VM0431': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'heavyweight hardcore female vocalists with solid powerful projection and grit, arrogant cynical distinctive male britpop vocal, soft breathy warm'},
    'VM0432': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'elegant refined tenor with classically elevated harmonic vocal phrasing, hit-songwriter female vocalists with raw soulful trap, transparent dewdrop-clear soprano'},
    'VM0433': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'suave romantic baritone with elegant continental, powerful open-throated male singer belting folk, polished velvety'},
    'VM0434': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'most sophisticated calm sensual mid-low female vocal with luxury tone, gentle acoustic to soaring, pioneering'},
    'VM0435': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, modern layered R&B alto with dense, genius sensual male'},
    'VM0436': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, emotional vocal trance, 90s G-Funk revivalist'},
    'VM0437': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'refreshing powerful female country-pop, gentle soothing baritone with the warmest, Brazilian folk-house hybrid'},
    'VM0438': {'cat':'B','tag':'중저음 매력','w':[40, 40, 20],'prompt':'mournful mid-bass commanding, raw angular mezzo with unconventional phrasing, Latin rock- crossover female'},
    'VM0439': {'cat':'B','tag':'중저음 매력','w':[50, 30, 20],'prompt':'percussion-performing with deeply sorrowful han-infused vocal power, passionate tender tenor with soulful Latin, aggressive hard-hitting male'},
    'VM0440': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'sorrowful French chanson female vocal pouring raw life pain like a violin, layered multitrack choral, commanding male vocalists'},
    'VM0441': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'sticky deep husky at the pinnacle, delicate symphonic metal, raw gravelly male vocalists'},
    'VM0442': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'dreamy atmospheric electronic vocal, deep, precise rhythmic Swedish diva, Clean Bandit, ethereal crystalline'},
    'VM0443': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'lush romantic male vocal blending classical piano grandeur with pop yearning, street-style diva soprano, Elvis-inspired charismatic male'},
    'VM0444': {'cat':'B','tag':'중저음 매력','w':[50, 30, 20],'prompt':'warm charismatic baritone with calypso-tinged folk orchestral humanity, hit-songwriter female vocalists with raw soulful trap, globally acclaimed UK'},
    'VM0445': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'bold theatrical baritone with brassy big band showman vocal energy, gentle acoustic to soaring, airy ethereal falsetto'},
    'VM0446': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'heavyweight drum-and-bass male vocalists with signature UK grime flow mastery, distinctive nasal indie tenor with quirky charm, crystalline narrative'},
    'VM0447': {'cat':'B','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'clear bright female pop vocal with ultra-high technique continuing the Mariah Carey legacy, flawless classic female, intense dramatic'},
    'VM0448': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'minimalist acoustic with deep resonance on simple folk melodies, master-architect male , French electroclash legend'},
    'VM0449': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'sorrowful French chanson female vocal pouring raw life pain like a violin, soft breathy warm, razor-sharp UK grime'},
    'VM0450': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'mournful mid-bass commanding orchestral-scale grand ballad narratives, operatic tenor completing orchestral-scale, sweet lyrical soprano'},
    'VM0451': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'commanding heavy baritone with powerful sensual deep soul growl, broken sobbing male vocal pouring desperate modern, pristine clean high-note'},
    'VM0452': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'husky deep mid-low adding mature depth, overwhelming falsetto high male , husky heartbreak-filled'},
    'VM0453': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'husky powerful soprano with overwhelming, gritty warm male keyboard-soul vocal with bluesy, gentle wistful male'},
    'VM0454': {'cat':'B','tag':'중저음 매력','w':[40, 40, 20],'prompt':'bold deep contralto with distinctive vibrato, raw gravelly male vocalists with ferocious barking energy, 90s G-Funk revivalist'},
    'VM0455': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'first lady of jazz, perfect pitch rhythm and freestyle scat female vocal, honest unpretentious warm, rapid-fire southern male'},
    'VM0456': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'blended operatic tenor ensemble with lush, soft breathy warm female pop vocal, Philadelphia dark-cloud trap'},
    'VM0457': {'cat':'B','tag':'압도적 고음','w':[60, 20, 20],'prompt':'Australian twin melody master, powerful vocal-driven progressive house, unique delicate female, inventive creative female vocalists'},
    'VM0458': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'5-octave female vocal with dolphin whistle register and R&B melisma, raw convulsive gravelly male vocal wringing every, serene quiet soprano with'},
    'VM0459': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'mellow melodic male vocalists with addictive hooks and relaxed hybrid ballad delivery, flawless technique male mastering, mysterious Eastern pentatonic'},
    'VM0460': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, warm robust tenor with grand sweeping romantic, ethereal breathtaking soprano'},
    'VM0461': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'bel-canto creating cinematic time-slip narratives with unique timbre, refreshing bright rock soprano with crisp attack, Polaris-winning Canadian female'},
    'VM0462': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, piercing powerful, soulful mezzo-soprano with'},
    'VM0463': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'monster-vocal soprano with devastating power and next-generation explosive technique, authentic R&B alto with golden-era groove and, pristine classical'},
    'VM0464': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, lush romantic male, rapid-fire versatile female'},
    'VM0465': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'clear steady male pop- hiding deep lyricism behind flashy performance, refined velvety tenor with elegant soaring, emotionally charged soprano'},
    'VM0466': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'emotionally explosive belting soprano with dramatic, genre-bending refined alto with jazz-soul sophistication, crystalline pure-toned male'},
    'VM0467': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'commanding heavy baritone with powerful sensual, distinctive nasal indie tenor with quirky charm, bouncy yet heartfelt'},
    'VM0468': {'cat':'B','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'operatic tenor completing orchestral-scale power with massive volume, husky heartbreak-filled, bright pure tenor'},
    'VM0469': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'ultimate female high-note queen with blade-sharp, husky intelligent female R&B vocal with, angelic fragile yet'},
    'VM0470': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'sorrowful bending-note master , youthful clear tenor with earnest modern, lush romantic male'},
    'VM0471': {'cat':'B','tag':'압도적 고음','w':[50, 30, 20],'prompt':'emotionally explosive belting soprano with dramatic orchestral climax power, airy ethereal male falsetto vocal building, trap EDM'},
    'VM0472': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'all-range female spanning deep bass, operatic tenor completing orchestral-scale, emotional vocal trance'},
    'VM0473': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'smooth classic baritone male crooner jazz pop vocal, queen of soul, gospel-based explosive powerful female vocal, folk-ballad optimized male'},
    'VM0474': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, clear earnest tenor with sweeping poetic, rugged male vocalists blending'},
    'VM0475': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, powerful venue-shaking female, youthful crystalline soprano'},
    'VM0476': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'dignified low-tone who sings like reciting poetry with gravitas, modern breathy alternative R&B vocal with, sharp organic indie-trap'},
    'VM0477': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'bouncy yet heartfelt country female vocal with, silvery celestial country harmony soprano with, smooth romantic male vocalists'},
    'VM0478': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavyweight drum-and-bass male vocalists with signature UK grime flow mastery, prodigy mastering saxophone to orchestra, wise philosophical male'},
    'VM0479': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep literary lyrical male ballad vocal with quiet resonance, crystalline soaring high male tenor with, hit-songwriter female vocalists'},
    'VM0480': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'most sophisticated calm sensual mid-low female vocal with luxury tone, refined groovy male , ethereal Nordic'},
    'VM0481': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'sophisticated mid-range vocal with refined phrasing and understated elegance, crystal-clear healing female, Chicago hardcore'},
    'VM0482': {'cat':'B','tag':'압도적 고음','w':[70, 20, 10],'prompt':'refreshing powerful female country-pop crossover vocal, soft breathy warm, authoritative male vocalists'},
    'VM0483': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'mellow melodic male vocalists with addictive hooks and relaxed hybrid ballad delivery, pioneering with signature vibrato who, quintessentially optimistic male'},
    'VM0484': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'Miss champion female vocalist with overwhelming pansori-based power shattering han, pansori-infused cinematic with elaborate melodic, ethereal Nordic'},
    'VM0485': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'sticky deep neo-soul male vocal that, gentle acoustic to soaring high notes, clear storytelling , street-style diva soprano'},
    'VM0486': {'cat':'B','tag':'허스키 감성','w':[50, 30, 20],'prompt':'pansori-master young melting fierce traditional soul into acoustic folk, comforting warm baritone with steady timeless, Bronx female'},
    'VM0487': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'doll-faced hiding explosive pansori-scaled cinematic high-note power, smooth silky male R&B piano vocal, Afrobeat- fusion male'},
    'VM0488': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, folk-ballad optimized male, inventive jazzy soprano'},
    'VM0489': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'authentic R&B alto with golden-era groove and smooth soulful vocal runs, pure clean soprano capturing quiet depth, revolutionary UK grime male'},
    'VM0490': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'gayageum-playing female hybrid bridging traditional, pristine clear soprano with ethereal purity suited, Eastern-melodic female vocalists'},
    'VM0491': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'smooth classic baritone male crooner jazz pop vocal, distinctive nasal indie, refined effortless alto'},
    'VM0492': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'sorrowful bending-note master with deeply mournful delivery, sophisticated folk soprano, deep resonant female'},
    'VM0493': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'scratched wounded raspy male grunge vocal, youthful crystalline soprano with emotionally transparent, fierce Miami trap duo'},
    'VM0494': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'commanding alto with passionate 90s ballad grit, rough torn raspy, Afrobeat- hybrid female'},
    'VM0495': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'anthem trance, heart-wrenching melodies, powerful emotional vocal trance hooks, explosive raspy male , precise pitch-perfect male pop'},
    'VM0496': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'sorrowful falsetto transitioning to angry melodic screaming male vocal, ice-cold sad rebellious female vocal with, airy ethereal male'},
    'VM0497': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'heavyweight commanding male vocalists with flawless flow and deep groove mastery, distinctive high-pitched male, earnest narrative male'},
    'VM0498': {'cat':'B','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep velvety contralto with smoldering sultry low-register warmth, master-architect radically mixing pansori, inventive creative female vocalists'},
    'VM0499': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, gentle acoustic to soaring high notes, clear storytelling , passionate raspy tenor'},
    'VM0500': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'minimalist acoustic with deep resonance on simple folk melodies, uniquely flavored with signature, quirky playful'},
    'VM0501': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'distinctive high-pitched male vocalists gangsta, dreamy Latin-pop female vocalists weaving ethereal harmonics, explosive raspy female vocalists'},
    'VM0502': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'razor-sharp UK grime male vocalists representing London streets with global authority, prodigious genius female, legendary nasal-melody female'},
    'VM0503': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'dramatic powerful male rock vocal with, legendary 80s female vocalists who spearheaded , addictive melodic'},
    'VM0504': {'cat':'C','tag':'일렉트로닉','w':[50, 30, 20],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, legendary Three 6 Mafia female vocalists ruling southern, idol-crossover female'},
    'VM0505': {'cat':'C','tag':'천상의 목소리','w':[40, 40, 20],'prompt':'nervous yet beautiful dreamy falsetto male, emerging female vocalists commanding heavy 808, 90s G-Funk revivalist'},
    'VM0506': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'cinematic riding large string sections with emotional stability and depth, dark European female, pansori-master young female'},
    'VM0507': {'cat':'C','tag':'크리스탈 톤','w':[60, 20, 20],'prompt':'genius sensual male vocal switching between falsetto and chest voice, 5-octave female vocal, hardcore'},
    'VM0508': {'cat':'C','tag':'펑키 그루브','w':[50, 30, 20],'prompt':'versatile male vocal from soft falsetto to rock screaming, powerhouse big-voiced who, hypnotic baby-voice male'},
    'VM0509': {'cat':'C','tag':'허스키 감성','w':[60, 20, 20],'prompt':'genius rhythmic soulful male vocal with brilliant melisma technique, most destructive female, deep-voiced southern female vocalists'},
    'VM0510': {'cat':'C','tag':'굵은 바리톤','w':[70, 20, 10],'prompt':'heavyweight deep-bass male vocalists dominating grandiose trap beats with boss authority, explosive power from, pharmacist-turned female'},
    'VM0511': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'nasal high-pitched male vocalists powerful pansori-toned who made the, pioneering'},
    'VM0512': {'cat':'C','tag':'허스키 매력','w':[60, 30, 10],'prompt':'rough textured male vocal with gritty emotional rock-ballad appeal and edge, intense 90s New York hardcore female vocalists, pansori-based male'},
    'VM0513': {'cat':'C','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'polished male vocalists with clean jazz-synthpop beats and sophisticated hybrid delivery, emerging female, revolutionary male vocalists who'},
    'VM0514': {'cat':'C','tag':'그루브 보컬','w':[50, 30, 20],'prompt':'atmospheric Canadian male vocalists crafting dreamy synth-trap soundscapes with soft delivery, explosive energy rough husky female rock and roll, authoritative male vocalists'},
    'VM0515': {'cat':'C','tag':'깊은 울림','w':[60, 30, 10],'prompt':'deep-voiced male vocalist-producer who powered Death Row Records golden era sound, gentle acoustic to soaring high notes, clear storytelling , explosive energy rough husky'},
    'VM0516': {'cat':'C','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'pristine smooth male easy-listening vocal with serene velvety mid-range tone, new-wave rock female vocalists who shattered boundaries between, refreshing powerful'},
    'VM0517': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'smooth romantic male vocalists delivering the most mellow G-Funk with laid-back flow, heavyweight hardcore female vocalists with solid, rhythmic groove master male'},
    'VM0518': {'cat':'C','tag':'깊은 울림','w':[50, 40, 10],'prompt':'thunderous deep-cave male vocalists who exploded Brooklyn , Miss champion female vocalist with overwhelming pansori-based, rough torn raspy'},
    'VM0519': {'cat':'C','tag':'깊은 베이스','w':[70, 20, 10],'prompt':'mellow melodic male vocalists with addictive hooks and relaxed hybrid ballad delivery, quiet warm soothing, arrogant cynical distinctive'},
    'VM0520': {'cat':'C','tag':'허스키 매력','w':[70, 20, 10],'prompt':'rough textured male vocal with gritty emotional rock-ballad appeal and edge, legendary 80s female, raw aching male piano'},
    'VM0521': {'cat':'C','tag':'깊은 베이스','w':[40, 40, 20],'prompt':'polished swinging male crooner vocal with, Eastern-melodic female vocalists crossing Asian tonality, deep-voiced southern female vocalists'},
    'VM0522': {'cat':'C','tag':'그루브 보컬','w':[60, 20, 20],'prompt':'precision-engineered modern male vocalists with relentless continuous trap flow dominance, highway queen female , raw aching male piano'},
    'VM0523': {'cat':'C','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'authoritative smooth male vocalists with business-mogul swagger and effortless delivery, musical-theater trained female, soulful male vocal with'},
    'VM0524': {'cat':'C','tag':'크리스탈 톤','w':[70, 20, 10],'prompt':'angelic fragile yet devastating falsetto male vocal with soul-shaking emotion, Afrobeat- hybrid female, explosive power from'},
    'VM0525': {'cat':'C','tag':'천상의 목소리','w':[50, 40, 10],'prompt':'angelic fragile yet devastating falsetto male vocal with soul-shaking emotion, elegant 60s layering sophisticated, next-generation hardcore female'},
    'VM0526': {'cat':'C','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'flawless technique male mastering every emotion perfectly, new-wave rock female vocalists, sharp organic indie-trap'},
    'VM0527': {'cat':'C','tag':'깊은 베이스','w':[50, 30, 20],'prompt':'heavyweight deep-bass male vocalists dominating grandiose trap beats with boss authority, radically alternative female vocalists fusing avant-garde, sorrowful bending-note master'},
    'VM0528': {'cat':'C','tag':'허스키 매력','w':[50, 40, 10],'prompt':'bel-canto creating cinematic time-slip narratives with unique timbre, technically versatile female vocalists freely riding R&B, soulful male vocal with'},
    'VM0529': {'cat':'C','tag':'일렉트로닉','w':[50, 40, 10],'prompt':'percussion-performing with deeply sorrowful han-infused vocal power, Chicago hardcore female vocalists with precise, thunderous military-grade male'},
    'VM0530': {'cat':'C','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep literary lyrical male ballad vocal with quiet resonance, fairy female, R&B-infused female vocalists riding'},
    'VM0531': {'cat':'C','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, cute nasally charming female electronic , hard-hitting slide- male'},
    'VM0532': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'technically gifted male vocalists with extraordinary rhyme arrangement on west-coast beats, unique sophisticated neo-soul, saddest tone in jazz'},
    'VM0533': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'explosive operatic metal male vocal like a human air raid siren, world-class speed-rap female vocalists switching effortlessly, new-wave rock female vocalists'},
    'VM0534': {'cat':'C','tag':'허스키 감성','w':[40, 40, 20],'prompt':'silky smooth sensual male, hardcore female vocalists filling beats, husky deep mid-low female'},
    'VM0535': {'cat':'C','tag':'그루브 보컬','w':[60, 30, 10],'prompt':'atmospheric Canadian male vocalists crafting dreamy synth-trap soundscapes with soft delivery, iconic pop female vocal with unique tone that, battle-rap legend female'},
    'VM0536': {'cat':'C','tag':'투명한 음색','w':[70, 20, 10],'prompt':'crystalline pure-toned male tenor revered as the emperor of classic , deep powerful female, raw rough soul-shaking'},
    'VM0537': {'cat':'C','tag':'천상의 목소리','w':[60, 30, 10],'prompt':'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes, all-range female spanning deep bass, wildly innovative southern male'},
    'VM0538': {'cat':'C','tag':'시원한 고음','w':[50, 30, 20],'prompt':'husky powerful male belting vocal with soul-drenched rock ballad intensity, whisper-soft literary female vocalists floating poetically over jazz, warm low-register male'},
    'VM0539': {'cat':'C','tag':'천상의 목소리','w':[70, 20, 10],'prompt':'warm low-register evoking hometown nostalgia with gentle phrasing, textbook , sharp high-tone female'},
    'VM0540': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'powerful UK national male vocalists fusing grime with, positive upbeat rhythmic, Caribbean-fusion male vocalists'},
    'VM0541': {'cat':'C','tag':'크리스탈 톤','w':[50, 30, 20],'prompt':'angelic fragile yet devastating falsetto male vocal with soul-shaking emotion, explosive powerful energetic mastering, Polaris-winning Canadian female'},
    'VM0542': {'cat':'C','tag':'그루브 보컬','w':[60, 30, 10],'prompt':'commanding male vocalists carrying west-coast and with authority, soft breathy warm female pop vocal, pioneering'},
    'VM0543': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'raspy warm male rock vocal with anthemic sing-along ballad grit, London-born healing female, dreamy Latin-pop female'},
    'VM0544': {'cat':'C','tag':'크리스탈 톤','w':[60, 20, 20],'prompt':'bright energetic radiating vitality with open airy tenor delivery, highway queen female , perfect vocal technique'},
    'VM0545': {'cat':'C','tag':'감성 폭발','w':[50, 30, 20],'prompt':'underground gritty male vocalists with raw sensibility and street authenticity, deep resonant who tenderly soothed, theatrical sweeping male'},
    'VM0546': {'cat':'C','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'clear smooth pure falsetto male R&B vocal, most destructive female rock vocal in, aggressive hard-hitting male'},
    'VM0547': {'cat':'C','tag':'깊은 베이스','w':[60, 20, 20],'prompt':'thunderous deep-cave male vocalists who exploded Brooklyn , idol-crossover female , timeless elegant male'},
    'VM0548': {'cat':'C','tag':'허스키 매력','w':[50, 40, 10],'prompt':'raw rough soul-shaking male rock vocal with emotional honesty, punk-rage female vocalists pioneering trap-metal by fusing rock, pioneering fierce female'},
    'VM0549': {'cat':'C','tag':'투명한 음색','w':[50, 40, 10],'prompt':'warm intimate male folk pop vocal with gentle rasp, Miss champion female vocalist with overwhelming pansori-based, arrogant cynical distinctive'},
    'VM0550': {'cat':'C','tag':'시원한 고음','w':[50, 40, 10],'prompt':'soaring male rock ballad vocal with polished tenor and guitar-driven passion, pioneering fierce female rock alto with, genius producing female'},
    'VM0551': {'cat':'C','tag':'허스키 감성','w':[40, 40, 20],'prompt':'sophisticated mid-century bridging modern, husky deep mid-low adding mature, innovative female vocalist-producer'},
    'VM0552': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'raspy warm male rock vocal with anthemic sing-along ballad grit, explosive hardcore female vocalists who shredded 90s Death, technically sharp male'},
    'VM0553': {'cat':'C','tag':'깊은 베이스','w':[60, 20, 20],'prompt':'minimalist acoustic with deep resonance on simple folk melodies, pure refreshing female , energetic flashy male vocalists'},
    'VM0554': {'cat':'C','tag':'투명한 음색','w':[50, 40, 10],'prompt':'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes, textbook with decades of live, dreamy atmospheric female'},
    'VM0555': {'cat':'C','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep literary lyrical male ballad vocal with quiet resonance, legendary Three 6 Mafia female vocalists ruling southern, gentle acoustic to soaring'},
    'VM0556': {'cat':'C','tag':'허스키 감성','w':[70, 20, 10],'prompt':'folk-ballad optimized with sweet sentimental melodic craftsmanship, cute bright female, mournful mid-bass male'},
    'VM0557': {'cat':'C','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, battle-rap legend female, wailing blues-rock male'},
    'VM0558': {'cat':'C','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'genius sensual male vocal switching between falsetto and chest voice with incredible range, revolutionary female vocalists fusing third-world percussion, unique sophisticated neo-soul'},
    'VM0559': {'cat':'C','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'atmospheric Canadian male vocalists crafting dreamy synth-trap soundscapes with soft delivery, legendary nasal sorrowful uniquely, feathery soft tender'},
    'VM0560': {'cat':'C','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'thunderous deep-cave male vocalists who exploded Brooklyn , global Billboard-hitting female vocalists with Thai-international swagger, pristine clean high-note'},
    'VM0561': {'cat':'C','tag':'깊은 베이스','w':[50, 30, 20],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, pansori-certified female crossover singing, nasal high-pitched male'},
    'VM0562': {'cat':'C','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'dignified low-tone who sings like reciting poetry with gravitas, Bronx female vocalists who conquered global, deep classic husky female'},
    'VM0563': {'cat':'C','tag':'굵은 바리톤','w':[70, 20, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, first foreign , plaintive gentle male'},
    'VM0564': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'bel-canto creating cinematic time-slip narratives with unique timbre, pioneering female vocalists who achieved the first, legendary storytelling male'},
    'VM0565': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'raw aching male piano vocal that erupts from whisper to desperate wail, viral hook-machine female vocalists crafting addictive trap, Afrobeat- hybrid female'},
    'VM0566': {'cat':'C','tag':'감성 폭발','w':[40, 40, 20],'prompt':'uniquely flavored with signature, folk-rock gentle comforting the nation, pansori-certified female crossover'},
    'VM0567': {'cat':'C','tag':'폭발 에너지','w':[70, 20, 10],'prompt':'bel-canto creating cinematic time-slip narratives with unique timbre, crystal clear yet, raw powerful female'},
    'VM0568': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'raw rough soul-shaking male rock vocal with emotional honesty, technically versatile female vocalists freely riding R&B, wise philosophical male'},
    'VM0569': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'tireless iron-throated high male rock vocal with unique groove, understated monotone female vocalists dominating, Miami hardcore female vocalists'},
    'VM0570': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'rhythmic male pop vocal freely switching between falsetto and chest voice, deep-voiced southern female vocalists with heavy 808 impact, distinctive husky low-tone'},
    'VM0571': {'cat':'C','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'refined groovy with idol-trained polish and solid vocal technique, pioneering female vocalists, razor-precise female'},
    'VM0572': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'explosive raspy with gut-wrenching sorrow and raw emotional power, Dutch female vocalists who captivated all of, sophisticated tension-chord female'},
    'VM0573': {'cat':'C','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'heavyweight drum-and-bass male vocalists with signature UK grime flow mastery, fierce Miami trap duo, arrogant cynical distinctive'},
    'VM0574': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'pansori-infused cinematic with elaborate melodic architecture and grand projection, 2NE1 hardcore female vocalists Billboard, autotune-wielding male vocalists'},
    'VM0575': {'cat':'C','tag':'중저음 매력','w':[60, 30, 10],'prompt':'traditional bending-note technician with earthy fermented-bean voice, most sophisticated calm sensual mid-low female, sorrowful bending-note master'},
    'VM0576': {'cat':'C','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'trendy stylish male vocalists layering fashion-forward aesthetics over New York , androgynous cold urban, clear smooth'},
    'VM0577': {'cat':'C','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'deep literary lyrical male ballad vocal with quiet resonance, crystal clear yet, versatile male vocalist seamlessly'},
    'VM0578': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'multi-talented male vocalist-producer with refined west-coast lyricism and groove mastery, perfect powerful female R&B pop vocal with flawless, revolutionary male vocalists who'},
    'VM0579': {'cat':'C','tag':'그루브 보컬','w':[60, 20, 20],'prompt':'trendy stylish male vocalists layering fashion-forward aesthetics over New York , deep classic husky female, psychedelic male vocalists'},
    'VM0580': {'cat':'C','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'transparent fragile male vocal with crystalline sad tone and quiet intensity, lethal off-beat female vocalists-singer delivering devastating, perfect powerful female R&B'},
    'VM0581': {'cat':'C','tag':'압도적 고음','w':[60, 30, 10],'prompt':'classically trained powerful high-note who launched power- era, idol-trained groovy female new-wave , explosive powerful energetic'},
    'VM0582': {'cat':'C','tag':'감성 폭발','w':[50, 30, 20],'prompt':'rugged bending-note cutting through grand horn and string ensembles, quiet warm soothing female jazz folk vocal, theatrical sweeping male'},
    'VM0583': {'cat':'C','tag':'그루브 보컬','w':[40, 40, 20],'prompt':'punk-rock crossover male vocalists who perfectly blended, hit-songwriter female vocalists with raw soulful trap, fierce Miami trap duo'},
    'VM0584': {'cat':'C','tag':'크리스탈 톤','w':[70, 20, 10],'prompt':'quintessentially optimistic with earthy rustic warmth and joy, 90s G-Funk revivalist, raw gravelly male vocalists'},
    'VM0585': {'cat':'C','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'distinctive high-pitched male vocalists gangsta rap with piercing tone, technically versatile female vocalists freely riding R&B, slow heavyweight UK underground'},
    'VM0586': {'cat':'C','tag':'감성 폭발','w':[70, 20, 10],'prompt':'folk-ballad optimized with sweet sentimental melodic craftsmanship, powerful pansori-toned female , solid high male'},
    'VM0587': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'hypnotic baby-voice male vocalists commanding rave-trap with addictive minimalist flow, crystal clear yet, velvety smooth perfect'},
    'VM0588': {'cat':'C','tag':'투명한 음색','w':[50, 30, 20],'prompt':'folk-ballad optimized with sweet sentimental melodic craftsmanship, quiet warm soothing female jazz folk vocal, smooth silky male'},
    'VM0589': {'cat':'C','tag':'허스키 감성','w':[50, 40, 10],'prompt':'raw rough soul-shaking male rock vocal with emotional honesty, most sophisticated calm sensual mid-low female, explosive raspy female vocalists'},
    'VM0590': {'cat':'C','tag':'감성 폭발','w':[50, 30, 20],'prompt':'raw convulsive gravelly male vocal wringing every note with blues agony, perfect powerful female R&B pop vocal with flawless, silky sliding male'},
    'VM0591': {'cat':'C','tag':'굵은 바리톤','w':[40, 40, 20],'prompt':'rich earthy male baritone comforting working-class, girl-group trained hiding solid traditional, most sophisticated calm'},
    'VM0592': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'desperately emotional high-pitched male that makes everyone cry, technically versatile female, underground legend male'},
    'VM0593': {'cat':'C','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'Afrobeat- fusion male vocalists Afroswing genre with infectious energy, highway queen female , gritty groovy modern'},
    'VM0594': {'cat':'C','tag':'천상의 목소리','w':[50, 30, 20],'prompt':'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes, sorrowful French chanson female vocal pouring raw life, passionate revolutionary male'},
    'VM0595': {'cat':'C','tag':'허스키 감성','w':[50, 40, 10],'prompt':'traditional bending-note technician with earthy fermented-bean voice, most sophisticated calm sensual mid-low female, global Billboard-hitting female'},
    'VM0596': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'versatile raw male vocalists spanning distorted lo-fi beats to tender acoustic rap, heavyweight hardcore female vocalists with solid, Dutch female vocalists'},
    'VM0597': {'cat':'C','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'pinnacle of soul male vocal with perfect high-tone technique, cute nasally charming female electronic , husky soulful female'},
    'VM0598': {'cat':'C','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'legendary pure high male tenor with effortless, legendary 80s female vocalists who spearheaded , velvety smooth perfect'},
    'VM0599': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'flawless technique male mastering every emotion perfectly, inventive creative female vocalists, wise philosophical male'},
    'VM0600': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'sandpaper-rough charming male vocal with loose swaggering rock-ballad phrasing, viral hook-machine female vocalists crafting addictive trap, solid high male'},
    'VM0601': {'cat':'C','tag':'폭발 에너지','w':[40, 40, 20],'prompt':'gritty charismatic male stadium-rock vocal with, dark European female vocalists commanding heavy trap, pansori-certified female crossover'},
    'VM0602': {'cat':'C','tag':'압도적 고음','w':[70, 20, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, ice-cold sad rebellious, pinnacle of'},
    'VM0603': {'cat':'C','tag':'깊은 베이스','w':[60, 30, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, husky intelligent female R&B vocal with, deep baritone-grade female'},
    'VM0604': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'sharp nervous ultra-high screaming male rock vocal with wide range, underground technical female vocalists with the, globally acclaimed UK'},
    'VM0605': {'cat':'C','tag':'감성 보컬','w':[70, 20, 10],'prompt':'deep-voiced male vocalist-producer who powered Death Row Records golden era sound, punk rock godmother, the king of'},
    'VM0606': {'cat':'C','tag':'펑키 그루브','w':[60, 30, 10],'prompt':'transparent fragile male vocal with crystalline sad tone and quiet intensity, UK club hyperpop female vocalists crossing electronic beats, pioneering UK grime-garage'},
    'VM0607': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'ice-cold monotone male vocalists embodying modern dark trap with deadpan delivery, refreshing powerful female country-pop, trailblazing New York female'},
    'VM0608': {'cat':'C','tag':'압도적 고음','w':[50, 30, 20],'prompt':'genius lyricist UK male vocalists delivering profound narratives over piano-driven beats, 5-octave female vocal with dolphin whistle, raw powerful female'},
    'VM0609': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'genius lyricist UK male vocalists delivering profound narratives over piano-driven beats, stable powerhouse with the most, genius producing female'},
    'VM0610': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'broken sobbing male vocal pouring desperate modern heartbreak with Scottish rasp, anime-aesthetic female vocalists, dreamy atmospheric female'},
    'VM0611': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'pioneering male vocalists who defined modern rhyme schemes with meticulous cadence, bouncy yet heartfelt country female vocal with, accessible narrative male'},
    'VM0612': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'overwhelming falsetto high male vocal dominating karaoke, cute nasally charming, commanding male vocalists'},
    'VM0613': {'cat':'C','tag':'그루브 보컬','w':[60, 20, 20],'prompt':'wordplay-brilliant duo male vocalists with pop-friendly beat chemistry, deep classic husky, underground technical female'},
    'VM0614': {'cat':'C','tag':'크리스탈 톤','w':[60, 20, 20],'prompt':'clear smooth pure falsetto male R&B vocal, whisper-soft literary female vocalists, androgynous cold urban'},
    'VM0615': {'cat':'C','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'crystalline soaring high male tenor with smooth Chicago soft-rock shimmer, commanding female vocalists who elevated with social, textbook traditional female'},
    'VM0616': {'cat':'C','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'hard-hitting slide- male vocalists with powerful 808 bass-riding technique, Griselda Records queen female, powerful rough soulful'},
    'VM0617': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'versatile male vocalist seamlessly fusing R&B over laid-back west-coast beats, musical-theater trained female power- with, explosive raspy female vocalists'},
    'VM0618': {'cat':'C','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'hook-driven addictive dominating with catchy refrains and deep emotion, viral hook-machine female vocalists crafting addictive trap, sorrowful falsetto transitioning'},
    'VM0619': {'cat':'C','tag':'거친 소울','w':[60, 20, 20],'prompt':'pop-rock acoustic male vocalists who conquered Billboard with accessible crossover sound, pop-ballad crossover female, fairy female'},
    'VM0620': {'cat':'C','tag':'압도적 고음','w':[40, 40, 20],'prompt':'emo-rock infused male vocalists fusing emotional intensity, clear bright female pop vocal with ultra-high technique, solid expressive female'},
    'VM0621': {'cat':'C','tag':'크리스탈 톤','w':[60, 30, 10],'prompt':'quintessentially optimistic with earthy rustic warmth and joy, dancehall-reggae female vocalists fusing Caribbean rhythms with, high-pitched screaming male'},
    'VM0622': {'cat':'C','tag':'감성 폭발','w':[40, 40, 20],'prompt':'raw aching male piano vocal that erupts from, clear bright female pop vocal, aggressive hard-hitting male'},
    'VM0623': {'cat':'C','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, pioneering UK grime-garage, legendary pure high'},
    'VM0624': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'commanding male vocalists carrying west-coast and global, mysterious powerful gothic female rock vocal piercing, rapid-fire versatile female'},
    'VM0625': {'cat':'C','tag':'허스키 감성','w':[70, 20, 10],'prompt':'rugged male vocalists blending gritty tone with gangster balladry and west-coast soul, refined , inventive genius male'},
    'VM0626': {'cat':'C','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes, Canadian dark-aesthetic female, silky laid-back male'},
    'VM0627': {'cat':'C','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'slow heavyweight UK underground male vocalists with iconic, revolutionary female vocalists fusing third-world percussion, rebellious melancholic raw retro'},
    'VM0628': {'cat':'C','tag':'허스키 감성','w':[40, 40, 20],'prompt':'raw rough soul-shaking male rock vocal, cute bright dominating highway-groove, charming deep baritone'},
    'VM0629': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'rapid-fire technical male vocalists balancing speed with accessible pop-ballad sensibility, sad sharp Irish traditional female vocal, wise philosophical male'},
    'VM0630': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'foundational male vocalists who architected modern trap culture and sound, barefoot diva, deeply appealing drawn from, wailing blues-rock male'},
    'VM0631': {'cat':'C','tag':'리듬 보컬','w':[60, 20, 20],'prompt':'relentless rapid-fire male vocalists with massive projection and dominance, rustic mid-low bending-note, pinnacle of'},
    'VM0632': {'cat':'C','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'electronic-trap crossover male vocalists blending synths with global hybrid , deep resonant who tenderly soothed, virtuoso male vocalists'},
    'VM0633': {'cat':'C','tag':'파워 보컬','w':[60, 20, 20],'prompt':'genius sensual male vocal switching between falsetto and chest voice, crystal clear yet, solid expressive female'},
    'VM0634': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'rhythmic groove master who electrified all generations with one hit, deep heavy contralto female vocal singing the, genius singer-songwriter male'},
    'VM0635': {'cat':'C','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'pinnacle of soul male vocal with perfect high-tone technique, glamorous west-coast female, The Voice, perfect'},
    'VM0636': {'cat':'C','tag':'일렉트로닉','w':[60, 20, 20],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, witty transatlantic female vocalists, Polaris-winning Canadian female'},
    'VM0637': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'raw convulsive gravelly male vocal wringing every note with blues agony, bold thick-toned , legendary nasal-melody female'},
    'VM0638': {'cat':'C','tag':'그루브 보컬','w':[70, 20, 10],'prompt':'silky sliding male vocalists perfecting melodic trap with fluid effortless delivery, bouncy yet heartfelt, fierce Miami trap duo'},
    'VM0639': {'cat':'C','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'punk-rock crossover male vocalists who perfectly blended hardcore punk with trap, barefoot diva, deeply appealing drawn from, deep resonant female'},
    'VM0640': {'cat':'C','tag':'거친 소울','w':[60, 30, 10],'prompt':'velvety smooth perfect male R&B ballad vocal with full volume, husky deep mid-low adding mature, raw convulsive gravelly'},
    'VM0641': {'cat':'C','tag':'허스키 감성','w':[40, 40, 20],'prompt':'theatrical sweeping male piano ballad vocal, modern female vocalists praised by legends for, overwhelming falsetto high'},
    'VM0642': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'genius lyricist UK male vocalists delivering profound narratives over piano-driven beats, mysterious powerful gothic female rock vocal piercing, UK club hyperpop female'},
    'VM0643': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'arrogant cynical distinctive male britpop vocal that defined an era, genius producing female vocalists who shatters idol, Griselda Records queen female'},
    'VM0644': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'multi-talented male vocalist-producer with refined west-coast lyricism and groove mastery, stable powerhouse with the most, technically sharp male'},
    'VM0645': {'cat':'C','tag':'허스키 매력','w':[40, 40, 20],'prompt':'raw convulsive gravelly male vocal wringing every, revolutionary female vocalists fusing third-world percussion, earthy rustic male'},
    'VM0646': {'cat':'C','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'high-pitched screaming male rock vocal, textbook hard rock tenor, microtonal Arab-maqam female, dramatic powerful male'},
    'VM0647': {'cat':'C','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'operatic tenor completing orchestral-scale power with massive volume, dark European female vocalists commanding heavy trap, fairy female'},
    'VM0648': {'cat':'C','tag':'펑키 그루브','w':[40, 40, 20],'prompt':'perfect vocal technique with appealing sweet, explosive hardcore female vocalists who shredded 90s Death, rapid-fire southern male'},
    'VM0649': {'cat':'C','tag':'천상의 목소리','w':[60, 20, 20],'prompt':'warm folk acoustic with heartfelt lonely storytelling, rich-volume female cinematic, technically versatile female'},
    'VM0650': {'cat':'C','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'lightning-fast male vocalists layering angelic melodies over rapid-fire delivery uniquely, otherworldly bizarre yet beautiful avant-garde female, feathery soft tender'},
    'VM0651': {'cat':'C','tag':'허스키 감성','w':[60, 20, 20],'prompt':'explosive raspy with gut-wrenching sorrow and raw emotional power, glamorous west-coast female, powerful open-throated male'},
    'VM0652': {'cat':'C','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, pure refreshing providing emotional calm, raspy warm male'},
    'VM0653': {'cat':'C','tag':'맑은 감성','w':[60, 20, 20],'prompt':'silky laid-back male vocalists with signature drawl and effortless west-coast groove, ethereal theatrical falsetto, warm intimate male'},
    'VM0654': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'emo-rock infused male vocalists fusing emotional intensity with rapid hi-hat trap, rustic mid-low bending-note , fairy female'},
    'VM0655': {'cat':'C','tag':'허스키 감성','w':[50, 30, 20],'prompt':'arrogant cynical distinctive male britpop vocal that defined an era, refined who distills deep traditional, classically trained powerful'},
    'VM0656': {'cat':'C','tag':'깊은 베이스','w':[40, 40, 20],'prompt':'mournful mid-bass commanding, first lady of jazz, perfect pitch rhythm and, dreamy Latin-pop female'},
    'VM0657': {'cat':'C','tag':'크리스탈 톤','w':[70, 20, 10],'prompt':'feathery soft tender male vocal with airy gentle folk-pop delivery, world-class speed-rap female, genius male vocal'},
    'VM0658': {'cat':'C','tag':'압도적 고음','w':[50, 30, 20],'prompt':'sacred powerful metal male vocal with commanding volume from small frame, all-range female spanning deep bass, bright cheerful male'},
    'VM0659': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'operatic tenor completing orchestral-scale power with massive volume, stage-dominating with addictive groove, Dutch female vocalists'},
    'VM0660': {'cat':'C','tag':'그루브 보컬','w':[70, 20, 10],'prompt':'addictive punchline male vocalists with playful charisma and infectious trap mastery, rustic mid-low bending-note, legendary harmony female'},
    'VM0661': {'cat':'C','tag':'천상의 목소리','w':[50, 40, 10],'prompt':'quintessentially optimistic with earthy rustic warmth and joy, spicy capsaicin-sharp with traditional, relentless rapid-fire male'},
    'VM0662': {'cat':'C','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'plaintive gentle soothing homesick hearts with simple heartfelt melody, fierce female vocalists from, new-wave rock female vocalists'},
    'VM0663': {'cat':'C','tag':'허스키 감성','w':[60, 20, 20],'prompt':'broken sobbing male vocal pouring desperate modern heartbreak with Scottish rasp, charming mid-low female, dignified low-tone male'},
    'VM0664': {'cat':'C','tag':'리듬 보컬','w':[60, 20, 20],'prompt':'solid high male vocal with retro and modern groove, energetic performer, crystal-clear healing female, stable powerhouse female'},
    'VM0665': {'cat':'C','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'slow heavyweight UK underground male vocalists with iconic deep bass flow delivery, bouncy southern trap female vocalists optimized, quintessentially optimistic male'},
    'VM0666': {'cat':'C','tag':'압도적 고음','w':[70, 20, 10],'prompt':'legendary storytelling male vocalists with unique accent and theatrical narrative flow, deep powerful female, perfect powerful female'},
    'VM0667': {'cat':'C','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'feathery soft tender male vocal with airy gentle folk-pop delivery, trailblazing New York female vocalists who set fashion, Latin Afro-beat female vocalists'},
    'VM0668': {'cat':'C','tag':'펑키 그루브','w':[40, 40, 20],'prompt':'addictive punchline male vocalists with playful charisma, ice-cold sad rebellious female vocal with, dreamy atmospheric female'},
    'VM0669': {'cat':'C','tag':'허스키 매력','w':[60, 30, 10],'prompt':'prodigy mastering saxophone to orchestra with epic narrative depth, cute nasally charming female electronic , raw rough soul-shaking'},
    'VM0670': {'cat':'C','tag':'리듬 보컬','w':[60, 30, 10],'prompt':'energetic flashy male vocalists with wild hybrid flow over hardcore club beats, refined who distills deep traditional, perfect powerful female R&B'},
    'VM0671': {'cat':'C','tag':'감성 폭발','w':[60, 30, 10],'prompt':'folk-ballad optimized with sweet sentimental melodic craftsmanship, explosive power from small frame, timeless clear, new-wave rock female vocalists'},
    'VM0672': {'cat':'C','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'modern male vocalists perfectly reviving 90s golden-era New York aesthetics, 5-octave female vocal, minimalist acoustic male'},
    'VM0673': {'cat':'C','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'legendary storytelling male vocalists with unique accent and theatrical narrative flow, legendary harmony female, highway queen female'},
    'VM0674': {'cat':'C','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'genius sensual male vocal switching between falsetto and chest voice, historic west-coast crew female vocalists with distinctive, explosive operatic metal'},
    'VM0675': {'cat':'C','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, Latin reggaeton- crossover female vocalists connecting Caribbean, crystal clear yet'},
    'VM0676': {'cat':'C','tag':'깊은 베이스','w':[70, 20, 10],'prompt':'slow heavyweight UK underground male vocalists with iconic deep bass flow delivery, Terror Squad pride female, clear steady male'},
    'VM0677': {'cat':'C','tag':'거친 소울','w':[50, 30, 20],'prompt':'genius lyricist UK male vocalists delivering profound narratives over piano-driven beats, ice-cold sad rebellious female vocal with, powerful rough soulful'},
    'VM0678': {'cat':'C','tag':'펑키 그루브','w':[40, 40, 20],'prompt':'funky freewheeling male vocalists with raw unfiltered, husky intelligent female R&B vocal with, most devastating powerful'},
    'VM0679': {'cat':'C','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'immortal deep baritone who elevated music with noble dignity, solid expressive female , soulful jazzy male vocalists'},
    'VM0680': {'cat':'C','tag':'시원한 고음','w':[60, 20, 20],'prompt':'stadium-filling resonant male vocal with powerful message delivery, rebellious melancholic raw retro, emo-rock infused male'},
    'VM0681': {'cat':'C','tag':'거친 소울','w':[50, 30, 20],'prompt':'the king , versatile male vocal covering rock ballad and folk, NYC underground queen female vocalists embodying alternative, folk-rock gentle female'},
    'VM0682': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'energetic flashy male vocalists with wild hybrid flow over hardcore club beats, pansori-master young female , rugged bending-note male'},
    'VM0683': {'cat':'C','tag':'압도적 고음','w':[60, 30, 10],'prompt':'heart-wrenching melodic male vocalists who epitomized emo-rap with devastating melodies, powerful venue-shaking commanding, deep powerful female'},
    'VM0684': {'cat':'C','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'nervous yet beautiful dreamy falsetto male vocal, ethereal and haunting, underground female, Australian-born female vocalists'},
    'VM0685': {'cat':'C','tag':'감성 폭발','w':[40, 40, 20],'prompt':'raw convulsive gravelly male vocal wringing every, explosive next-generation female cinematic , bright cheerful male'},
    'VM0686': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'rapid-fire technical male vocalists balancing speed with accessible pop-ballad sensibility, husky soulful female vocal fusing and, triplet-flow male vocalists'},
    'VM0687': {'cat':'C','tag':'그루브 보컬','w':[60, 20, 20],'prompt':'paradigm-shifting male vocalist-producer and genre greatest sonic innovator ever, folk-rock gentle female , rich-volume female cinematic'},
    'VM0688': {'cat':'C','tag':'깊은 울림','w':[50, 40, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, textbook traditional with the most, crystal clear yet'},
    'VM0689': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'classical crossover harmonizing operatic power with grand orchestral scale, commanding female vocalists who elevated with social, light floating female vocalists'},
    'VM0690': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'theatrical sweeping male piano ballad vocal with dramatic crescendo delivery, France greatest-selling female vocalists with epic, psychedelic male vocalists'},
    'VM0691': {'cat':'C','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'clear steady male pop- hiding deep lyricism behind flashy performance, original all-rounder female vocalists with rapid-fire, glamorous west-coast female'},
    'VM0692': {'cat':'C','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'powerful UK national male vocalists fusing grime with classic soul harmonics brilliantly, barefoot diva, deeply appealing drawn from, prodigious genius female'},
    'VM0693': {'cat':'C','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'tireless iron-throated high male rock vocal, underground female vocalists crossing hardcore rock, sharp nervous ultra-high'},
    'VM0694': {'cat':'C','tag':'투명한 음색','w':[70, 20, 10],'prompt':'warm intimate male folk pop vocal with gentle rasp, raw powerful black-soul-based, hook-driven addictive male'},
    'VM0695': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'velvety smooth perfect male R&B ballad vocal with full volume, rapid-fire versatile female vocalists with unmatched, lightning-fast male vocalists'},
    'VM0696': {'cat':'C','tag':'깊은 베이스','w':[50, 40, 10],'prompt':'mournful mid-bass commanding orchestral-scale grand ballad narratives, unique delicate representing a generation, poetic jazz-harmony female'},
    'VM0697': {'cat':'C','tag':'압도적 고음','w':[60, 20, 20],'prompt':'versatile raw male vocalists spanning distorted lo-fi beats to tender acoustic rap, doll-faced female , emotive building male'},
    'VM0698': {'cat':'C','tag':'그루브 보컬','w':[50, 30, 20],'prompt':'passionate revolutionary male vocalists with soul-stirring delivery and poetic intensity, dancer-trained graceful with clear, silky smooth'},
    'VM0699': {'cat':'C','tag':'허스키 매력','w':[50, 40, 10],'prompt':'overwhelming male R&B lead vocal with rich harmonics, Caribbean-flavored female vocalists who effortlessly rides, rugged male vocalists blending'},
    'VM0700': {'cat':'C','tag':'거친 소울','w':[60, 20, 20],'prompt':'dreamy alternative male vocalists who implanted psychedelic rock sensibility into , saddest tone in jazz, pop-ballad crossover female'},
    'VM0701': {'cat':'C','tag':'펑키 그루브','w':[60, 30, 10],'prompt':'pop-rock acoustic male vocalists who conquered Billboard with accessible crossover sound, first foreign champion female vocalist who, angelic fragile yet'},
    'VM0702': {'cat':'C','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'wordplay-brilliant duo male vocalists with pop-friendly beat chemistry, androgynous cold urban, desperately emotional high-pitched'},
    'VM0703': {'cat':'C','tag':'투명한 음색','w':[40, 40, 20],'prompt':'sky-high angelic male falsetto vocal soaring, R&B-infused female vocalists riding 808 glide bass , fairy female'},
    'VM0704': {'cat':'C','tag':'압도적 고음','w':[60, 30, 10],'prompt':'sacred powerful metal male vocal with commanding volume from small frame, pansori-master young melting fierce traditional, crystal-clear healing female'},
    'VM0705': {'cat':'C','tag':'감성 폭발','w':[40, 40, 20],'prompt':'sophisticated mid-century bridging modern, deep powerful female vocal consuming jazz rock and, ultra-fast UK grime female'},
    'VM0706': {'cat':'C','tag':'리듬 보컬','w':[50, 40, 10],'prompt':'autotune-wielding male vocalists who perfected modern melodic trap with hypnotic delivery, deep heavy contralto female vocal singing the, girl-group trained female'},
    'VM0707': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'energetic flashy male vocalists with wild hybrid flow over hardcore club beats, most sophisticated calm sensual mid-low female, new-wave rock female vocalists'},
    'VM0708': {'cat':'C','tag':'펑키 그루브','w':[50, 30, 20],'prompt':'authoritative smooth male vocalists with business-mogul swagger and effortless delivery, saddest tone in jazz history, wounded soul female, original all-rounder'},
    'VM0709': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'explosive rapid-fire male vocalists with razor-sharp diction and unmatched global impact, explosive power from small frame, timeless clear, prodigious genius female'},
    'VM0710': {'cat':'C','tag':'감성 폭발','w':[60, 20, 20],'prompt':'prodigy mastering saxophone to orchestra with epic narrative depth, emerging female, smooth silky male'},
    'VM0711': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'lightning-fast male vocalists layering angelic melodies over rapid-fire delivery uniquely, The Voice, perfect female vocal with flawless power pitch, folk-ballad optimized male'},
    'VM0712': {'cat':'C','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, clear bright female pop, glamorous west-coast female'},
    'VM0713': {'cat':'C','tag':'거친 소울','w':[60, 30, 10],'prompt':'rugged bending-note cutting through grand horn and string ensembles, world-class 5-octave powerful with dramatic high, folk-ballad optimized male'},
    'VM0714': {'cat':'C','tag':'리듬 보컬','w':[40, 40, 20],'prompt':'triplet-flow male vocalists who rewrote global trap, girl-group trained hiding solid traditional, soft breathy warm'},
    'VM0715': {'cat':'C','tag':'크리스탈 톤','w':[60, 30, 10],'prompt':'bright cheerful male pop vocal with catchy melodic 60s piano flair, sharp organic indie-trap female vocalists creating the, heavyweight hardcore'},
    'VM0716': {'cat':'C','tag':'중저음 매력','w':[50, 40, 10],'prompt':'thunderous deep-cave male vocalists who exploded Brooklyn , sorrowful French chanson female vocal pouring raw life, storytelling piano male'},
    'VM0717': {'cat':'C','tag':'허스키 감성','w':[70, 20, 10],'prompt':'raw rough soul-shaking male rock vocal with emotional honesty, punk-rage female vocalists pioneering, bright energetic male'},
    'VM0718': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'earthy rustic combining rural folk sentiment with tradition, legendary high-tone female vocalists with rhythmic agility, polished Atlanta trap'},
    'VM0719': {'cat':'C','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'stable soaring high-note riding grand traditional melodies, explosive raspy female vocalists with southern trap energy, deep literary lyrical'},
    'VM0720': {'cat':'C','tag':'그루브 보컬','w':[40, 40, 20],'prompt':'fleet-footed male vocalists with dazzling speed and showmanship, flawless classic female vocal mastering Broadway and, wildly innovative southern male'},
    'VM0721': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'versatile raw male vocalists spanning distorted lo-fi beats to tender acoustic rap, musical-theater trained female power- with, passionate climbing male'},
    'VM0722': {'cat':'C','tag':'압도적 고음','w':[60, 20, 20],'prompt':'powerful open-throated male singer belting folk sorrows with piercing clarity, textbook traditional female , pure refreshing female'},
    'VM0723': {'cat':'C','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, UK club hyperpop female vocalists crossing electronic beats, first lady of jazz'},
    'VM0724': {'cat':'C','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'classical crossover harmonizing operatic power with grand orchestral scale, sharp high-tone female vocalists with clever off-beat, pansori-master young female'},
    'VM0725': {'cat':'C','tag':'허스키 매력','w':[70, 20, 10],'prompt':'husky gravelly male vocalists delivering authentic Atlanta street narratives with grit, quiet warm soothing, hypnotic baby-voice male'},
    'VM0726': {'cat':'C','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'silky warm healing gentle male vocal like velvet, competition-bred female, passionate revolutionary male'},
    'VM0727': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'explosive operatic metal male vocal like a, innovative female vocalist-producer with revolutionary visual, understated monotone'},
    'VM0728': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'uniquely flavored with signature nasal bending-note technique mastery, poetic jazz-harmony female, relentless southern female'},
    'VM0729': {'cat':'C','tag':'시원한 고음','w':[40, 40, 20],'prompt':'sorrowful falsetto transitioning to angry melodic, fairy female vocal with perfect breath, Afrobeat- hybrid female'},
    'VM0730': {'cat':'C','tag':'그루브 보컬','w':[50, 30, 20],'prompt':'fleet-footed male vocalists with dazzling speed and showmanship from the golden era, ethereal theatrical falsetto female vocal with, prodigious genius female'},
    'VM0731': {'cat':'C','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'mournful mid-bass commanding orchestral-scale grand ballad narratives, powerful venue-shaking female, idol-crossover female'},
    'VM0732': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'chart-dominating male vocalists-singer who demolished the, raw powerful black-soul-based female belting vocal, refined groovy male'},
    'VM0733': {'cat':'C','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'raspy warm male rock vocal with anthemic sing-along ballad grit, understated monotone , smooth classic'},
    'VM0734': {'cat':'C','tag':'맑은 감성','w':[70, 20, 10],'prompt':'folk-rooted gentle comforting the nation with plain warm delivery, revolutionary female vocalists, dreamy sophisticated falsetto'},
    'VM0735': {'cat':'C','tag':'압도적 고음','w':[60, 30, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, androgynous cold urban charismatic female vocal, sharp nervous ultra-high'},
    'VM0736': {'cat':'C','tag':'압도적 고음','w':[50, 30, 20],'prompt':'rhythmic all-rounder with powerful diction and stage-breaking energy, rich-volume female cinematic standing firm, explosive operatic metal'},
    'VM0737': {'cat':'C','tag':'허스키 감성','w':[60, 20, 20],'prompt':'explosive raspy with gut-wrenching sorrow and raw emotional power, punk-rage female vocalists pioneering, powerful venue-shaking female'},
    'VM0738': {'cat':'C','tag':'일렉트로닉','w':[60, 30, 10],'prompt':'rapid-fire technical male vocalists balancing speed with accessible pop-ballad sensibility, husky soulful female vocal fusing and, most sophisticated calm'},
    'VM0739': {'cat':'C','tag':'리듬 보컬','w':[70, 20, 10],'prompt':'solid high male vocal with retro and modern groove, energetic performer, deep-voiced southern female vocalists, dramatic operatic male'},
    'VM0740': {'cat':'C','tag':'투명한 음색','w':[70, 20, 10],'prompt':'versatile male vocal from soft falsetto to rock screaming, deep husky soulful, highway queen female'},
    'VM0741': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'cold atmospheric male vocalists delivering chilling street narratives with west-coast cool, deep powerful female vocal, cute nasally charming'},
    'VM0742': {'cat':'C','tag':'허스키 매력','w':[50, 30, 20],'prompt':'rough torn raspy male vocal pouring soul until the last breath, fierce Miami trap duo female vocalists with unrestrained, pansori-based male'},
    'VM0743': {'cat':'C','tag':'크리스탈 톤','w':[60, 30, 10],'prompt':'transparent fragile male vocal with crystalline sad tone and quiet intensity, fierce female vocalists from Ruff Ryders dominating 2000s, world-class 5-octave powerful'},
    'VM0744': {'cat':'C','tag':'그루브 보컬','w':[40, 40, 20],'prompt':'heart-wrenching melodic male vocalists who epitomized, brilliant nightingale celebrated as the, rapid-fire technical male'},
    'VM0745': {'cat':'C','tag':'허스키 매력','w':[70, 20, 10],'prompt':'overwhelming male R&B lead vocal with rich harmonics, fierce female vocalists from, androgynous cold urban'},
    'VM0746': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'solid high male vocal with retro and modern groove, textbook with decades of live, multi-talented male vocalists-actor'},
    'VM0747': {'cat':'C','tag':'투명한 음색','w':[60, 20, 20],'prompt':'quintessentially optimistic with earthy rustic warmth and joy, perfect powerful female R&B, refined'},
    'VM0748': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'master-architect radically mixing pansori, technically versatile female vocalists freely riding R&B, globally acclaimed UK'},
    'VM0749': {'cat':'C','tag':'파워 보컬','w':[50, 30, 20],'prompt':'overwhelming falsetto high male vocal dominating karaoke, flawless classic female vocal mastering Broadway and, bold thick-toned'},
    'VM0750': {'cat':'C','tag':'감성 보컬','w':[70, 20, 10],'prompt':'smooth classic baritone male crooner jazz pop vocal, rich-volume female cinematic, saddest tone in jazz'},
    'VM0751': {'cat':'D','tag':'맑은 감성','w':[50, 30, 20],'prompt':'crystalline pure-toned male tenor revered as the emperor of classic , eccentric brilliant male vocalists commanding neo-soul, husky deep mid-low female'},
    'VM0752': {'cat':'D','tag':'감성 보컬','w':[60, 20, 20],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, raw aching male piano, delicate symphonic metal'},
    'VM0753': {'cat':'D','tag':'감성 폭발','w':[60, 20, 20],'prompt':'lush romantic male vocal blending classical piano grandeur with pop yearning, soulful mezzo-soprano with, quintessentially optimistic male'},
    'VM0754': {'cat':'D','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'solid high male vocal with retro and modern groove, energetic performer, soft gentle tenor, the living goddess of'},
    'VM0755': {'cat':'D','tag':'맑은 감성','w':[50, 30, 20],'prompt':'pristine classical crossover soprano with angelic symphonic serenity, explosive powerhouse belter with massive, relaxed mellow baritone'},
    'VM0756': {'cat':'D','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'brilliant nightingale celebrated as the golden voice of the 50s-60s, melodic hook-master male vocalist who defined G-Funk, smooth romantic male vocalists'},
    'VM0757': {'cat':'D','tag':'파워 보컬','w':[50, 40, 10],'prompt':'powerful heartfelt classic pop male vocal with piano accompaniment, paradigm-shifting male vocalist-producer and genre, legendary nasal-melody female'},
    'VM0758': {'cat':'D','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'explosive power from small frame, timeless clear sorrowful , anthem trance, heart-wrenching, bold theatrical baritone'},
    'VM0759': {'cat':'D','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'classically trained powerful high-note who launched power- era, modern male vocalists, velvety deep crooning'},
    'VM0760': {'cat':'D','tag':'감성 폭발','w':[50, 30, 20],'prompt':'silky smooth sensual male Motown soul vocal, warm intimate male folk pop vocal, explosive female disco'},
    'VM0761': {'cat':'D','tag':'천상의 목소리','w':[50, 40, 10],'prompt':'elegant 60s layering sophisticated arrangements over traditional melody, soft gentle tenor with intimate breathy delivery, genius sensual male vocal'},
    'VM0762': {'cat':'D','tag':'폭발 에너지','w':[50, 40, 10],'prompt':'powerful UK national male vocalists fusing grime with classic soul harmonics brilliantly, dramatic powerful male rock vocal with 4-octave theatrical, poetic conscious male vocalists'},
    'VM0763': {'cat':'D','tag':'중저음 매력','w':[60, 20, 20],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, agile scatting tenor, cerebral eloquent male'},
    'VM0764': {'cat':'D','tag':'일렉트로닉','w':[60, 30, 10],'prompt':'solid high male vocal with retro and modern groove, multi-talented male vocalist-producer with refined west-coast, clear bright female pop'},
    'VM0765': {'cat':'D','tag':'압도적 고음','w':[60, 30, 10],'prompt':'explosive power from small frame, timeless clear sorrowful , precise rhythmic Swedish diva, Clean Bandit, airy ethereal male'},
    'VM0766': {'cat':'D','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'legendary Three 6 Mafia female vocalists ruling southern underground with fierce authority, positive upbeat rhythmic, technically gifted male'},
    'VM0767': {'cat':'D','tag':'일렉트로닉','w':[60, 30, 10],'prompt':'young luminous classical crossover soprano with operatic innocence and grace, fierce female vocalists from Ruff Ryders dominating 2000s, cute nasally charming'},
    'VM0768': {'cat':'D','tag':'크리스탈 톤','w':[50, 30, 20],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, pioneering male vocalists mastering both hardcore , world-class soprano with'},
    'VM0769': {'cat':'D','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, revolutionary female vocalists, smooth R&B singing over'},
    'VM0770': {'cat':'D','tag':'중저음 매력','w':[60, 20, 20],'prompt':'charming deep baritone male vocal, the king of rock and roll, distinctive high-pitched male, suave romantic baritone'},
    'VM0771': {'cat':'D','tag':'파워 보컬','w':[60, 20, 20],'prompt':'classical crossover harmonizing operatic power with grand orchestral scale, punk-rock crossover male, sophisticated folk soprano'},
    'VM0772': {'cat':'D','tag':'감성 보컬','w':[40, 40, 20],'prompt':'brilliant nightingale celebrated as the, rough raspy alto with raw heartfelt, hard-hitting precise male'},
    'VM0773': {'cat':'D','tag':'감성 보컬','w':[50, 30, 20],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, pioneering male vocalists who defined modern rhyme, gritty warm male'},
    'VM0774': {'cat':'D','tag':'거친 소울','w':[70, 20, 10],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, rhythmic male pop, rustic mid-low bending-note'},
    'VM0775': {'cat':'D','tag':'그루브 보컬','w':[50, 30, 20],'prompt':'legendary Three 6 Mafia female vocalists ruling southern underground with fierce authority, heavy dubstep-trap hybrid, dark aggressive bass, explosive operatic metal'},
    'VM0776': {'cat':'D','tag':'거친 소울','w':[50, 30, 20],'prompt':'elegant 60s layering sophisticated arrangements over traditional melody, melodic hook-master male vocalist who defined G-Funk, refined'},
    'VM0777': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'retro 8-bit electro-trap pioneer, funky chiptune rebel, original genre bender, heavy 808 trap EDM, deep powerful female'},
    'VM0778': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'explosive power from small frame, timeless clear sorrowful , foundational male vocalists who architected modern trap, ethereal theatrical falsetto'},
    'VM0779': {'cat':'D','tag':'시원한 고음','w':[60, 20, 20],'prompt':'pristine classical crossover soprano with angelic symphonic serenity, powerfully raspy male, delicate lyrical tenor'},
    'VM0780': {'cat':'D','tag':'감성 폭발','w':[60, 20, 20],'prompt':'legendary harmony showcasing textbook traditional duet vocal mastery, warm intimate male, smooth R&B singing over'},
    'VM0781': {'cat':'D','tag':'펑키 그루브','w':[60, 30, 10],'prompt':'legendary harmony showcasing textbook traditional duet vocal mastery, revolutionary male vocalists who weaponized his voice as, rapid-fire versatile female'},
    'VM0782': {'cat':'D','tag':'파워 보컬','w':[60, 20, 20],'prompt':'majestic operatic tenor with soaring classical purity and grand resonance, commanding dramatic diva, haunting atmospheric'},
    'VM0783': {'cat':'D','tag':'감성 폭발','w':[60, 30, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery that makes the world cry, accessible narrative male vocalists layering popular, lyrical light tenor'},
    'VM0784': {'cat':'D','tag':'감성 폭발','w':[70, 20, 10],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, rhythmic male pop, The Voice, perfect female'},
    'VM0785': {'cat':'D','tag':'파워 보컬','w':[70, 20, 10],'prompt':'elegant refined tenor with classically elevated harmonic vocal phrasing, inventive creative female vocalists, feathery high tenor'},
    'VM0786': {'cat':'D','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'wise philosophical male vocalists layering classic soul with modern narratives, wordplay-brilliant duo male vocalists with, passionate revolutionary male'},
    'VM0787': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, feathery soft tender male vocal with, NYC underground queen'},
    'VM0788': {'cat':'D','tag':'깊은 울림','w':[70, 20, 10],'prompt':'charming deep baritone male vocal, the king of rock and roll, calm low mid-range, laid-back mellow baritone'},
    'VM0789': {'cat':'D','tag':'거친 소울','w':[60, 30, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery that makes the world cry, desperately emotional high-pitched male , creative fusion soprano'},
    'VM0790': {'cat':'D','tag':'감성 보컬','w':[60, 30, 10],'prompt':'refined French chanteuse with classic cinematic vocal elegance, raw desperate soprano with unfiltered emotional intensity, solid high male'},
    'VM0791': {'cat':'D','tag':'투명한 음색','w':[50, 30, 20],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, warm versatile mezzo with theatrical, revolutionary male vocalists who'},
    'VM0792': {'cat':'D','tag':'거친 소울','w':[60, 30, 10],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, barefoot diva, deeply appealing drawn from, polished hybrid male'},
    'VM0793': {'cat':'D','tag':'크리스탈 톤','w':[70, 20, 10],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, humorous witty male , nervous yet beautiful'},
    'VM0794': {'cat':'D','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'charming deep baritone male vocal, the king of rock and roll, raw angular mezzo, doll-faced female'},
    'VM0795': {'cat':'D','tag':'감성 보컬','w':[40, 40, 20],'prompt':'deep classic husky female vocal with, powerful rough soulful female blues vocal, crystal clear yet'},
    'VM0796': {'cat':'D','tag':'깊은 울림','w':[60, 20, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, polished velvety, mellow melodic male vocalists'},
    'VM0797': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'classical crossover harmonizing operatic power with grand orchestral scale, folk-rock gentle female , husky theatrical baritone'},
    'VM0798': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'elegant silken high tenor with refined classical pop gentle sophistication, razor-sharp punchline male vocalists with dazzling lyrical, soft dreamy'},
    'VM0799': {'cat':'D','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'solid high male vocal with retro and modern groove, energetic performer, fierce sharp-tongued New York female vocalists, legendary harmony female'},
    'VM0800': {'cat':'D','tag':'일렉트로닉','w':[60, 30, 10],'prompt':'legendary high-tone female vocalists with rhythmic agility and iconic vocal presence, rhythmic groove master who electrified, agile scatting tenor'},
    'VM0801': {'cat':'D','tag':'투명한 음색','w':[60, 30, 10],'prompt':'young luminous classical crossover soprano with operatic innocence and grace, sharp nervous ultra-high screaming male rock, wise philosophical male'},
    'VM0802': {'cat':'D','tag':'천상의 목소리','w':[60, 30, 10],'prompt':'elegant silken high tenor with refined classical pop gentle sophistication, warm low-register evoking hometown, clear pristine soprano'},
    'VM0803': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'powerful heartfelt classic pop male vocal with piano accompaniment, soft dreamy, gravelly soulful'},
    'VM0804': {'cat':'D','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'elegant silken high tenor with refined classical pop gentle sophistication, The Voice, perfect female vocal with flawless, sharp organic indie-trap'},
    'VM0805': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'charismatic retro tenor with vintage seventies pop-rock warmth and flair, underground technical female, whisper-soft literary female vocalists'},
    'VM0806': {'cat':'D','tag':'굵은 바리톤','w':[60, 20, 20],'prompt':'rustic mid-low bending-note anchoring legendary harmony foundations, authentic R&B alto, barefoot diva, deeply appealing'},
    'VM0807': {'cat':'D','tag':'파워 보컬','w':[40, 40, 20],'prompt':'classically trained powerful high-note , commanding female vocalists who elevated with social, blended operatic tenor'},
    'VM0808': {'cat':'D','tag':'펑키 그루브','w':[50, 30, 20],'prompt':'legendary harmony showcasing textbook traditional duet vocal mastery, explosive Canadian male vocalists-singer with 80s-90s, lightning-fast male vocalists'},
    'VM0809': {'cat':'D','tag':'압도적 고음','w':[70, 20, 10],'prompt':'elegant refined tenor with classically elevated harmonic vocal phrasing, iconic pop female vocal, sharp sorrowful tenor'},
    'VM0810': {'cat':'D','tag':'맑은 감성','w':[40, 40, 20],'prompt':'pure clean soprano capturing quiet depth, explosive belting soprano with piercing volume, transparent dewdrop-clear soprano'},
    'VM0811': {'cat':'D','tag':'투명한 음색','w':[60, 20, 20],'prompt':'crystalline nightingale who dominated early classic , bouncy yet heartfelt, breathy dreamy falsetto'},
    'VM0812': {'cat':'D','tag':'감성 보컬','w':[60, 20, 20],'prompt':'soulful male vocal with deep pain and emotion, the godfather of soul, theatrical sweeping male, world-class 5-octave powerful'},
    'VM0813': {'cat':'D','tag':'투명한 음색','w':[50, 40, 10],'prompt':'pristine classical crossover soprano with angelic symphonic serenity, highway queen with tender yet, trend-setting'},
    'VM0814': {'cat':'D','tag':'거친 소울','w':[70, 20, 10],'prompt':'bright cheerful male pop vocal with catchy melodic 60s piano flair, prodigious genius female, sorrowful French chanson female'},
    'VM0815': {'cat':'D','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'legendary 80s female vocalists who spearheaded mainstream with infectious energy, - queen female vocalist who conquered both Japan, revolutionary UK grime male'},
    'VM0816': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, plaintive gentle soothing homesick hearts, rustic mid-low bending-note'},
    'VM0817': {'cat':'D','tag':'파워 보컬','w':[70, 20, 10],'prompt':'powerful UK national male vocalists fusing grime with classic soul harmonics brilliantly, funky freewheeling male, unwavering cool'},
    'VM0818': {'cat':'D','tag':'굵은 바리톤','w':[50, 40, 10],'prompt':'rustic mid-low bending-note anchoring legendary harmony foundations, warm folk acoustic , dreamy Latin-pop female'},
    'VM0819': {'cat':'D','tag':'허스키 매력','w':[60, 20, 20],'prompt':'rustic mid-low bending-note anchoring legendary harmony foundations, cinematic , legendary harmony female'},
    'VM0820': {'cat':'D','tag':'거친 소울','w':[70, 20, 10],'prompt':'textbook traditional with the most flavorful classic bending delivery, youthful clear tenor, gentle wistful male'},
    'VM0821': {'cat':'D','tag':'펑키 그루브','w':[50, 30, 20],'prompt':'solid high male vocal with retro and modern groove, transparent earnest soprano with heartfelt, gritty soulful'},
    'VM0822': {'cat':'D','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'classical soprano female vocal pioneering symphonic metal genre, crystal clear yet steel-strong female belting, atmospheric Canadian male'},
    'VM0823': {'cat':'D','tag':'깊은 베이스','w':[60, 20, 20],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, dark European female, smoky low alto'},
    'VM0824': {'cat':'D','tag':'압도적 고음','w':[60, 20, 20],'prompt':'explosive power from small frame, timeless clear sorrowful , multi-genre soprano with, sweet lyrical soprano'},
    'VM0825': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'legendary storytelling male vocalists with unique accent and theatrical narrative flow, androgynous cold urban, explosive power from'},
    'VM0826': {'cat':'D','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'classical crossover harmonizing operatic power, intense 90s New York hardcore female vocalists, dignified low-tone male'},
    'VM0827': {'cat':'D','tag':'리듬 보컬','w':[70, 20, 10],'prompt':'legendary Three 6 Mafia female vocalists ruling southern underground with fierce authority, ethereal breathtaking soprano, global Billboard-hitting female'},
    'VM0828': {'cat':'D','tag':'허스키 감성','w':[50, 40, 10],'prompt':'legendary nasal-melody who comforted a colonized nation with sorrow, clear smooth pure falsetto, heavy dubstep-trap hybrid'},
    'VM0829': {'cat':'D','tag':'크리스탈 톤','w':[70, 20, 10],'prompt':'crystalline pure-toned male tenor revered as the emperor of classic , inventive creative female vocalists, deep emotional dramatic'},
    'VM0830': {'cat':'D','tag':'압도적 고음','w':[50, 40, 10],'prompt':'explosive power from small frame, timeless clear sorrowful , powerful rich baritone-tenor with sweeping orchestral, crystalline nightingale female'},
    'VM0831': {'cat':'D','tag':'감성 폭발','w':[50, 40, 10],'prompt':'brilliant nightingale celebrated as the golden voice of the 50s-60s, raw aching male piano vocal that erupts from, dreamy sophisticated'},
    'VM0832': {'cat':'D','tag':'천상의 목소리','w':[50, 40, 10],'prompt':'legendary pure high male tenor with effortless sustained arena rock notes, feathery soft tender male vocal with, flawless crystal clear'},
    'VM0833': {'cat':'D','tag':'거친 소울','w':[40, 40, 20],'prompt':'legendary harmony showcasing textbook, underground legend male vocalists who built southern, thunderous deep-cave male vocalists'},
    'VM0834': {'cat':'D','tag':'폭발 에너지','w':[50, 40, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, hook-driven addictive dominating with catchy, legendary harmony female'},
    'VM0835': {'cat':'D','tag':'시원한 고음','w':[40, 40, 20],'prompt':'powerful UK national male vocalists fusing grime with, overwhelming falsetto high male , majestic operatic tenor'},
    'VM0836': {'cat':'D','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, passionate tender tenor with soulful Latin, anthem trance, heart-wrenching'},
    'VM0837': {'cat':'D','tag':'파워 보컬','w':[70, 20, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, rapid-fire versatile female, flawless technique male'},
    'VM0838': {'cat':'D','tag':'감성 폭발','w':[60, 30, 10],'prompt':'legendary harmony showcasing textbook traditional duet vocal mastery, foundational male vocalists who architected modern trap, soft dreamy'},
    'VM0839': {'cat':'D','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'most destructive female rock vocal in, plaintive gentle soothing homesick hearts, transcendent tenor with'},
    'VM0840': {'cat':'D','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'powerful heartfelt classic pop male vocal, wildly innovative southern male vocalists and genre, refreshing clear soprano'},
    'VM0841': {'cat':'D','tag':'중저음 매력','w':[50, 30, 20],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, sweet romantic tenor with handsome, rhythmic male pop'},
    'VM0842': {'cat':'D','tag':'일렉트로닉','w':[40, 40, 20],'prompt':'classical soprano female vocal pioneering, intellectually refined male vocalists blending conscious, Miami hardcore female vocalists'},
    'VM0843': {'cat':'D','tag':'깊은 베이스','w':[60, 20, 20],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, next-generation hardcore female, Miss champion female'},
    'VM0844': {'cat':'D','tag':'감성 보컬','w':[50, 30, 20],'prompt':'flawless classic female vocal mastering Broadway and pop with zero error, husky powerful soprano with overwhelming, underground technical female'},
    'VM0845': {'cat':'D','tag':'일렉트로닉','w':[50, 40, 10],'prompt':'legendary 80s female vocalists who spearheaded mainstream with infectious energy, deep-voiced southern female vocalists with heavy 808 impact, London-born healing female'},
    'VM0846': {'cat':'D','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'legendary nasal-melody who comforted a colonized nation with sorrow, Afrobeat- fusion male vocalists Afroswing, explosive female disco'},
    'VM0847': {'cat':'D','tag':'폭발 에너지','w':[40, 40, 20],'prompt':'soaring rock soprano with legendary, intimate whispery baritone with atmospheric, powerful rough soulful'},
    'VM0848': {'cat':'D','tag':'폭발 에너지','w':[50, 30, 20],'prompt':'queen of soul, gospel-based explosive powerful female vocal full of holy spirit, bright energetic radiating vitality with, rapid-fire technical male'},
    'VM0849': {'cat':'D','tag':'압도적 고음','w':[60, 30, 10],'prompt':'legendary pure high male tenor with effortless sustained arena rock notes, powerful pansori-toned who made the, deep resonant female'},
    'VM0850': {'cat':'D','tag':'펑키 그루브','w':[60, 30, 10],'prompt':'solid high male vocal with retro and modern groove, pure crystal-clear folk soprano with gentle, refined French'},
    'VM0851': {'cat':'D','tag':'감성 보컬','w':[60, 30, 10],'prompt':'lush romantic male vocal blending classical piano grandeur with pop yearning, evolved male vocal reaching divine territory from folk, Australian-born female vocalists'},
    'VM0852': {'cat':'D','tag':'맑은 천상','w':[40, 40, 20],'prompt':'young luminous classical crossover soprano with, bright narrative acoustic pop vocal with, rich deep mid-low'},
    'VM0853': {'cat':'D','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'legendary 80s female vocalists who spearheaded mainstream with infectious energy, technically versatile female, explosive raspy female vocalists'},
    'VM0854': {'cat':'D','tag':'리듬 보컬','w':[50, 30, 20],'prompt':'retro 8-bit electro-trap pioneer, funky chiptune rebel, original genre bender, refined velvety tenor with elegant soaring, nervous yet beautiful'},
    'VM0855': {'cat':'D','tag':'투명한 음색','w':[70, 20, 10],'prompt':'young luminous classical crossover soprano with operatic innocence and grace, uniquely flavored male, girl-group trained female'},
    'VM0856': {'cat':'D','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'solid high male vocal with retro and modern groove, global Billboard-hitting female vocalists with Thai-international swagger, cute nasally charming'},
    'VM0857': {'cat':'D','tag':'크리스탈 톤','w':[40, 40, 20],'prompt':'pristine classical crossover soprano with, underground legend male vocalists who built southern, refined slightly nasal'},
    'VM0858': {'cat':'D','tag':'천상의 목소리','w':[60, 30, 10],'prompt':'most destructive female rock vocal in history, blood-vessel-popping screaming wail, warm intimate male folk pop vocal, modern healing tenor'},
    'VM0859': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, clear bright female pop vocal with ultra-high technique, laid-back mellow baritone'},
    'VM0860': {'cat':'D','tag':'압도적 고음','w':[60, 20, 20],'prompt':'legendary pure high male tenor with effortless sustained arena rock notes, delicate symphonic metal, polished velvety'},
    'VM0861': {'cat':'D','tag':'맑은 천상','w':[60, 20, 20],'prompt':'bright cheerful male pop vocal with catchy melodic 60s piano flair, soft dreamy, transcendent tenor with'},
    'VM0862': {'cat':'D','tag':'투명한 음색','w':[60, 20, 20],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, pansori-based male , genius sensual male vocal'},
    'VM0863': {'cat':'D','tag':'깊은 베이스','w':[60, 20, 20],'prompt':'smooth classic baritone male crooner jazz pop vocal, global Billboard-hitting female, raspy warm male'},
    'VM0864': {'cat':'D','tag':'리듬 보컬','w':[40, 40, 20],'prompt':'solid high male vocal with retro, raw rough soul-shaking male rock vocal, pure crystal-clear folk'},
    'VM0865': {'cat':'D','tag':'감성 폭발','w':[40, 40, 20],'prompt':'elegant 60s layering sophisticated, trendy stylish male vocalists layering fashion-forward aesthetics, precise rhythmic Swedish'},
    'VM0866': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'earnest warm tenor with pure heartfelt delivery and classic ballad phrasing, precisely polished , modern male vocalists'},
    'VM0867': {'cat':'D','tag':'폭발 에너지','w':[40, 40, 20],'prompt':'majestic operatic tenor with soaring classical, raw explosive belting vocal tearing through, globally acclaimed UK'},
    'VM0868': {'cat':'D','tag':'압도적 고음','w':[70, 20, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, the living goddess of, heavy dubstep-trap hybrid'},
    'VM0869': {'cat':'D','tag':'감성 보컬','w':[50, 40, 10],'prompt':'brilliant nightingale celebrated as the golden voice of the 50s-60s, trembling soulful male vocal with aching falsetto, smooth classic'},
    'VM0870': {'cat':'D','tag':'거친 소울','w':[60, 20, 20],'prompt':'soulful male vocal with deep pain and emotion, the godfather of soul, genius lyricist UK, percussion-performing male'},
    'VM0871': {'cat':'D','tag':'천상의 목소리','w':[60, 20, 20],'prompt':'elegant 60s layering sophisticated arrangements over traditional melody, minimal clear soprano, refreshing bright rock'},
    'VM0872': {'cat':'D','tag':'깊은 베이스','w':[40, 40, 20],'prompt':'rebellious melancholic raw retro soul jazz female vocal, immortal deep baritone who elevated, deep heavy contralto'},
    'VM0873': {'cat':'D','tag':'굵은 바리톤','w':[40, 40, 20],'prompt':'velvety deep crooning baritone with effortless, technically gifted male vocalists with extraordinary rhyme, natural conversational mid-range'},
    'VM0874': {'cat':'D','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'charming deep baritone male vocal, the king of rock and roll, distinctive husky low-tone male vocalists with gritty, bold theatrical baritone'},
    'VM0875': {'cat':'D','tag':'맑은 감성','w':[50, 40, 10],'prompt':'crystalline pure-toned male tenor revered as the emperor of classic , romantic aged baritone with weathered folk, explosive energy rough husky'},
    'VM0876': {'cat':'D','tag':'깊은 울림','w':[60, 20, 20],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, powerfully raspy male, wailing blues-rock male'},
    'VM0877': {'cat':'D','tag':'압도적 고음','w':[40, 40, 20],'prompt':'powerful heartfelt classic pop male vocal, operatic tenor completing orchestral-scale, inventive jazzy soprano'},
    'VM0878': {'cat':'D','tag':'일렉트로닉','w':[60, 20, 20],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, trailblazing New York female, precision-engineered modern male'},
    'VM0879': {'cat':'D','tag':'감성 보컬','w':[50, 30, 20],'prompt':'silky smooth sensual male Motown soul vocal, smooth R&B singing over 808 glide bass trap, legendary harmony female'},
    'VM0880': {'cat':'D','tag':'천상의 목소리','w':[60, 20, 20],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, smooth urban tenor, genius singer-songwriter male'},
    'VM0881': {'cat':'D','tag':'파워 보컬','w':[50, 40, 10],'prompt':'soaring rock soprano with legendary high-range stadium power, rich earthy male baritone comforting working-class, smooth romantic male vocalists'},
    'VM0882': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'charismatic retro tenor with vintage seventies pop-rock warmth and flair, warm versatile mezzo with theatrical, prodigy'},
    'VM0883': {'cat':'D','tag':'파워 보컬','w':[50, 30, 20],'prompt':'elegant silken high tenor with refined classical pop gentle sophistication, theatrical sweeping male piano ballad vocal, whisper-soft literary female vocalists'},
    'VM0884': {'cat':'D','tag':'시원한 고음','w':[50, 30, 20],'prompt':'most destructive female rock vocal in history, blood-vessel-popping screaming wail, multi-genre soprano with piercing high notes, classical crossover male'},
    'VM0885': {'cat':'D','tag':'파워 보컬','w':[50, 30, 20],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, revolutionary male vocalists who weaponized his voice as, fairy female'},
    'VM0886': {'cat':'D','tag':'맑은 천상','w':[70, 20, 10],'prompt':'pristine classical crossover soprano with angelic symphonic serenity, pure crystalline tenor, trendy urban mezzo'},
    'VM0887': {'cat':'D','tag':'압도적 고음','w':[70, 20, 10],'prompt':'majestic operatic tenor with soaring classical purity and grand resonance, dance anthem, creative fusion soprano'},
    'VM0888': {'cat':'D','tag':'리듬 보컬','w':[60, 30, 10],'prompt':'legendary Three 6 Mafia female vocalists ruling southern underground with fierce authority, airy ethereal male falsetto vocal building, Brooklyn deep-voiced female'},
    'VM0889': {'cat':'D','tag':'감성 보컬','w':[50, 40, 10],'prompt':'legendary nasal-melody who comforted a colonized nation with sorrow, smooth R&B singing over 808 glide bass trap, textbook traditional female'},
    'VM0890': {'cat':'D','tag':'허스키 감성','w':[50, 40, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery that makes the world cry, bold thick-toned female vocalists anchoring songs, quirky playful'},
    'VM0891': {'cat':'D','tag':'펑키 그루브','w':[40, 40, 20],'prompt':'husky intelligent female R&B vocal with, explosive Canadian male vocalists-singer with 80s-90s, humorous witty male'},
    'VM0892': {'cat':'D','tag':'천상의 목소리','w':[50, 30, 20],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, most sophisticated calm sensual mid-low female, breathy dreamy falsetto'},
    'VM0893': {'cat':'D','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, precise surgical male vocalists with clean triplet, mellow melodic male vocalists'},
    'VM0894': {'cat':'D','tag':'굵은 바리톤','w':[40, 40, 20],'prompt':'rustic mid-low bending-note , lethal off-beat female vocalists-singer delivering devastating, punchy dynamic male'},
    'VM0895': {'cat':'D','tag':'거친 소울','w':[70, 20, 10],'prompt':'legendary nasal-melody who comforted a colonized nation with sorrow, edgy youthful, ice-cold sad rebellious'},
    'VM0896': {'cat':'D','tag':'투명한 음색','w':[60, 20, 20],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, angsty alternative, unique bright indie'},
    'VM0897': {'cat':'D','tag':'그루브 보컬','w':[40, 40, 20],'prompt':'solid high male vocal with retro and, tireless iron-throated high male rock vocal, deep-voiced southern female vocalists'},
    'VM0898': {'cat':'D','tag':'허스키 매력','w':[50, 30, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, dreamy breathy contralto with cinematic, textbook'},
    'VM0899': {'cat':'D','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'wise philosophical male vocalists layering classic soul with modern narratives, warm intimate male folk pop vocal, husky powerful male'},
    'VM0900': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'earnest warm tenor with pure heartfelt delivery and classic ballad phrasing, UK club hyperpop female, rapid-fire technical male'},
    'VM0901': {'cat':'D','tag':'파워 보컬','w':[60, 30, 10],'prompt':'powerful UK national male vocalists fusing grime with classic soul harmonics brilliantly, commanding male vocalists carrying west-coast and global, tireless iron-throated high'},
    'VM0902': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'classically trained powerful high-note who launched power- era, breezy light acoustic vocal with sunny, autotune-wielding male vocalists'},
    'VM0903': {'cat':'D','tag':'리듬 보컬','w':[50, 30, 20],'prompt':'rustic mid-low bending-note anchoring legendary harmony foundations, nasal high-pitched male vocalists Elvis-inspired charismatic male'},
    'VM0904': {'cat':'D','tag':'폭발 에너지','w':[60, 20, 20],'prompt':'powerful heartfelt classic pop male vocal with piano accompaniment, tireless iron-throated high, power pop-rock EDM'},
    'VM0905': {'cat':'D','tag':'감성 보컬','w':[40, 40, 20],'prompt':'lush romantic male vocal blending classical piano, screaming high tenor with razor-sharp power, fleet-footed male vocalists with'},
    'VM0906': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'classical soprano female vocal pioneering symphonic metal genre, transparent fragile male vocal with crystalline sad, explosive crunk-pioneer male'},
    'VM0907': {'cat':'D','tag':'감성 폭발','w':[60, 30, 10],'prompt':'brilliant nightingale celebrated as the golden voice of the 50s-60s, authoritative male vocalists who defined , delicate lyrical tenor'},
    'VM0908': {'cat':'D','tag':'리듬 보컬','w':[40, 40, 20],'prompt':'legendary 80s female vocalists who spearheaded , deep powerful female vocal consuming jazz rock and, refreshing bright rock'},
    'VM0909': {'cat':'D','tag':'맑은 감성','w':[50, 40, 10],'prompt':'pure clean soprano capturing quiet depth with timeless graceful phrasing, stable powerhouse with the most, folk-rock gentle female'},
    'VM0910': {'cat':'D','tag':'크리스탈 톤','w':[40, 40, 20],'prompt':'polished warm soprano with elegant 60s, NYC underground queen female vocalists embodying alternative, gritty charismatic male'},
    'VM0911': {'cat':'D','tag':'굵은 바리톤','w':[50, 30, 20],'prompt':'rustic mid-low bending-note anchoring legendary harmony foundations, mournful mid-bass commanding, smooth romantic male vocalists'},
    'VM0912': {'cat':'D','tag':'감성 폭발','w':[50, 30, 20],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, relentless southern female vocalists with heavyweight, silky warm'},
    'VM0913': {'cat':'D','tag':'거친 소울','w':[60, 20, 20],'prompt':'bright cheerful male pop vocal with catchy melodic 60s piano flair, distinctive husky baritone, gritty soulful'},
    'VM0914': {'cat':'D','tag':'일렉트로닉','w':[50, 30, 20],'prompt':'powerful heartfelt classic pop male vocal with piano accompaniment, perfect pitch hybrid vocal, Zedd Stay, 2NE1 hardcore female'},
    'VM0915': {'cat':'D','tag':'거친 소울','w':[70, 20, 10],'prompt':'husky intelligent female R&B vocal with classic piano accompaniment, pop-rock acoustic male, deep literary lyrical'},
    'VM0916': {'cat':'D','tag':'허스키 감성','w':[50, 40, 10],'prompt':'legendary harmony showcasing textbook traditional duet vocal mastery, creative fusion soprano blending traditional , genius singer-songwriter male'},
    'VM0917': {'cat':'D','tag':'리듬 보컬','w':[60, 20, 20],'prompt':'solid high male vocal with retro and modern groove, energetic performer, raw powerful black-soul-based, warm intimate male'},
    'VM0918': {'cat':'D','tag':'폭발 에너지','w':[60, 20, 20],'prompt':'legendary high-tone female vocalists with rhythmic agility and iconic vocal presence, sorrowful falsetto transitioning, The Voice, perfect female'},
    'VM0919': {'cat':'D','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'husky intelligent female R&B vocal with classic piano accompaniment, next-generation hardcore female, legendary Three 6 Mafia'},
    'VM0920': {'cat':'D','tag':'감성 보컬','w':[70, 20, 10],'prompt':'husky intelligent female R&B vocal with classic piano accompaniment, pansori-master young female , warm classic tenor'},
    'VM0921': {'cat':'D','tag':'거친 소울','w':[60, 20, 20],'prompt':'the king , versatile male vocal covering rock ballad and folk, warm robust tenor, rough raspy alto'},
    'VM0922': {'cat':'D','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, versatile male vocal, monster-vocal soprano with'},
    'VM0923': {'cat':'D','tag':'감성 폭발','w':[40, 40, 20],'prompt':'soulful male vocal with deep pain and emotion, feathery soft tender male vocal with, global Billboard-hitting female'},
    'VM0924': {'cat':'D','tag':'감성 보컬','w':[60, 30, 10],'prompt':'textbook traditional with the most flavorful classic bending delivery, easygoing sunny tenor with playful organic, revolutionary UK grime male'},
    'VM0925': {'cat':'D','tag':'그루브 보컬','w':[60, 30, 10],'prompt':'Dutch female vocalists who captivated all of Europe with classic flow, airy ethereal male falsetto vocal building, sweeping dramatic tenor'},
    'VM0926': {'cat':'D','tag':'폭발 에너지','w':[50, 30, 20],'prompt':'soaring rock soprano with legendary high-range stadium power, husky powerful male belting vocal with, world-class speed-rap female'},
    'VM0927': {'cat':'D','tag':'허스키 매력','w':[50, 40, 10],'prompt':'elegant 60s layering sophisticated arrangements over traditional melody, breezy light acoustic vocal with sunny, sharp sorrowful tenor'},
    'VM0928': {'cat':'D','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'polished swinging male crooner vocal with, eccentric brilliant male vocalists commanding neo-soul, mellow melodic male vocalists'},
    'VM0929': {'cat':'D','tag':'허스키 감성','w':[70, 20, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, deep heavy charismatic, folk-rooted gentle male'},
    'VM0930': {'cat':'D','tag':'굵은 바리톤','w':[50, 30, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, sophisticated mid-century bridging modern, heavyweight commanding male'},
    'VM0931': {'cat':'D','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'smooth classic baritone male crooner jazz pop vocal, gravelly uniquely husky deep male jazz vocal, one, fairy female'},
    'VM0932': {'cat':'D','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'legendary high-tone female vocalists with rhythmic agility and iconic vocal presence, UK house-DnB hit vocal, chart-dominating, France greatest-selling female'},
    'VM0933': {'cat':'D','tag':'폭발 에너지','w':[60, 20, 20],'prompt':'flawless classic female vocal mastering Broadway and pop with zero error, stable soaring high-note, sad sharp Irish'},
    'VM0934': {'cat':'D','tag':'허스키 매력','w':[60, 20, 20],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, feathery soft tender, lyrical light tenor'},
    'VM0935': {'cat':'D','tag':'크리스탈 톤','w':[40, 40, 20],'prompt':'bright cheerful male pop vocal with catchy, dreamy Latin-pop female vocalists weaving ethereal harmonics, powerful husky alto'},
    'VM0936': {'cat':'D','tag':'투명한 음색','w':[50, 30, 20],'prompt':'pure clean soprano capturing quiet depth with timeless graceful phrasing, sorrowful French chanson female vocal pouring raw life, 5-octave female vocal'},
    'VM0937': {'cat':'D','tag':'허스키 감성','w':[40, 40, 20],'prompt':'legendary nasal-melody who comforted a, deep heavy charismatic low male vocal, literary poetic mezzo-soprano'},
    'VM0938': {'cat':'D','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'legendary storytelling male vocalists with unique accent and theatrical narrative flow, authoritative deep male, electronic-trap crossover male'},
    'VM0939': {'cat':'D','tag':'허스키 매력','w':[60, 30, 10],'prompt':'husky intelligent female R&B vocal with classic piano accompaniment, smooth silky male R&B piano vocal, overwhelming falsetto high'},
    'VM0940': {'cat':'D','tag':'일렉트로닉','w':[50, 30, 20],'prompt':'legendary Three 6 Mafia female vocalists ruling southern underground with fierce authority, laid-back mellow baritone with serene breezy, Eurovision cinematic electro'},
    'VM0941': {'cat':'D','tag':'일렉트로닉','w':[50, 40, 10],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, eccentric brilliant male vocalists commanding neo-soul, explosive hardcore female vocalists'},
    'VM0942': {'cat':'D','tag':'중저음 매력','w':[40, 40, 20],'prompt':'velvety deep crooning baritone with effortless, deep emotional dramatic male , inventive genius male'},
    'VM0943': {'cat':'D','tag':'그루브 보컬','w':[70, 20, 10],'prompt':'solid high male vocal with retro and modern groove, rough textured male, versatile raw male vocalists'},
    'VM0944': {'cat':'D','tag':'맑은 천상','w':[60, 20, 20],'prompt':'pure clean soprano capturing quiet depth with timeless graceful phrasing, rhythmic powerhouse, thunderous military-grade male'},
    'VM0945': {'cat':'D','tag':'폭발 에너지','w':[60, 20, 20],'prompt':'legendary Three 6 Mafia female vocalists ruling southern underground with fierce authority, powerful open-throated male, hard-hitting slide- male'},
    'VM0946': {'cat':'D','tag':'허스키 매력','w':[50, 30, 20],'prompt':'legendary harmony showcasing textbook traditional duet vocal mastery, quirky playful soprano with witty, deep emotional dramatic'},
    'VM0947': {'cat':'D','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'the king , versatile male vocal covering rock ballad and folk, mysterious witch-like rough vibrato female rock, clear earnest tenor'},
    'VM0948': {'cat':'D','tag':'일렉트로닉','w':[50, 40, 10],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, Eurovision cinematic electro queen, grand synthpop, new-wave rock female vocalists'},
    'VM0949': {'cat':'D','tag':'파워 보컬','w':[50, 40, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, new-wave rock female vocalists who shattered boundaries between, elegant silken high'},
    'VM0950': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, energetic flashy male vocalists with wild hybrid flow, refined'},
    'VM0951': {'cat':'D','tag':'감성 보컬','w':[70, 20, 10],'prompt':'legendary nasal-melody who comforted a colonized nation with sorrow, agile scatting tenor, new-wave rock female vocalists'},
    'VM0952': {'cat':'D','tag':'펑키 그루브','w':[60, 30, 10],'prompt':'husky intelligent female R&B vocal with classic piano accompaniment, Australian-born female vocalists who topped Billboard , intense 90s New'},
    'VM0953': {'cat':'D','tag':'감성 폭발','w':[60, 30, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery that makes the world cry, raw desperate soprano with unfiltered emotional intensity, unique bright indie'},
    'VM0954': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'queen of soul, gospel-based explosive powerful female vocal full of holy spirit, pharmacist-turned female , textbook'},
    'VM0955': {'cat':'D','tag':'맑은 감성','w':[50, 40, 10],'prompt':'crystalline nightingale who dominated early classic , honest deep tenor with raw sincerity and, funky freewheeling male'},
    'VM0956': {'cat':'D','tag':'투명한 음색','w':[60, 20, 20],'prompt':'crystalline pure-toned male tenor revered as the emperor of classic , smooth silky male, battle-rap legend female'},
    'VM0957': {'cat':'D','tag':'감성 보컬','w':[40, 40, 20],'prompt':'saddest tone in jazz history, wounded soul female, global trendy tenor with cinematic pop polish, revolutionary female vocalists'},
    'VM0958': {'cat':'D','tag':'거친 소울','w':[50, 40, 10],'prompt':'soulful male vocal with deep pain and emotion, the godfather of soul, electronic-trap crossover male vocalists blending synths with, luxurious deep soulful'},
    'VM0959': {'cat':'D','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'rustic mid-low bending-note anchoring legendary harmony foundations, most sophisticated calm, raw rough soul-shaking'},
    'VM0960': {'cat':'D','tag':'감성 폭발','w':[70, 20, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, pristine clean high-note, punk-rage female vocalists pioneering'},
    'VM0961': {'cat':'D','tag':'허스키 매력','w':[70, 20, 10],'prompt':'legendary nasal-melody who comforted a colonized nation with sorrow, intense dramatic, sad sharp Irish'},
    'VM0962': {'cat':'D','tag':'깊은 베이스','w':[40, 40, 20],'prompt':'refined French chanteuse with classic, romantic aged baritone with weathered folk, heavyweight commanding male'},
    'VM0963': {'cat':'D','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'solid high male vocal with retro and modern groove, energetic performer, minimalist acoustic with deep resonance, sophisticated folk soprano'},
    'VM0964': {'cat':'D','tag':'리듬 보컬','w':[50, 30, 20],'prompt':'charming deep baritone male vocal, the king of rock and roll, wordplay-brilliant duo male vocalists with, soulful jazzy male vocalists'},
    'VM0965': {'cat':'D','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'flawless classic female vocal mastering Broadway and pop with zero error, explosive raspy female vocalists with southern trap energy, legendary Three 6 Mafia'},
    'VM0966': {'cat':'D','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'elegant refined tenor with classically elevated harmonic vocal phrasing, warm intimate male, Latin reggaeton- crossover'},
    'VM0967': {'cat':'D','tag':'감성 폭발','w':[60, 20, 20],'prompt':'refined French chanteuse with classic cinematic vocal elegance, punchy dynamic male, world-class speed-rap female'},
    'VM0968': {'cat':'D','tag':'깊은 울림','w':[70, 20, 10],'prompt':'charming deep baritone male vocal, the king of rock and roll, dreamy sophisticated falsetto, perfect powerful female'},
    'VM0969': {'cat':'D','tag':'거친 소울','w':[50, 40, 10],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, rough torn raspy male vocal pouring soul, pansori-infused cinematic male'},
    'VM0970': {'cat':'D','tag':'깊은 울림','w':[50, 40, 10],'prompt':'smooth classic baritone male crooner jazz pop vocal, soft breathy warm female pop vocal, solid expressive female'},
    'VM0971': {'cat':'D','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'retro 8-bit electro-trap pioneer, funky chiptune rebel, original genre bender, fleet-footed male vocalists with dazzling speed and showmanship, addictive melodic'},
    'VM0972': {'cat':'D','tag':'리듬 보컬','w':[40, 40, 20],'prompt':'timeless elegant male jazz crooner vocal, dreamy Latin-pop female vocalists weaving ethereal harmonics, legendary storytelling male'},
    'VM0973': {'cat':'D','tag':'그루브 보컬','w':[70, 20, 10],'prompt':'legendary 80s female vocalists who spearheaded mainstream with infectious energy, most sophisticated calm, sophisticated folk soprano'},
    'VM0974': {'cat':'D','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, dreamy atmospheric house, lush vocal pads, emotional healing trance'},
    'VM0975': {'cat':'D','tag':'일렉트로닉','w':[60, 30, 10],'prompt':'powerful UK national male vocalists fusing grime with classic soul harmonics brilliantly, rapid-fire technical male vocalists balancing speed, Bronx female'},
    'VM0976': {'cat':'D','tag':'폭발 에너지','w':[40, 40, 20],'prompt':'rebellious melancholic raw retro soul jazz female vocal, smooth urban tenor with silky tone and, high-pitched screaming male'},
    'VM0977': {'cat':'D','tag':'맑은 감성','w':[50, 40, 10],'prompt':'pristine classical crossover soprano with angelic symphonic serenity, heavyweight commanding male vocalists with flawless flow, polished hybrid male'},
    'VM0978': {'cat':'D','tag':'시원한 고음','w':[50, 30, 20],'prompt':'young luminous classical crossover soprano with operatic innocence and grace, husky powerful male belting vocal with, overwhelming falsetto high'},
    'VM0979': {'cat':'D','tag':'중저음 매력','w':[60, 30, 10],'prompt':'wise philosophical male vocalists layering classic soul with modern narratives, resonant deep baritone with dramatic anthemic, mellow melodic male vocalists'},
    'VM0980': {'cat':'D','tag':'압도적 고음','w':[60, 30, 10],'prompt':'majestic operatic tenor with soaring classical purity and grand resonance, cool understated spoken-word folk vocal with, bold theatrical baritone'},
    'VM0981': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'legendary Three 6 Mafia female vocalists ruling southern underground with fierce authority, fierce sharp-tongued New, polished warm soprano'},
    'VM0982': {'cat':'D','tag':'그루브 보컬','w':[60, 30, 10],'prompt':'legendary 80s female vocalists who spearheaded mainstream with infectious energy, rich commanding contralto with majestic, funky freewheeling male'},
    'VM0983': {'cat':'D','tag':'감성 폭발','w':[60, 20, 20],'prompt':'legendary nasal sorrowful uniquely toned female vocal soaking the soul, viral hook-machine female, pinnacle of'},
    'VM0984': {'cat':'D','tag':'그루브 보컬','w':[70, 20, 10],'prompt':'legendary high-tone female vocalists with rhythmic agility and iconic vocal presence, global EDM hit, lush romantic male'},
    'VM0985': {'cat':'D','tag':'일렉트로닉','w':[50, 30, 20],'prompt':'pure clean soprano capturing quiet depth with timeless graceful phrasing, perfect pitch hybrid vocal, Zedd Stay, 2NE1 hardcore female'},
    'VM0986': {'cat':'D','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'wise philosophical male vocalists layering classic soul with modern narratives, tireless iron-throated high, Elvis-inspired charismatic male'},
    'VM0987': {'cat':'D','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'solid high male vocal with retro and modern groove, energetic performer, prodigious genius female cinematic , battle-rap legend female'},
    'VM0988': {'cat':'D','tag':'천상의 목소리','w':[60, 20, 20],'prompt':'powerful UK national male vocalists fusing grime with classic soul harmonics brilliantly, gentle refined male, tender longing falsetto'},
    'VM0989': {'cat':'D','tag':'거친 소울','w':[60, 30, 10],'prompt':'brilliant nightingale celebrated as the golden voice of the 50s-60s, explosive powerhouse belter with massive, pristine smooth male'},
    'VM0990': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, nasal high-pitched male, 2NE1 hardcore female'},
    'VM0991': {'cat':'D','tag':'맑은 감성','w':[70, 20, 10],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, global trendy tenor, rich deep mid-low'},
    'VM0992': {'cat':'D','tag':'일렉트로닉','w':[50, 30, 20],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, pioneering male vocalists who defined modern rhyme, Bronx female'},
    'VM0993': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'classical soprano female vocal pioneering symphonic metal genre, fierce Miami trap duo, young prodigy female'},
    'VM0994': {'cat':'D','tag':'감성 보컬','w':[60, 20, 20],'prompt':'legendary harmony showcasing textbook traditional duet vocal mastery, percussion-performing male , pioneering fierce female'},
    'VM0995': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'classical crossover harmonizing operatic power with grand orchestral scale, aggressive hard-hitting male, pristine clean high-note'},
    'VM0996': {'cat':'D','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, rich velvety contralto with breathtaking sustained, crystalline soaring'},
    'VM0997': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'queen of soul, gospel-based explosive powerful female vocal full of holy spirit, massive cinematic soprano, trendy stylish male'},
    'VM0998': {'cat':'D','tag':'폭발 에너지','w':[50, 40, 10],'prompt':'classical soprano female vocal pioneering symphonic metal genre, massive operatic soprano with stadium-shaking, lethal off-beat female'},
    'VM0999': {'cat':'D','tag':'감성 폭발','w':[40, 40, 20],'prompt':'the king , versatile male vocal, hard-hitting gangster male vocalists delivering textbook, precise pitch-perfect male pop'},
    'VM1000': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'classical crossover harmonizing operatic power with grand orchestral scale, warm low-register evoking hometown, pop-ballad crossover female'},
}


class VvipReq(BaseModel):
    prompt: str = ""
    lyrics: str = ""
    style: str = "pop"
    bpm: int = 120
    key: str = "C"
    chords: str = ""
    vocal: str = ""  # cgo-382: male/female/duet/bgm/child/choir
    voice_mix_id: str = ""  # cgo-390: VOICE_MIX 프리셋 ID (VM0001~VM1000)
    timbre: str = ""        # cgo-444: 앱이 보낸 이름 없는 음색 묘사 (있으면 이것을 우선 사용)
    rap_slot: str = ""      # cgo-445: 랩이 들어갈 자리 none/verse2/bridge/prechorus/intro
    rap_timbre: str = ""    # cgo-445: 랩 음색 묘사 (이름 없음)
    rap_gender: str = ""    # cgo-447: 랩 성별 male/female/duet/choir


@app.post("/vvip_generate")
def vvip_generate(req: VvipReq):
    """
    CGO-FULI VVIP: 사용자 프롬프트 → 가수이름→보컬설명 변환 → Suno API → 곡 생성
    원칙: 최소 2명 이상의 서로 다른 보컬리스트 조합 필수
    """
    if not APIFRAME_KEY:
        return JSONResponse(status_code=500, content={
            "ok": False, "error": "API key가 설정되지 않았습니다."
        })

    prompt = req.prompt
    matched_artists = []  # cgo-393 fix: voice_mix_id 경로에서도 참조되므로 사전 초기화

    # ── cgo-394: 보컬 음색 추출 (VOICE_MIX 또는 VOICE_MAP) ──
    vocal_timbre = ""  # 순수 음색 특징만 (장르 키워드 제거됨)
    # cgo-444: 앱이 보낸 음색 묘사를 그대로 쓴다. 가수 이름이 없으므로 치환도 필요 없고,
    #          퍼센트는 이미 "순서 + 분량"으로 번역돼 있다.
    if (req.timbre or '').strip():
        vocal_timbre = req.timbre.strip()
    elif req.voice_mix_id and req.voice_mix_id in VOICE_MIX:
        # cgo-390: VOICE_MIX 프리셋 경로
        mix = VOICE_MIX[req.voice_mix_id]
        vocal_timbre = mix["prompt"]
    else:
        # cgo-382: VOICE_MAP 개별 아티스트 경로
        matched_artists = []
        seen_descriptions: set = set()
        for artist, description in VOICE_MAP.items():
            if artist in prompt and description not in seen_descriptions:
                matched_artists.append(artist)
                seen_descriptions.add(description)
        if len(seen_descriptions) < 2:  # cgo-403: 최소 2명 (저작권 보호 — 단일 아티스트 음색 복제 방지)
            return JSONResponse(status_code=400, content={
                "ok": False,
                "error": f"보컬리스트를 2명 이상 선택해 주세요. (현재 {len(seen_descriptions)}명)",
                "hint": "저작권 보호를 위해 최소 2명의 보컬 믹스가 필요합니다. 예: 임재범+훌리오이글레시아스",
                "matched": len(seen_descriptions)
            })
        converted_prompt = prompt
        for artist, description in VOICE_MAP.items():
            if artist in converted_prompt:
                converted_prompt = converted_prompt.replace(artist, description)
        vocal_timbre = converted_prompt
        if len(seen_descriptions) == 1:
            vocal_timbre = "solo vocal, " + vocal_timbre

    # ── cgo-395: 성별 태그 강화 + 이성 음색 필터링 ──
    vocal_tag = ""
    if req.vocal == "male":
        vocal_tag = "male vocals only, all male singers, no female vocals, "
    elif req.vocal == "female":
        vocal_tag = "female vocals only, all female singers, no male vocals, "
    elif req.vocal == "duet":
        vocal_tag = "male and female duet, "

    # cgo-395: VOICE_MIX 프리셋에서 이성 음색 설명 제거
    # 예: 남성 선택 시 "sticky deep husky female" 세그먼트 제거
    import re as _re395
    _from_app445 = bool((req.timbre or '').strip())   # cgo-446: 앱이 만든 묘사인가
    if (not _from_app445) and req.vocal in ("male", "female") and vocal_timbre:
        _opp = "female" if req.vocal == "male" else "male"
        _segs = [s.strip() for s in vocal_timbre.split(',') if s.strip()]
        _filtered = [s for s in _segs if not _re395.search(r'\b' + _opp + r'\b', s, _re395.IGNORECASE)]
        if _filtered:
            vocal_timbre = ', '.join(_filtered)
        else:
            # 전부 제거되면 성별 기본 음색
            vocal_timbre = "powerful male vocalist" if req.vocal == "male" else "powerful female vocalist"

    # style 필드: 보컬 음색 간결화 (120자) + 장르 + BPM + key → 4~7 키워드
    timbre_short = vocal_timbre.strip().strip(',').strip()
    # cgo-446: 앱이 보낸 묘사는 순서·분량이 비중을 뜻하므로 자르면 안 된다.
    #          (예전에는 120자로 잘려 성별 접두사만 남고 묘사가 통째로 버려졌다.)
    if (not _from_app445) and len(timbre_short) > 120:
        # 쉼표 기준으로 앞쪽 핵심 음색만 유지
        parts = timbre_short.split(',')
        trimmed = []
        length = 0
        for p in parts:
            p = p.strip()
            if length + len(p) > 120:
                break
            trimmed.append(p)
            length += len(p) + 2
        timbre_short = ', '.join(trimmed) if trimmed else parts[0][:120]

    # cgo-401: 전 장르 보컬 아키타입 경계 허물기
    # Suno가 장르 키워드("trot","hip-hop","EDM" 등)를 보면 해당 장르의 보컬 아키타입을 강제 적용
    # → VOICE_MIX 1000개 + VOICE_MAP 1225개 음색이 무시됨
    # 해법: 보컬 음색이 지정된 경우, 장르 키워드를 음악적 특성 서술로 대체
    # → 장르 느낌(리듬·악기·분위기)은 유지하면서 보컬은 사용자가 선택한 대로 적용
    # → 보컬 미지정 시에는 원래 장르 키워드 유지 (Suno 기본 보컬 OK)
    _GENRE_DESCRIPTORS_401 = {
        'korean trot': 'Korean retro ballad, 2-beat duple rhythm, emotional vibrato, sentimental',
        'trot': 'Korean retro ballad, 2-beat duple rhythm, emotional vibrato, sentimental',
        'hip-hop': 'rhythmic vocal flow, heavy 808 bass, trap hi-hats, boom-bap beat',
        'hip hop': 'rhythmic vocal flow, heavy 808 bass, trap hi-hats, boom-bap beat',
        'rap': 'rhythmic vocal delivery, heavy bass, snappy snare, boom-bap beat',
        'edm dance': 'electronic synth, 4-on-the-floor beat, energetic build-up and drop',
        'edm': 'electronic synth, 4-on-the-floor beat, energetic build-up and drop',
        'rock': 'electric guitar driven, powerful drums, distorted tone, energetic',
        'heavy metal': 'heavy distorted guitar riffs, double bass drum, aggressive power',
        'metal': 'heavy distorted guitar riffs, double bass drum, aggressive power',
        'k-pop': 'Korean modern pop, catchy hook, polished production, dynamic arrangement',
        'kpop': 'Korean modern pop, catchy hook, polished production, dynamic arrangement',
        'korean ballad': 'Korean emotional ballad, piano strings, slow tempo, heartfelt melody',
        'r&b soul': 'smooth groove, neo-soul chords, warm bass, sensual laid-back rhythm',
        'r&b': 'smooth groove, neo-soul chords, warm bass, sensual laid-back rhythm',
        'rnb': 'smooth groove, neo-soul chords, warm bass, sensual laid-back rhythm',
        'soul': 'soulful groove, gospel harmony, warm organic feel',
        'jazz': 'jazz swing feel, walking bass, brushed drums, extended chord voicings',
        'blues': 'blues shuffle rhythm, 12-bar progression, warm guitar, soulful bends',
        'country': 'acoustic guitar, steel guitar, Nashville production, storytelling melody',
        'reggae': 'offbeat skank guitar, deep bass, laid-back groove, tropical feel',
        'latin': 'Latin percussion, clave rhythm, warm brass, passionate melody',
        'classical': 'orchestral strings, dynamic expression, refined melody, concert hall',
        'ambient': 'atmospheric pads, ethereal texture, slow evolving soundscape, dreamy',
        'folk acoustic': 'acoustic guitar, gentle fingerpicking, warm organic storytelling',
        'folk': 'acoustic guitar, gentle fingerpicking, warm organic storytelling',
        'pop ballad': 'emotional pop ballad, piano-driven, soaring melody, heartfelt',
        'pop': 'catchy pop melody, polished production, upbeat arrangement',
    }
    import re as _re401
    _style_for_suno = req.style
    _genre_replaced = False
    if timbre_short:
        # 보컬 음색이 있을 때만 장르 키워드를 음악적 서술로 대체
        # 긴 키워드부터 매칭 (예: "korean trot"이 "trot"보다 먼저)
        for _gk in sorted(_GENRE_DESCRIPTORS_401.keys(), key=len, reverse=True):
            if _gk in _style_for_suno.lower():
                _pattern = _re401.compile(_re401.escape(_gk), _re401.IGNORECASE)
                _style_for_suno = _pattern.sub(_GENRE_DESCRIPTORS_401[_gk], _style_for_suno, count=1)
                _genre_replaced = True
                break
    # cgo-447: 장르 서술로 바꾸면 "emotional emotional pop ballad"처럼 같은 낱말이
    #          연달아 겹칠 수 있다. 붙어 있는 중복 낱말만 하나로 줄인다.
    _style_for_suno = _re401.sub(r'\b(\w+)(\s+\1\b)+', r'\1', _style_for_suno, flags=_re401.IGNORECASE)
    # cgo-446: 앱 묘사에는 이미 성별 문구가 들어 있다. 서버가 또 붙이면
    #          "male vocals only … female vocals only" 처럼 모순된 지시가 되어
    #          여성을 골라도 남성 목소리가 나왔다. 앱 묘사가 있으면 그것만 쓴다.
    if _from_app445:
        vocal_tag = ""
    if timbre_short:
        suno_style = f"{vocal_tag}{timbre_short}, {_style_for_suno}, {req.bpm} BPM, key of {req.key}"
    else:
        suno_style = f"{vocal_tag}{req.style}, {req.bpm} BPM, key of {req.key}"
    _before478 = len(suno_style)
    suno_style = _compact_style(suno_style)                 # cgo-478
    if _before478 != len(suno_style):
        print(f"[vvip] 스타일 압축 {_before478}자 → {len(suno_style)}자", flush=True)

    # ═══ cgo-445: 랩 구간 삽입 ═══
    # 래퍼 음색만 지정해서는 Suno가 랩을 넣지 않는다. 가사에 [Rap] 구조 태그가 있어야 한다.
    _RAP_TAG = {
        'verse2':    ('[Verse 2]', '[Rap Verse]'),          # 2절을 통째로 랩으로
        'bridge':    ('[Bridge]', '[Rap Break]'),           # 브릿지를 랩으로 — 발라드에 가장 잘 맞는다
        'prechorus': ('[Pre-Chorus]', '[Rap]'),             # 후렴 직전 짧게
        'intro':     ('[Intro]', '[Rap Intro]'),            # 곡 머리에서 치고 들어옴
    }

    def _apply_rap(lyr: str, slot: str) -> str:
        """가사에 랩 구조 태그를 심는다. 자리가 없으면 적절한 위치에 새로 만든다."""
        if not slot or slot == 'none' or not lyr:
            return lyr
        pair = _RAP_TAG.get(slot)
        if not pair:
            return lyr
        want, tag = pair
        if tag in lyr:                      # 이미 있음
            return lyr
        if want in lyr:                     # 기존 구간을 랩으로 바꾼다
            return lyr.replace(want, tag, 1)
        lines = lyr.split('\n')
        if slot == 'intro':
            return tag + '\n' + lyr
        # 마지막 [Chorus] 앞에 끼워 넣는다 (브릿지·프리코러스·2절 대체 모두 여기로)
        idx = [i for i, l in enumerate(lines) if l.strip().startswith('[Chorus')]
        at = idx[-1] if idx else len(lines)
        return '\n'.join(lines[:at] + [tag, ''] + lines[at:])

    _rap_on = bool((req.rap_slot or '').strip()) and req.rap_slot != 'none'
    if _rap_on:
        _rt = (req.rap_timbre or '').strip()
        _rg = (req.rap_gender or '').strip().lower()

        # cgo-447: 메인 보컬이 여성인데 랩이 남성이면(또는 그 반대),
        # 앞쪽 "no male vocals / no female vocals"가 래퍼를 원천 금지해 버린다.
        # 그래서 배제 문구를 '노래하는 부분에 한해서'로 범위를 좁힌다.
        _lead = ('f' if suno_style.startswith('female vocals only')
                 else 'm' if suno_style.startswith('male vocals only') else '')
        _rapg = 'f' if _rg == 'female' else 'm' if _rg == 'male' else ''
        if _lead and _rapg and _lead != _rapg:
            suno_style = suno_style.replace(
                'female vocals only, all female singers, no male vocals, ',
                'female lead vocals on every sung line, all sung parts by a female singer, ', 1)
            suno_style = suno_style.replace(
                'male vocals only, all male singers, no female vocals, ',
                'male lead vocals on every sung line, all sung parts by a male singer, ', 1)

        suno_style = suno_style.rstrip().rstrip(',') + ', with a featured rap section'
        if _rt:
            suno_style += ' performed by ' + _rt
        if _lead and _rapg and _lead != _rapg:
            _lw = 'female' if _lead == 'f' else 'male'
            _rw = 'female' if _rapg == 'f' else 'male'
            suno_style += (f'; the rap section only is rapped by a {_rw} voice'
                           f', every other section is sung by the {_lw} lead')

    # lyrics 필드: 가사 + [Verse]/[Chorus] 메타태그 삽입
    has_lyrics = bool(req.lyrics and req.lyrics.strip())
    suno_lyrics = ""
    if has_lyrics:
        raw_lyrics = req.lyrics.strip()
        # cgo-413: 주제/설명 vs 실제 가사 자동 판별
        # 3줄 이하 + 100자 미만 + 메타태그 없음 → 주제 설명으로 간주
        _lyric_lines_chk = [l for l in raw_lyrics.split('\n') if l.strip()]
        _has_meta_chk = any(tag in raw_lyrics for tag in ['[Verse', '[Chorus', '[Bridge', '[Intro', '[Outro', '[Hook'])
        if not _has_meta_chk and len(_lyric_lines_chk) <= 2 and len(raw_lyrics) < 100:
            # 짧은 설명문 → 가사가 아닌 주제/분위기로 취급
            has_lyrics = False
            suno_style = suno_style.rstrip() + ", " + raw_lyrics
    if has_lyrics:
        # 이미 메타태그가 있으면 그대로, 없으면 기본 구조 삽입
        has_metatags = any(tag in raw_lyrics for tag in ['[Verse', '[Chorus', '[Bridge', '[Intro', '[Outro', '[Hook'])
        if has_metatags:
            suno_lyrics = raw_lyrics
        else:
            # 줄바꿈 기준으로 Verse/Chorus 자동 태깅
            lyric_lines = [l for l in raw_lyrics.split('\n') if l.strip()]
            if len(lyric_lines) <= 4:
                suno_lyrics = "[Verse]\n" + raw_lyrics
            elif len(lyric_lines) <= 8:
                mid = len(lyric_lines) // 2
                suno_lyrics = "[Verse]\n" + '\n'.join(lyric_lines[:mid]) + "\n\n[Chorus]\n" + '\n'.join(lyric_lines[mid:])
            else:
                third = len(lyric_lines) // 3
                suno_lyrics = "[Verse 1]\n" + '\n'.join(lyric_lines[:third])
                suno_lyrics += "\n\n[Chorus]\n" + '\n'.join(lyric_lines[third:third*2])
                suno_lyrics += "\n\n[Verse 2]\n" + '\n'.join(lyric_lines[third*2:])
        if _rap_on:
            suno_lyrics = _apply_rap(suno_lyrics, req.rap_slot)

    # ── apiframe.ai v2 API 호출 (비동기: job_id만 즉시 반환) ──
    # cgo-394: 코드진행은 Suno가 파싱 불가 → 프롬프트에서 제외
    api_body = {
        "prompt": suno_lyrics if has_lyrics else suno_style,
        "model": "suno",
        "sunoParams": {
            "custom_mode": has_lyrics,
            "instrumental": False,
            "model_version": "V4_5PLUS"
        }
    }
    if has_lyrics:
        api_body["sunoParams"]["style"] = suno_style

    try:
        resp = http_requests.post(
            'https://api.apiframe.ai/v2/music/generate',
            headers={
                'X-API-Key': APIFRAME_KEY,
                'Content-Type': 'application/json'
            },
            json=api_body,
            timeout=30
        )

        if not resp.ok:
            api_err = ""
            try:
                api_err = resp.text[:300]
            except Exception:
                pass
            return JSONResponse(status_code=502, content={
                "ok": False,
                "error": f"Suno API 호출 실패 (HTTP {resp.status_code})",
                "hint": api_err or "apiframe.ai 서비스 상태를 확인하세요"
            })

        result = resp.json()
        job_id = result.get('id') or result.get('jobId') or result.get('task_id')

        if not job_id:
            return JSONResponse(status_code=502, content={
                "ok": False,
                "error": "API에서 job ID를 받지 못했습니다.",
                "hint": str(result)[:300]
            })

        # 즉시 반환 — 클라이언트가 /vvip_status/{job_id}로 폴링
        return JSONResponse(content={
            "ok": True,
            "job_id": job_id,
            "status": "PROCESSING",
            "matched_artists": matched_artists
        })

    except http_requests.exceptions.RequestException as e:
        return JSONResponse(status_code=502, content={
            "ok": False,
            "error": f"API 연결 실패: {str(e)}"
        })


@app.get("/vvip_status/{job_id}")
def vvip_status(job_id: str):
    """비동기 VVIP 곡 생성 상태 조회 — 클라이언트가 5초마다 폴링"""
    if not APIFRAME_KEY:
        return JSONResponse(status_code=500, content={
            "ok": False, "error": "API 키 미설정"
        })
    try:
        status_resp = http_requests.get(
            f'https://api.apiframe.ai/v2/jobs/{job_id}',
            headers={'X-API-Key': APIFRAME_KEY},
            timeout=15
        )
        if not status_resp.ok:
            return JSONResponse(status_code=502, content={
                "ok": False,
                "error": f"상태 조회 실패 (HTTP {status_resp.status_code})"
            })
        status_data = status_resp.json()
    except http_requests.exceptions.RequestException as e:
        return JSONResponse(status_code=502, content={
            "ok": False,
            "error": f"상태 조회 연결 실패: {str(e)}"
        })

    job_status = (status_data.get('status') or '').upper()

    if job_status == 'COMPLETED':
        res = status_data.get('result', status_data)
        tracks = res.get('tracks', [])
        audio_url = None
        title = 'VVIP Song'
        if tracks:
            audio_url = tracks[0].get('audioUrl') or tracks[0].get('audio_url')
            title = tracks[0].get('title', title)
        if not audio_url:
            return JSONResponse(status_code=502, content={
                "ok": False,
                "error": "완료되었으나 오디오 URL을 찾을 수 없습니다.",
                "hint": str(res)[:300]
            })
        return JSONResponse(content={
            "ok": True,
            "status": "COMPLETED",
            "audio_url": audio_url,
            "title": title,
            "job_id": job_id
        })
    elif job_status == 'FAILED':
        return JSONResponse(content={
            "ok": False,
            "status": "FAILED",
            "error": "Suno 곡 생성이 실패했습니다.",
            "hint": str(status_data.get('error', '다시 시도해 주세요'))[:200]
        })
    else:
        return JSONResponse(content={
            "ok": True,
            "status": "PROCESSING",
            "job_id": job_id
        })


# ── cgo-374: 보컬 멜로디 추출 (Spotify Basic Pitch AI) ──────────────────
_NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def _midi_to_note_name(midi_num):
    """MIDI 번호 → 음표명 (60 → 'C4')"""
    octave = (midi_num // 12) - 1
    note = _NOTE_NAMES_SHARP[midi_num % 12]
    return f"{note}{octave}"

def _extract_melody_impl(audio_path, bpm=120, max_seconds=300):
    """오디오 파일에서 보컬 멜로디 추출 — Spotify Basic Pitch AI 엔진

    librosa pyin 대비 장점:
    - AI 학습 모델로 정확도 대폭 향상
    - 다성음(polyphonic) 감지 가능
    - Onset(음 시작점) 정밀 감지
    - melodia_trick으로 주선율(보컬) 자동 추출
    """
    from basic_pitch.inference import predict
    import librosa

    # Spotify Basic Pitch AI 채보
    model_output, midi_data, note_events = predict(
        audio_path,
        onset_threshold=0.5,
        frame_threshold=0.3,
        minimum_note_length=80,                       # 80ms 미만 노이즈 제거
        minimum_frequency=librosa.note_to_hz('B2'),    # cgo-411: C4→B2 보컬 하한 확장 (남성 저음 보컬 잘림 방지, 베이스 악기는 melodia_trick이 필터)
        maximum_frequency=librosa.note_to_hz('C6'),   # 보컬 상한
        melodia_trick=True,                           # 주선율(보컬) 추출 강화
    )

    if not note_events:
        return []

    # note_events: [(start_sec, end_sec, midi_pitch, amplitude, pitch_bends), ...]
    # max_seconds 제한 + raw_events 변환
    raw_events = []
    for evt in note_events:
        start_sec = evt[0]
        end_sec = evt[1]
        midi_pitch = int(evt[2])
        amplitude = float(evt[3])

        if start_sec >= max_seconds:
            break

        dur_sec = min(end_sec, max_seconds) - start_sec
        if dur_sec < 0.05:
            continue

        raw_events.append({
            'midi': midi_pitch,
            'start': start_sec,
            'dur': dur_sec,
            'amp': amplitude
        })

    # 동시 발음 시 가장 강한 음만 남기기 (단선율 멜로디)
    raw_events.sort(key=lambda e: (e['start'], -e['amp']))
    melody_line = []
    last_end = 0.0
    for evt in raw_events:
        if evt['start'] >= last_end - 0.02:  # 20ms 허용치
            melody_line.append(evt)
            last_end = evt['start'] + evt['dur']

    # 16분음표 그리드 양자화 + 순차 이벤트 빌드
    beat_dur = 60.0 / bpm
    melody_events = []
    prev_end_beats = 0.0
    prev_end_sec = 0.0

    for evt in melody_line:
        start_beats = evt['start'] / beat_dur
        dur_beats = evt['dur'] / beat_dur

        # 16분음표(0.25) 양자화
        start_beats = round(start_beats * 4) / 4
        dur_beats = round(dur_beats * 4) / 4
        dur_beats = max(0.25, min(dur_beats, 4.0))

        # 이전 노트와 현재 사이 쉼표 삽입
        gap = round(start_beats - prev_end_beats, 2)
        if gap >= 0.25:
            gap_sec = round(gap * beat_dur, 3)
            melody_events.append({'name': None, 'dur': gap, 'start_sec': round(prev_end_sec, 3)})
            prev_end_sec += gap_sec

        note_name = _midi_to_note_name(evt['midi'])
        melody_events.append({
            'name': note_name,
            'dur': round(dur_beats, 2),
            'start_sec': round(evt['start'], 3)
        })
        prev_end_beats = start_beats + dur_beats
        prev_end_sec = evt['start'] + evt['dur']

    return melody_events


@app.post("/extract_melody")
async def extract_melody_endpoint(request: Request):
    """cgo-375: 보컬 멜로디 추출 — Spotify Basic Pitch AI (C4~C6 보컬 전용, 악기음 제거)"""
    data = await request.json()
    audio_url = data.get("audio_url")
    bpm = float(data.get("bpm", 120))

    if not audio_url:
        return JSONResponse(status_code=400, content={"ok": False, "error": "audio_url 필수"})

    try:
        # 오디오 다운로드
        resp = http_requests.get(audio_url, timeout=30)
        if resp.status_code != 200:
            return JSONResponse(status_code=502, content={
                "ok": False, "error": f"오디오 다운로드 실패 (HTTP {resp.status_code})"
            })

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(resp.content)
            temp_path = f.name

        try:
            notes = _extract_melody_impl(temp_path, bpm=bpm)
            note_count = len([n for n in notes if n.get('name')])
            return JSONResponse(content={
                "ok": True,
                "notes": notes,
                "total_notes": note_count,
                "bpm": bpm
            })
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "ok": False, "error": f"멜로디 추출 실패: {str(e)}"
        })


@app.get("/vvip_generate")
def vvip_info():
    """VVIP 엔드포인트 상태 확인"""
    return JSONResponse(content={
        "ok": True,
        "endpoint": "vvip_generate",
        "artists": len(set(VOICE_MAP.values())),
        "api_configured": bool(APIFRAME_KEY),
        "rule": "minimum 2 different vocal combinations required"
    })


# ═══ cgo-422: AI 가사 생성 (Udio API + 로컬 폴백 생성기) ═══
import random as _lyrics_random
import hashlib as _lyrics_hash

def _generate_korean_lyrics(topic: str, style: str) -> str:
    """서버 로컬 한국어 가사 생성기 — Udio 실패 시 폴백.
    cgo-443: 주제 글(topic)은 '분위기 판별'에만 쓰고 가사 줄에는 절대 넣지 않는다.
    (이전에는 f'{topic}... 안녕' 처럼 사용자가 쓴 문장이 그대로 가사에 박혔다.)
    어미를 무작위로 이어 붙이던 방식도 폐기 — 완성된 문장만 쓴다."""
    seed = int(_lyrics_hash.md5(f"{topic}{style}{time.time():.0f}".encode()).hexdigest()[:8], 16)
    rng = _lyrics_random.Random(seed)
    kw = (topic or '').lower()
    sad_kw = ['이별','슬픈','눈물','그리움','아픔','외로','헤어','비','울','잊','떠나','보내','미련','상처']
    love_kw = ['사랑','설렘','첫사랑','두근','키스','고백','달콤','행복','그대','함께','연인','곁']
    night_kw = ['밤','별','달','새벽','어둠','불빛','노을','하늘']
    hope_kw = ['희망','꿈','미래','빛','시작','용기','날개','비상','내일']
    is_sad = any(w in kw for w in sad_kw)
    is_love = any(w in kw for w in love_kw)
    is_night = any(w in kw for w in night_kw)
    is_hope = any(w in kw for w in hope_kw)
    if is_sad:
        mood = 'sad'
    elif is_hope and not is_love:
        mood = 'hope'
    elif is_night and not is_love:
        mood = 'night'
    else:
        mood = 'love'

    V = {
    'sad': ['텅 빈 거리에 네 흔적만 남아 있어','지워지지 않는 기억 속을 헤매고 있어',
        '너의 빈자리가 이렇게 큰 줄 몰랐어','흐린 창밖으로 빗물이 흘러내려',
        '돌아올 수 없는 그 시간이 아프잖아','입술 끝에 네 이름이 맴돌아',
        '아직도 네가 곁에 있는 것만 같아','바람이 불면 너의 향기가 스쳐가',
        '매일 밤 그리움이 나를 찾아와','사랑이 이별이 되던 그 순간에',
        '우리가 걸었던 그 길이 흐릿해져','시간이 흘러도 이 아픔은 그대로야',
        '네가 남긴 말들이 아직 귓가에 맴돌아','혼자 남은 방 안에 적막만 가득해'],
    'love': ['네가 웃을 때 세상이 환해지는 것 같아','달콤한 이 순간이 꿈처럼 흘러가',
        '심장이 두근두근 멈추질 않아','너의 눈빛 속에 내가 담겨 있잖아',
        '매일이 특별해지는 건 네 덕분이야','손끝이 닿는 순간 전부 다 멈춰',
        '이런 감정은 처음이라 어쩔 줄 몰라','네 목소리만 들어도 하루가 완성돼',
        '꿈인지 현실인지 모를 만큼 행복해','세상 누구보다 네가 빛나 보여',
        '너와 걷는 이 길이 더없이 따뜻해','아무 말 없이 곁에 있어도 좋아'],
    'night': ['별빛이 쏟아지는 이 밤에','어둠 속에서 너의 미소가 빛나',
        '달이 우리를 비추는 이 순간','새벽 공기가 조용히 스며들어',
        '창밖으로 달빛이 내려앉아','고요한 밤하늘에 소원을 빌어',
        '도시의 불빛 아래 우리 둘만의 시간','노을이 지는 하늘을 함께 바라봐'],
    'hope': ['어둠이 지나면 반드시 빛이 와','내일을 향해 두 팔을 펼쳐',
        '넘어져도 다시 일어나는 거야','내 안의 날개를 활짝 펴는 순간',
        '두려움 너머에 기다리는 내일이 있어','포기하지 않을 거야 끝까지',
        '한 걸음씩 천천히 나아가고 있어','멈추지 않으면 언젠가 닿을 거야']}
    C = {
    'sad': ['보고 싶어 보고 싶어 또 보고 싶어','잊으려 해도 자꾸만 떠올라',
            '이 밤이 지나면 괜찮아질까','눈물이 마를 때까지 기다릴게',
            '너 없는 세상은 너무 차가워','다시 한번만 네 곁에 있고 싶어',
            '사랑했잖아 우린 정말 사랑했잖아'],
    'love': ['너를 사랑해 오늘도 내일도 영원히','이 세상 끝까지 함께 걸어갈래',
             '너라는 기적이 내게 찾아온 거야','매 순간 네가 있어 난 완벽해',
             '사랑한다 말할게 수백 번이라도','네 곁이 나의 전부야'],
    'night': ['이 밤이 우리를 감싸 안아줘','별빛 아래 너와 나 단둘이',
              '깊어가는 밤에 너를 생각해','달빛이 우리 둘을 비춰줘'],
    'hope': ['날아올라 더 높이 더 멀리','우리의 꿈은 멈추지 않아',
             '빛나는 내일을 향해 달려가자','할 수 있어 난 할 수 있어 믿어봐']}
    B = {
    'sad': ['시간아 제발 멈춰줘','한 번만 더 안아줄 수 있다면','이게 마지막이라 해도','우리의 계절은 끝나지 않아'],
    'love': ['매일 밤 꿈속에서도 너를 만나','세상이 뭐라 해도 난 너야','이 노래가 끝나도 우린 영원해','약속할게 영원히 네 곁에'],
    'night': ['이 밤이 끝나지 않았으면 해','새벽이 와도 여기 있어줘','별이 지기 전에 네게 닿고 싶어'],
    'hope': ['지금 이 순간이 시작이야','함께라면 못할 게 없어','내일은 오늘보다 더 빛날 거야','눈을 감고 느껴봐 우리의 미래를']}
    O = {
    'sad': ['그래도 사랑했었다','이 노래가 끝나면 놓아줄게','안녕 나의 계절아'],
    'love': ['사랑해 영원히','너와 함께라면 어디든','우리의 이야기는 계속돼'],
    'night': ['이 밤을 기억할게','별빛처럼 오래 남아줘','조용히 눈을 감아'],
    'hope': ['우리는 할 수 있어','빛나는 내일을 향해','이건 끝이 아닌 시작이야']}

    pool = list(V[mood]); rng.shuffle(pool)
    v1, v2 = pool[:4], pool[4:8]
    ch = rng.sample(C[mood], min(4, len(C[mood])))
    br = rng.sample(B[mood], min(2, len(B[mood])))
    out = ['[Verse 1]'] + v1 + ['', '[Chorus]'] + ch + ['', '[Verse 2]'] + v2
    out += ['', '[Chorus]'] + ch + ['', '[Bridge]'] + br + ['', '[Outro]', rng.choice(O[mood])]
    return '\n'.join(out)

# ═══ cgo-439: AI 가사 생성 안정화 ═══
# 기존 문제: Udio 가사 API는 비동기 job → job_id만 클라이언트로 넘기고 클라이언트가 폴링.
#   · job FAILED / 상태조회 HTTP 오류 / 완료됐지만 가사 필드 위치가 다름 → 로컬 폴백 없이 오류·시간초과로 끝남
# 수정: 서버가 직접 job을 끝까지 폴링(최대 ~40초) → 가사 필드를 재귀 탐색 → 어떤 실패든 로컬 생성기로 폴백.
#   → 클라이언트는 항상 {"ok":true,"lyrics":...} 를 받음 (가사가 반드시 뜸)
_LYRICS_KEYS = ('lyrics', 'lyric', 'text', 'content')

def _extract_lyrics(obj, depth=0):
    """apiframe 응답(JSON) 어디에 있든 가사 문자열을 찾아 반환"""
    if depth > 6 or obj is None:
        return ''
    if isinstance(obj, str):
        return ''
    if isinstance(obj, list):
        for it in obj:
            v = _extract_lyrics(it, depth + 1)
            if v:
                return v
        return ''
    if isinstance(obj, dict):
        for k in _LYRICS_KEYS:
            v = obj.get(k)
            if isinstance(v, str) and len(v.strip()) > 20:
                return v.strip()
            if isinstance(v, (list, dict)):
                vv = _extract_lyrics(v, depth + 1)
                if vv:
                    return vv
        for k in ('result', 'data', 'output', 'results', 'tracks', 'items'):
            if k in obj:
                v = _extract_lyrics(obj[k], depth + 1)
                if v:
                    return v
    return ''

def _job_id_of(obj):
    if not isinstance(obj, dict):
        return None
    for k in ('id', 'jobId', 'job_id', 'task_id', 'taskId'):
        if obj.get(k):
            return str(obj[k])
    d = obj.get('data') or obj.get('result')
    if isinstance(d, dict):
        for k in ('id', 'jobId', 'job_id', 'task_id', 'taskId'):
            if d.get(k):
                return str(d[k])
    return None

def _poll_apiframe_job(job_id: str, max_wait: float = 40.0, interval: float = 2.5):
    """apiframe job 상태를 서버에서 직접 폴링. 반환: (lyrics or '', 오류설명)"""
    deadline = time.time() + max_wait
    last = ''
    while time.time() < deadline:
        try:
            r = http_requests.get(
                f'https://api.apiframe.ai/v2/jobs/{job_id}',
                headers={'X-API-Key': APIFRAME_KEY},
                timeout=15
            )
            if not r.ok:
                last = f"job 조회 HTTP {r.status_code}: {r.text[:150]}"
                if r.status_code in (401, 403, 404):
                    return '', last
            else:
                d = r.json()
                st = str(d.get('status') or '').upper()
                if st in ('COMPLETED', 'SUCCEEDED', 'SUCCESS', 'DONE', 'FINISHED'):
                    lyr = _extract_lyrics(d)
                    return (lyr, '') if lyr else ('', f"완료됐지만 가사 없음: {str(d)[:200]}")
                if st in ('FAILED', 'ERROR', 'CANCELLED', 'CANCELED'):
                    return '', f"job 실패: {str(d.get('error') or d)[:200]}"
                lyr = _extract_lyrics(d.get('result')) if isinstance(d, dict) else ''
                if lyr:
                    return lyr, ''
                last = f"status={st or '?'}"
        except Exception as e:
            last = f"job 조회 예외: {e}"
        time.sleep(interval)
    return '', f"시간 초과 ({last})"

# cgo-469: 앱의 '노래 언어' 15가지 — 그 나라 글자 이름을 함께 줘야 AI가 덜 헷갈린다.
_LANG_NATIVE = {
    'Korean': '한국어', 'English': 'English', 'Japanese': '日本語', 'Chinese': '中文',
    'Spanish': 'Español', 'French': 'Français', 'German': 'Deutsch', 'Italian': 'Italiano',
    'Portuguese': 'Português', 'Russian': 'Русский', 'Arabic': 'العربية', 'Hindi': 'हिन्दी',
    'Indonesian': 'Bahasa Indonesia', 'Thai': 'ภาษาไทย', 'Vietnamese': 'Tiếng Việt',
}


def _is_korean_lang(lang) -> bool:
    return (lang or 'Korean').strip().lower().startswith('korea')


def _generate_fallback_lyrics(topic: str, style: str, lang: str) -> str:
    """AI 가사가 실패했을 때 쓰는 비상 가사.
    cgo-469: 예전에는 어떤 언어를 골랐든 한국어 가사가 나왔다. 영어 노래를 주문했는데
    한국어 가사가 오는 것은 안 쓰느니만 못하다. 한국어가 아니면 영어로 돌려준다.
    주제 글은 분위기 판별에만 쓰고 가사 줄에는 넣지 않는다(cgo-443과 같은 원칙)."""
    if _is_korean_lang(lang):
        return _generate_korean_lyrics(topic, style)
    kw = (topic or '').lower()
    sad = any(w in kw for w in ('sad', 'tear', 'lonely', 'rain', 'goodbye', 'miss', 'break',
                                '이별', '슬픈', '눈물', '그리움', '외로'))
    hope = any(w in kw for w in ('hope', 'dream', 'light', 'tomorrow', 'rise', 'begin',
                                 '희망', '꿈', '빛', '시작'))
    if sad:
        v1 = ["The room still keeps the shape of you", "and every quiet hour knows your name.",
              "I learned the weight of empty chairs,", "I learned that nothing stays the same."]
        ch = ["So let the evening take me slow,", "let the streetlights blur the rest.",
              "If I can't hold you anymore,", "I'll hold the way you left."]
        br = ["And maybe time is not a wound,", "maybe time is only wide."]
    elif hope:
        v1 = ["Morning breaks against the window,", "and the dark gives up its hold.",
              "Everything I thought was ending", "turns out to be the road."]
        ch = ["So I'm walking into daylight,", "with my whole heart open wide.",
              "Every step I thought would break me", "built the ground beneath my stride."]
        br = ["I was never really falling,", "I was learning how to fly."]
    else:
        v1 = ["There's a song inside the quiet,", "something only we can hear.",
              "In the space between the heartbeats,", "that's the place I find you near."]
        ch = ["And we'll carry it together,", "through the noise and through the night.",
              "Every ordinary moment", "turning gold against the light."]
        br = ["Nothing lasts, and that's the beauty,", "that's the reason we hold tight."]
    return ("[Verse]\n" + "\n".join(v1)
            + "\n\n[Chorus]\n" + "\n".join(ch)
            + "\n\n[Verse]\n" + "\n".join(reversed(v1))
            + "\n\n[Bridge]\n" + "\n".join(br)
            + "\n\n[Chorus]\n" + "\n".join(ch))


def _local_lyrics_response(topic, style, api_error='', lang='Korean'):
    try:
        return JSONResponse(content={"ok": True,
                                     "lyrics": _generate_fallback_lyrics(topic, style, lang),
                                     "source": "local", "api_error": api_error, "lang": lang,
                                     "reason": _why(api_error) if api_error else ''})
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": f"가사 생성 실패: {str(e)}", "api_error": api_error})

# cgo-443: 혼잡("No available capacity") 같은 일시적 실패는 재시도한다.
_RETRYABLE = ('no available capacity','please retry','rate limit','too many','temporarily',
              'overload','timeout','timed out','503','502','504','429','시간 초과')

def _is_retryable(msg: str) -> bool:
    m = (msg or '').lower()
    return any(k in m for k in _RETRYABLE)

def _why(api_error: str) -> str:
    """사용자에게 보여줄 한 줄 사유"""
    m = (api_error or '').lower()
    if 'no available capacity' in m or 'please retry' in m:
        return 'AI 가사 서버가 혼잡합니다 — 잠시 후 다시 눌러 주세요'
    if '401' in m or 'unauthorized' in m:
        return 'AI 가사 서버 인증 오류 — API 키를 확인해 주세요'
    if '402' in m or 'credit' in m or 'insufficient' in m:
        return 'AI 가사 크레딧이 부족합니다'
    if 'rate limit' in m or '429' in m:
        return '요청이 많아 잠시 제한되었습니다 — 잠시 후 다시 시도해 주세요'
    if '시간 초과' in (api_error or '') or 'timeout' in m:
        return 'AI 가사 생성이 오래 걸려 기본 가사로 만들었습니다'
    # cgo-476: 아는 경우가 아니면 '연결하지 못했습니다'로 뭉개지 말고
    # 실제로 돌아온 말을 그대로 보여준다. 로그를 뒤지지 않고도 원인을 안다.
    raw = (api_error or '').strip()
    if raw:
        return 'AI 가사 실패 → ' + raw[:160]
    return 'AI 가사 서버에 연결하지 못했습니다'

# ── cgo-470: 가사를 Suno에게 받는다 ────────────────────────────────
# 그동안 가사는 Udio(다른 회사 음악엔진)에서 받고 노래는 Suno가 불렀다.
# 가사는 '글'인데 Udio의 작곡 대기열 뒤에 줄을 서느라 "no available capacity"가
# 자주 떴다. 노래를 부를 Suno에게 가사도 맡기면 대기열이 갈라지고, 가사와 곡이
# 같은 집에서 나와 결도 맞는다. Udio는 예비로만 남긴다.
_SUNO_LYRICS_URL = 'https://api.apiframe.pro/suno-lyrics'
_SUNO_FETCH_URL = 'https://api.apiframe.pro/fetch'


def _hdr(style: str) -> dict:
    """apiframe은 창구(.pro/.ai)마다 인증 헤더 이름이 다르다. 둘 다 시도해 본다."""
    key = 'Authorization' if style == 'auth' else 'X-API-Key'
    return {key: APIFRAME_KEY, 'Content-Type': 'application/json'}


def _suno_fetch(task_id: str, style: str, max_wait: float = 38.0, interval: float = 2.5):
    """suno-lyrics 작업이 끝날 때까지 기다린다 → (가사, 오류설명)"""
    deadline = time.time() + max_wait
    last = ''
    while time.time() < deadline:
        time.sleep(interval)
        try:
            r = http_requests.post(_SUNO_FETCH_URL, headers=_hdr(style),
                                   json={'task_id': task_id}, timeout=15)
            if r.ok:
                d = r.json()
                lyr = _extract_lyrics(d)
                if lyr:
                    return lyr, ''
                st = str(d.get('status') or '').lower()
                if st in ('failed', 'error'):
                    return '', f"가사 생성 실패: {str(d)[:200]}"
                last = f"진행 중({st or '상태 미표시'})"
            else:
                last = f"fetch HTTP {r.status_code}: {r.text[:150]}"
                if r.status_code in (401, 403, 404):
                    return '', last
        except Exception as e:
            last = f"fetch 예외: {e}"
    return '', last or '시간 초과'


def _suno_lyrics_once(prompt_text: str):
    """Suno 가사 API 1회 시도 → (가사, 오류설명)"""
    last = ''
    for style in ('auth', 'xkey'):
        try:
            resp = http_requests.post(_SUNO_LYRICS_URL, headers=_hdr(style),
                                      json={'prompt': prompt_text}, timeout=30)
        except Exception as e:
            last = f"연결 예외: {e}"
            continue
        if resp.status_code in (401, 403):
            last = f"인증 거부(HTTP {resp.status_code})"
            continue                      # 헤더 이름을 바꿔 한 번 더
        if not resp.ok:
            return '', f"HTTP {resp.status_code}: {resp.text[:200]}"
        try:
            result = resp.json()
        except Exception:
            return '', f"응답이 JSON이 아님: {resp.text[:150]}"
        lyr = _extract_lyrics(result)
        if lyr:
            return lyr, ''
        task_id = _job_id_of(result)
        if not task_id:
            return '', f"task_id 없음: {str(result)[:200]}"
        return _suno_fetch(task_id, style)
    return '', last or 'suno-lyrics 호출 실패'


# ── cgo-473: 가사는 '글 쓰는 AI'에게 받는다 ─────────────────────────
# 가사는 글인데 그동안 음악 만드는 기계(Udio)의 GPU 대기열 뒤에 줄을 섰다.
# 그래서 "No available capacity"가 상습적으로 떴고, 될 때까지 몇 번씩 눌러야 했다.
# 글 쓰는 AI는 2~4초면 답하고, 20개 언어를 제대로 쓰고, 한 곡에 1원이 안 든다.
#
# 특정 회사에 묶이지 않게 만들었다. 레일웨이 Variables에 아래 셋만 넣으면 된다:
#   CGO_LLM_KEY   : 그 회사에서 받은 열쇠            (이것만 넣으면 켜진다)
#   CGO_LLM_URL   : 주소 (안 넣으면 Gemini 기본값)
#   CGO_LLM_MODEL : 모델 이름 (안 넣으면 기본값)
# OpenAI·Gemini·Groq·DeepSeek 등 'chat/completions' 방식이면 그대로 통한다.
CGO_LLM_KEY = os.environ.get('CGO_LLM_KEY', '').strip()
CGO_LLM_URL = os.environ.get(
    'CGO_LLM_URL',
    'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions').strip()
CGO_LLM_MODEL = os.environ.get('CGO_LLM_MODEL', 'gemini-3.8-flash').strip()
# cgo-479: 가장 최신 모델은 전 세계가 몰려서 503(붐빔)이 자주 난다.
# 붐비면 옆 모델로 갈아탄다. 가사는 가벼운 모델로도 충분히 잘 쓴다.
# CGO_LLM_MODELS 에 쉼표로 적어 두면 그 순서를 따른다.
CGO_LLM_MODELS = [x.strip() for x in os.environ.get(
    'CGO_LLM_MODELS',
    f'{CGO_LLM_MODEL},gemini-3.7-flash,gemini-3.5-flash-lite,gemini-3.1-flash-lite'
).split(',') if x.strip()]
_MODEL_REST = {}        # 붐비는 모델은 잠시 쉬게 둔다 {모델: 언제까지}


def _clean_lyrics(txt: str) -> str:
    """cgo-474: AI가 '생각한 과정'을 가사에 섞어 보내는 것을 걷어낸다.
    실제로 이런 것이 가사 칸에 그대로 들어왔다:
        tags only? [Verse], [Chorus], [Bridge]. Check.
        *   Chorus repeats? Yes, exactly the same.
    주문을 아무리 다듬어도 모델은 가끔 이런다. 받는 쪽에서도 막는다."""
    import re as _re
    t = (txt or '').strip()
    t = _re.sub(r'^```[a-zA-Z]*\s*', '', t)
    t = _re.sub(r'\s*```$', '', t).strip()

    # 첫 번째 섹션 태그([Verse] 같은 줄)부터가 진짜 가사다. 그 앞은 전부 머리말.
    m = _re.search(r'^\s*\[[^\]\n]{1,40}\]\s*$', t, _re.M)
    if m:
        t = t[m.start():]

    out = []
    for ln in t.split('\n'):
        s = ln.strip()
        if not s:
            out.append('')
            continue
        if _re.match(r'^(\*|-|•|\d+[.)])\s', s):      # 글머리표 = 자기 점검표
            continue
        low = s.lower()
        if low.startswith(('note:', 'here is', 'here are', 'okay', 'sure,', 'i hope',
                           'let me know', 'translation:', 'title:', '참고:', '제목:')):
            continue
        if '?' in s and _re.search(r'\b(check|yes|no)\b\.?$', low):   # "...? Check." / "...? Yes"
            continue
        out.append(s)

    t = '\n'.join(out)
    t = _re.sub(r'\n{3,}', '\n\n', t).strip()
    return t


def _llm_lyrics(topic: str, style: str, lang: str):
    """글 쓰는 AI에게 가사를 받는다 → (가사, 오류설명)"""
    if not CGO_LLM_KEY:
        return '', 'LLM 키 미설정'
    nat = _LANG_NATIVE.get(lang, lang)
    system = (f"You write song lyrics in {lang} ({nat}). "
              "Reply with the lyrics only \u2014 no preamble, no notes, no lists, no review.")
    user = (f"Theme (direction for you, never a line of the song): {topic}\n"
            f"Style: {style}\n"
            "Format: [Verse] / [Chorus] / [Bridge] tags, 16-24 short singable lines, "
            "chorus repeats word for word.")
    # cgo-475: Gemini 3 계열은 '생각'을 끌 수 없고, 그 생각이 출력 글자 한도를 먹는다.
    # 한도를 1,200자로 잡아뒀더니 생각하다가 한도가 끝나 가사가 안 나왔다
    # (혼잣말만 나온 것도 이 때문). 한도를 넉넉히 주고 생각은 최소로 시킨다.
    body = {"model": CGO_LLM_MODEL,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 1.0, "max_tokens": 4000,
            "reasoning_effort": "low"}
    now = time.time()
    last_err = ''
    r = None
    for model in CGO_LLM_MODELS:
        if _MODEL_REST.get(model, 0) > now:
            continue                                   # 아직 쉬는 중인 모델
        body["model"] = model
        for _try in (1, 2):
            try:
                r = http_requests.post(
                    CGO_LLM_URL,
                    headers={'Authorization': f'Bearer {CGO_LLM_KEY}',
                             'Content-Type': 'application/json'},
                    json=body, timeout=60)
            except Exception as e:
                last_err = f"LLM 연결 예외: {e}"
                r = None
                break
            # reasoning_effort를 모르는 곳이면 그 항목만 빼고 한 번 더
            if r.status_code == 400 and 'reasoning' in r.text.lower() and _try == 1:
                body.pop('reasoning_effort', None)
                continue
            break
        if r is None:
            continue
        if r.ok:
            break
        last_err = f"LLM HTTP {r.status_code}: {r.text[:200]}"
        if r.status_code in (503, 429, 500, 502, 504):
            _MODEL_REST[model] = time.time() + 60       # 붐빔 — 1분 쉬게 두고 옆 모델로
            print(f"[LLM] {model} 붐빔({r.status_code}) → 다음 모델", flush=True)
            continue
        if r.status_code in (400, 404):
            _MODEL_REST[model] = time.time() + 3600     # 이 열쇠로는 못 쓰는 모델
            print(f"[LLM] {model} 사용 불가({r.status_code}) → 다음 모델", flush=True)
            continue
        break                                           # 인증 오류 등은 갈아타도 소용없다
    if r is None or not r.ok:
        return '', last_err or 'LLM 호출 실패'
    if body["model"] != CGO_LLM_MODELS[0]:
        print(f"[LLM] {body['model']} 로 가사 생성", flush=True)
    try:
        d = r.json()
        ch = d['choices'][0]
        txt = (ch['message'].get('content') or '').strip()
        if not txt and ch.get('finish_reason') == 'length':
            return '', "LLM이 생각만 하다 글자 한도에 걸림 (max_tokens 부족)"
    except Exception as e:
        return '', f"LLM 응답 해석 실패: {e} / {r.text[:150]}"
    txt = _clean_lyrics(txt)                      # cgo-474: 생각 과정 걷어내기
    if len(txt) < 30:
        return '', f"LLM 가사가 너무 짧음: {txt[:80]}"
    return txt, ''


def _compact_style(st: str, keep_tail: int = 2) -> str:
    """cgo-478: 수노 스타일 문구를 '키워드 조합'으로 줄인다.
    수노는 긴 문장보다 쉼표로 끊은 핵심 단어들을 훨씬 잘 알아듣는다.
    그동안 우리 문구는 332자/13조각이었다 — 분위기 카드 하나가 영어 문구를
    3개씩 달고 오면서 'Western melodic writing / international pop sensibility /
    A-list Western production'처럼 같은 말이 세 번 들어가기도 했다.
    맨 뒤 BPM·조성은 항상 남긴다(수노가 이 둘은 확실히 읽는다)."""
    import re as _re
    segs = [x.strip() for x in (st or '').split(',')]
    segs = [x for x in segs if x]
    if len(segs) <= keep_tail:
        return st
    head, tail = segs[:-keep_tail], segs[-keep_tail:]      # tail = BPM, key of X
    kept, seen_words = [], set()
    for seg in head:
        low = seg.lower()
        if any(low == k.lower() or low in k.lower() for k in kept):
            continue                                        # 똑같거나 이미 포함된 말
        words = set(_re.findall(r'[a-z]{4,}', low))
        if words and len(words & seen_words) >= max(1, len(words) * 0.6):
            continue                                        # 같은 뜻을 또 말하는 조각
        kept.append(seg)
        seen_words |= words
        if len(kept) >= 9:
            break
    out = ', '.join(kept + tail)
    if len(out) > 240:                                      # 그래도 길면 뒤에서 자른다
        while len(', '.join(kept + tail)) > 240 and len(kept) > 3:
            kept.pop()
        out = ', '.join(kept + tail)
    return out


def _udio_once(prompt_text: str):
    """Udio 가사 API 1회 시도 → (가사, 오류설명). cgo-473부터는 예비 통로."""
    try:
        resp = http_requests.post(
            'https://api.apiframe.ai/v2/music/udio/lyrics',
            headers={'X-API-Key': APIFRAME_KEY, 'Content-Type': 'application/json'},
            json={"prompt": prompt_text, "duration": 97},
            timeout=30
        )
        if not resp.ok:
            return '', f"HTTP {resp.status_code}: {resp.text[:200]}"
        result = resp.json()
        lyr = _extract_lyrics(result)
        if lyr:
            return lyr, ''
        job_id = _job_id_of(result)
        if not job_id:
            return '', f"job ID 없음: {str(result)[:200]}"
        return _poll_apiframe_job(job_id, max_wait=22.0)
    except Exception as e:
        return '', f"API 연결 예외: {e}"

@app.post("/generate_lyrics")
def generate_lyrics(body: dict):
    """AI 가사 생성 — 글 쓰는 AI → 예비로 Udio → 비상 가사. 항상 가사를 돌려준다."""
    topic = (body.get('topic') or '').strip()
    style = (body.get('style') or 'pop ballad').strip()
    # cgo-471: 글을 안 써도 막지 않는다. 고른 분위기(style)만으로도 가사를 만든다.
    # 전에는 400을 돌려줘서, 클릭만으로 곡을 만들려는 사람은 여기서 멈췄다.
    if not topic:
        topic = style if style and style != 'pop ballad' else 'a quiet feeling that is hard to name'
    _lang0 = (body.get('lang') or '').strip() or 'Korean'      # cgo-469
    if body.get('local_only') or not APIFRAME_KEY:
        return _local_lyrics_response(topic, style, '' if APIFRAME_KEY else 'API 키 미설정', _lang0)

    # cgo-469: 노래 언어를 앱에서 받아 그 언어로 쓰게 한다.
    # 지금까지는 "Korean ... Write in Korean"이 못 박혀 있어서, 앱에서 영어를 골라도
    # 서버가 한국어 가사만 주문했다. 앱은 lang을 보내고 있었는데 서버가 안 읽었다.
    lang = (body.get('lang') or '').strip() or 'Korean'
    _nat = _LANG_NATIVE.get(lang, lang)
    prompt_text = (f"Song lyrics in {lang} about: {topic}. Musical style: {style}. "
                   f"Write every single line in {lang} ({_nat}) — do not use any other language. "
                   f"Include [Verse], [Chorus], [Bridge] structure tags. "
                   f"Output the lyrics only.")[:2000]
    api_error = ''
    # cgo-470: Suno 먼저, 안 되면 Udio, 그래도 안 되면 비상 가사.
    # 앱이 120초에 끊으므로 95초 안에서 끝낸다 — 늦게 주느니 비상 가사라도 주는 게 낫다.
    # cgo-472: Suno 가사 창구는 '없다'. apiframe v2의 Suno는 /v2/music/generate 하나뿐이고
    # 그것은 곡을 통째로 만드는 창구다. 가사만 받는 길은 Udio 쪽에만 있다.
    # cgo-470에서 Suno 가사를 먼저 부르게 했던 것은 매번 HTTP 400을 맞는 헛걸음이었다 — 걷어낸다.
    #
    # 재시도 횟수도 3회 → 1회로 줄인다. 요청 한 번마다 비용이 나가므로,
    # 안 될 때 혼자 세 번 더 두드리는 것보다 사용자가 다시 누르게 하는 편이 낫다.
    # cgo-473: ① 글 쓰는 AI → ② Udio(예비) → ③ 비상 가사
    lyr, api_error = _llm_lyrics(topic, style, lang)
    if lyr:
        return JSONResponse(content={"ok": True, "lyrics": lyr, "source": "ai"})
    if api_error != 'LLM 키 미설정':
        print(f"[generate_lyrics] 글쓰는AI 실패: {api_error}", flush=True)

    lyr, err2 = _udio_once(prompt_text)
    if lyr:
        return JSONResponse(content={"ok": True, "lyrics": lyr, "source": "udio"})
    # cgo-474: 주 통로는 글쓰는AI다. 그쪽 사유를 가리지 않는다 —
    # 전에는 Udio의 '자리 없음'이 덮어써서 진짜 원인이 안 보였다.
    if api_error in ('', 'LLM 키 미설정'):
        api_error = err2
    print(f"[generate_lyrics] 둘 다 실패 → 비상 가사 | AI:{api_error} | Udio:{err2}", flush=True)
    return _local_lyrics_response(topic, style, api_error, lang)   # cgo-469

@app.get("/lyrics_status/{job_id}")
def lyrics_status(job_id: str, topic: str = '', style: str = 'pop ballad', lang: str = 'Korean'):
    """(구버전 클라이언트 호환) 가사 job 상태 조회 — 실패 시 topic이 있으면 로컬 가사로 폴백"""
    if not APIFRAME_KEY:
        return _local_lyrics_response(topic, style, 'API 키 미설정', lang) if topic else \
            JSONResponse(content={"ok": False, "status": "FAILED", "error": "API 키 미설정"})
    lyr, err = _poll_apiframe_job(job_id, max_wait=8.0, interval=2.0)
    if lyr:
        return JSONResponse(content={"ok": True, "status": "COMPLETED", "lyrics": lyr})
    if err.startswith('시간 초과'):
        return JSONResponse(content={"ok": True, "status": "PROCESSING", "job_id": job_id})
    if topic:
        r = _generate_korean_lyrics(topic, style)
        return JSONResponse(content={"ok": True, "status": "COMPLETED", "lyrics": r, "source": "local", "api_error": err})
    return JSONResponse(content={"ok": False, "status": "FAILED", "error": f"가사 생성 실패: {err}"})



# ═══ CGO 보컬 믹서 API (cgo-390) ═══
import random as _vmrandom

@app.get("/voice-mix/random")
def voice_mix_random(count: int = 1):
    """랜덤 보컬 믹스 프리셋 반환. count=1~5"""
    count = max(1, min(count, 5))
    ids = _vmrandom.sample(list(VOICE_MIX.keys()), min(count, len(VOICE_MIX)))
    result = []
    for mid in ids:
        m = VOICE_MIX[mid]
        result.append({
            "id": mid,
            "cat": m["cat"],
            "tag": m["tag"],
            "w": m["w"],
        })
    return {"mixes": result}

@app.get("/voice-mix/{mix_id}")
def voice_mix_get(mix_id: str):
    """특정 믹스 ID의 프롬프트 반환 (서버에서만 프롬프트 노출)"""
    if mix_id not in VOICE_MIX:
        return {"error": "not found"}
    m = VOICE_MIX[mix_id]
    return {
        "id": mix_id,
        "cat": m["cat"],
        "tag": m["tag"],
        "w": m["w"],
        "prompt": m["prompt"],
    }

@app.post("/voice-mix/custom")
def voice_mix_custom(layers: dict):
    """
    5레이어 가중치 기반 커스텀 보컬 프롬프트 생성.
    입력: {"L1":60,"L2":30,"L3":10,"L4":0,"L5":0}
    서버에서 레이어별 아티스트를 가중 선택 → 합성 프롬프트 반환.
    """
    layer_weights = {
        "L1": layers.get("L1", 0),
        "L2": layers.get("L2", 0),
        "L3": layers.get("L3", 0),
        "L4": layers.get("L4", 0),
        "L5": layers.get("L5", 0),
    }
    total = sum(layer_weights.values())
    if total == 0:
        return {"error": "at least one layer must have weight > 0"}

    # Normalize to 100
    for k in layer_weights:
        layer_weights[k] = round(layer_weights[k] * 100 / total)

    # Pick closest preset by layer profile similarity
    best_id = None
    best_score = float('inf')
    for mid, m in VOICE_MIX.items():
        # Compare layer distribution via weights
        score = abs(m["w"][0] - max(layer_weights.values()))
        if score < best_score:
            best_score = score
            best_id = mid

    if best_id:
        m = VOICE_MIX[best_id]
        return {
            "id": best_id,
            "tag": m["tag"],
            "prompt": m["prompt"],
            "layers": layer_weights,
        }
    return {"error": "no match"}


# cgo-439: 실행 블록은 반드시 파일 맨 끝 (모든 라우트 등록 후 서버 시작)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
