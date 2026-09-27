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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional, Dict

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


@app.get("/")
def root():
    return {"ok": True, "service": "cgo-render", "sf2": _find_sf2(), "vvip": True}


# ═══════════════════════════════════════════════════════════════════
# VVIP AI 보컬 엔드포인트 — Suno via apiframe.ai
# CGO-FULI 작곡 데이터 + 사용자 프롬프트 → Suno API → AI 보컬 곡 생성
# 원칙: 최소 2명 이상의 서로 다른 보컬리스트 조합 필수
# ═══════════════════════════════════════════════════════════════════

APIFRAME_KEY = os.environ.get('APIFRAME_KEY', 'afk_32b0a883e107754089c02eb5977a0945958096b7')

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
    '임재범': 'beast-like husky male vocal with Korean emotional sorrow and raw power',
    '이문세': 'deep literary lyrical male Korean pop ballad vocal with quiet resonance',
    '김범수': 'flawless technique male Korean vocal mastering every emotion perfectly',
    '박효신': 'evolved male vocal reaching divine territory from folk to pop ballad perfection',
    '임창정': 'desperately emotional high-pitched male Korean vocal that makes everyone cry',
    '조용필': 'the king of Korean pop, versatile male vocal covering rock ballad and folk',
    '김광석': 'warm folk acoustic Korean male vocal with heartfelt lonely storytelling',
    '김현식': 'rough torn raspy male vocal pouring soul until the last breath',
    '하현우': 'stable ultra-high male vocal testing human limits with incredible range',
    '이수': 'overwhelming falsetto high male Korean ballad vocal dominating karaoke',
    '나얼': 'pinnacle of Korean R&B soul male vocal with perfect high-tone technique',
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
    '로린힐': 'husky soulful female vocal fusing hip-hop and R&B with deep charm',
    '로린 힐': 'husky soulful female vocal fusing hip-hop and R&B with deep charm',
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
    '소향': 'world-class 5-octave powerful Korean female vocal with dramatic high notes and pop diva power',
    '태연': 'unique delicate Korean female vocal representing a generation with perfect control and emotion',
    '이선희': 'explosive power from small frame, timeless clear sorrowful Korean female vocal',
    '백지영': 'husky heartbreak-filled Korean female OST ballad queen vocal',
    '박정현': 'Korean R&B fairy female vocal with perfect breath control and brilliant melisma',
    '거미': 'deep husky soulful Korean female ballad vocal with Black soul influence',
    '이은미': 'barefoot diva, deeply appealing Korean female vocal drawn from the depths of the heart',
    '인순이': 'explosive powerful energetic Korean female vocal mastering soul dance and pop',
    '윤미래': 'sticky deep husky Korean female hip-hop R&B vocal at the pinnacle',
    '아이유': 'gentle acoustic to soaring high notes, clear storytelling Korean female vocal defining an era',
    '심수봉': 'legendary nasal sorrowful uniquely toned Korean trot folk female vocal soaking the soul',
    '소찬휘': 'ultimate Korean female high-note queen with blade-sharp piercing rapid vocal delivery',
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
    '나훈아': 'deep emotional dramatic Korean male trot vocal with powerful vibrato',
}


class VvipReq(BaseModel):
    prompt: str = ""
    lyrics: str = ""
    style: str = "pop"
    bpm: int = 120
    key: str = "C"
    chords: str = ""


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

    # ── 최소 2명 이상 보컬 조합 검증 (CGO-FULI VVIP 원칙) ──
    matched_artists = []
    seen_descriptions: set = set()
    for artist, description in VOICE_MAP.items():
        if artist in prompt and description not in seen_descriptions:
            matched_artists.append(artist)
            seen_descriptions.add(description)

    if len(seen_descriptions) < 2:
        return JSONResponse(status_code=400, content={
            "ok": False,
            "error": "VVIP는 2명 이상의 서로 다른 목소리 조합이 필요합니다.",
            "hint": "예: 파워디바+감성발라드 등 2개 이상 스타일을 선택하세요",
            "matched": len(seen_descriptions)
        })

    # ── 가수 이름 → 보컬 설명 변환 (Suno 정책 준수) ──
    converted_prompt = prompt
    for artist, description in VOICE_MAP.items():
        if artist in converted_prompt:
            converted_prompt = converted_prompt.replace(artist, description)

    # ── Suno 프롬프트 조합 ──
    suno_prompt = f"{converted_prompt}, {req.style}, {req.bpm} BPM, key of {req.key}"
    if req.chords:
        suno_prompt += f", chord progression: {req.chords}"

    # ── apiframe.ai v2 API 호출 (비동기: job_id만 즉시 반환) ──
    has_lyrics = bool(req.lyrics and req.lyrics.strip())
    api_body = {
        "prompt": req.lyrics.strip() if has_lyrics else suno_prompt,
        "model": "suno",
        "sunoParams": {
            "custom_mode": has_lyrics,
            "instrumental": False,
            "model_version": "V4_5PLUS"
        }
    }
    if has_lyrics:
        api_body["sunoParams"]["style"] = suno_prompt

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
