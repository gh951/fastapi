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
    '메리제이블라이즈': 'commanding hip-hop soul alto with passionate 90s ballad grit',
    '메리 제이 블라이즈': 'commanding hip-hop soul alto with passionate 90s ballad grit',
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
    '바비킴': 'unique husky voice blending soul reggae groove with hip-hop inflection',
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
    '안예은': 'creative fusion soprano blending traditional Korean vocal color with modern tone',
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
    '정인': 'uniquely textured alto with rare Korean soulful groove and smoky warmth',
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
    '노토리어스 비아이쥐': 'heavyweight commanding male rapper with flawless flow and deep groove mastery',
    '노토리어스비아이쥐': 'heavyweight commanding male rapper with flawless flow and deep groove mastery',
    '나스': 'poetic conscious male rapper with precise lyrical craftsmanship and street narrative depth',
    '제이 지': 'authoritative smooth male rapper with business-mogul swagger and effortless delivery',
    '제이지': 'authoritative smooth male rapper with business-mogul swagger and effortless delivery',
    '라킴': 'pioneering male rapper who defined modern rhyme schemes with meticulous cadence',
    '에미넴': 'explosive rapid-fire male rapper with razor-sharp diction and unmatched global impact',
    '디엠엑스': 'raw gravelly male rapper with ferocious barking energy and intense emotional delivery',
    '빅 엘': 'razor-sharp punchline male rapper with dazzling lyrical acrobatics and tragic brilliance',
    '빅엘': 'razor-sharp punchline male rapper with dazzling lyrical acrobatics and tragic brilliance',
    '메서드 맨': 'distinctive husky low-tone male rapper with gritty charismatic flow and swagger',
    '메서드맨': 'distinctive husky low-tone male rapper with gritty charismatic flow and swagger',
    '레드맨': 'funky freewheeling male rapper with raw unfiltered energy on boom-bap beats',
    '모스 뎁': 'intellectually refined male rapper blending conscious lyricism with jazz-boom-bap soul',
    '모스뎁': 'intellectually refined male rapper blending conscious lyricism with jazz-boom-bap soul',
    '탈립 퀠리': 'cerebral eloquent male rapper elevating hip-hop artistry with intricate rhyme patterns',
    '탈립퀠리': 'cerebral eloquent male rapper elevating hip-hop artistry with intricate rhyme patterns',
    '스릭 릭': 'legendary storytelling male rapper with unique accent and theatrical narrative flow',
    '스릭릭': 'legendary storytelling male rapper with unique accent and theatrical narrative flow',
    '엘엘 쿨 제이': 'pioneering male rapper mastering both hardcore hip-hop and smooth love-rap delivery',
    '엘엘쿨제이': 'pioneering male rapper mastering both hardcore hip-hop and smooth love-rap delivery',
    '빅 푸니셔': 'relentless rapid-fire male rapper with massive projection and boom-bap dominance',
    '빅푸니셔': 'relentless rapid-fire male rapper with massive projection and boom-bap dominance',
    '빅 대디 케인': 'fleet-footed male rapper with dazzling speed and showmanship from the golden era',
    '빅대디케인': 'fleet-footed male rapper with dazzling speed and showmanship from the golden era',
    '에이셉 라키': 'trendy stylish male rapper layering fashion-forward aesthetics over New York boom-bap',
    '에이셉라키': 'trendy stylish male rapper layering fashion-forward aesthetics over New York boom-bap',
    '크리스 크로스': 'authoritative male rapper who defined boom-bap philosophy with intellectual power',
    '크리스크로스': 'authoritative male rapper who defined boom-bap philosophy with intellectual power',
    '빅 원': 'underground gritty male rapper with raw boom-bap sensibility and street authenticity',
    '빅원': 'underground gritty male rapper with raw boom-bap sensibility and street authenticity',
    '구루': 'monotone mid-bass male rapper fusing jazz harmonics with boom-bap groove seamlessly',
    '조이 배드애스': 'modern male rapper perfectly reviving 90s golden-era New York boom-bap aesthetics',
    '조이배드애스': 'modern male rapper perfectly reviving 90s golden-era New York boom-bap aesthetics',
    # ── 🌴 웨스트 코스트 & 지펑크 (West Coast / G-Funk) ──
    '투팍 샤커': 'passionate revolutionary male rapper with soul-stirring delivery and poetic intensity',
    '투팍샤커': 'passionate revolutionary male rapper with soul-stirring delivery and poetic intensity',
    '투팍': 'passionate revolutionary male rapper with soul-stirring delivery and poetic intensity',
    '스눕 독': 'silky laid-back male rapper with signature drawl and effortless west-coast groove',
    '스눕독': 'silky laid-back male rapper with signature drawl and effortless west-coast groove',
    '닥터 드레': 'authoritative deep male vocal who architected G-Funk sound and discovered legends',
    '닥터드레': 'authoritative deep male vocal who architected G-Funk sound and discovered legends',
    '아이스 큐브': 'aggressive hard-hitting male rapper with menacing gangsta delivery and sharp wit',
    '아이스큐브': 'aggressive hard-hitting male rapper with menacing gangsta delivery and sharp wit',
    '켄드릭 라마': 'virtuoso male rapper with chameleonic vocal range and Pulitzer-level storytelling',
    '켄드릭라마': 'virtuoso male rapper with chameleonic vocal range and Pulitzer-level storytelling',
    '더 게임': 'rugged male rapper blending gritty tone with gangster balladry and west-coast soul',
    '더게임': 'rugged male rapper blending gritty tone with gangster balladry and west-coast soul',
    '이지 이': 'distinctive high-pitched male rapper who pioneered gangsta rap with piercing tone',
    '이지이': 'distinctive high-pitched male rapper who pioneered gangsta rap with piercing tone',
    '네이트 독': 'melodic hook-master male vocalist who defined G-Funk with soulful singing-rap fusion',
    '네이트독': 'melodic hook-master male vocalist who defined G-Funk with soulful singing-rap fusion',
    '워렌 지': 'smooth romantic male rapper delivering the most mellow G-Funk with laid-back flow',
    '워렌지': 'smooth romantic male rapper delivering the most mellow G-Funk with laid-back flow',
    '엑지빗': 'powerfully raspy male rapper with aggressive west-coast punch and commanding presence',
    '쿨리오': 'chart-dominating male rapper with infectious melodic hooks and global crossover appeal',
    '사이프레스 힐': 'nasal high-pitched male rapper who pioneered Latin west-coast hip-hop uniquely',
    '사이프레스힐': 'nasal high-pitched male rapper who pioneered Latin west-coast hip-hop uniquely',
    '맥 텐': 'hard-hitting gangster male rapper delivering textbook west-coast hardcore with authority',
    '맥텐': 'hard-hitting gangster male rapper delivering textbook west-coast hardcore with authority',
    '디제이 퀵': 'multi-talented male rapper-producer with refined west-coast lyricism and groove mastery',
    '디제이퀵': 'multi-talented male rapper-producer with refined west-coast lyricism and groove mastery',
    '쿠럽': 'technically gifted male rapper with extraordinary rhyme arrangement on west-coast beats',
    '대즈 딜린저': 'deep-voiced male rapper-producer who powered Death Row Records golden era sound',
    '대즈딜린저': 'deep-voiced male rapper-producer who powered Death Row Records golden era sound',
    '엠씨 아이트': 'cold atmospheric male rapper delivering chilling street narratives with west-coast cool',
    '엠씨아이트': 'cold atmospheric male rapper delivering chilling street narratives with west-coast cool',
    '스쿨보이 큐': 'gritty groovy modern male rapper with aggressive delivery on contemporary west-coast',
    '스쿨보이큐': 'gritty groovy modern male rapper with aggressive delivery on contemporary west-coast',
    '타이 달라 사인': 'versatile male vocalist seamlessly fusing rap and R&B over laid-back west-coast beats',
    '타이달라사인': 'versatile male vocalist seamlessly fusing rap and R&B over laid-back west-coast beats',
    '비지 본': 'lightning-fast male rapper layering angelic melodies over rapid-fire delivery uniquely',
    '비지본': 'lightning-fast male rapper layering angelic melodies over rapid-fire delivery uniquely',
    # ── 🔥 서던 힙합 & 트랩 (Southern / Trap) ──
    '티아이': 'commanding male rapper who coined trap music with authoritative Atlanta swagger',
    '구찌 메인': 'foundational male rapper who architected modern trap hip-hop culture and sound',
    '구찌메인': 'foundational male rapper who architected modern trap hip-hop culture and sound',
    '릴 웨인': 'inventive genius male rapper with unique voice and otherworldly metaphorical wordplay',
    '릴웨인': 'inventive genius male rapper with unique voice and otherworldly metaphorical wordplay',
    '퓨처': 'autotune-wielding male rapper who perfected modern melodic trap with hypnotic delivery',
    '트래비스 콧': 'psychedelic male rapper fusing rock energy with stadium-shaking trap production',
    '트래비스콧': 'psychedelic male rapper fusing rock energy with stadium-shaking trap production',
    '영 턱': 'revolutionary male rapper who weaponized his voice as an instrument destroying conventions',
    '영턱': 'revolutionary male rapper who weaponized his voice as an instrument destroying conventions',
    '빅 보이': 'rapid-fire southern male rapper elevating OutKast with virtuoso technical precision',
    '빅보이': 'rapid-fire southern male rapper elevating OutKast with virtuoso technical precision',
    '안드레 3000': 'wildly innovative southern male rapper and hip-hop genre most artistic genre-bending voice',
    '안드레3000': 'wildly innovative southern male rapper and hip-hop genre most artistic genre-bending voice',
    '루다크리스': 'hard-hitting precise male rapper with thunderous diction and humorous southern delivery',
    '릭 로스': 'heavyweight deep-bass male rapper dominating grandiose trap beats with boss authority',
    '릭로스': 'heavyweight deep-bass male rapper dominating grandiose trap beats with boss authority',
    '퀘이보': 'triplet-flow male rapper who rewrote global trap standards with infectious cadence',
    '오프셋': 'technically sharp male rapper with rapid-fire triplet flow and aggressive punch',
    '테이크오프': 'precise surgical male rapper with clean triplet delivery and understated excellence',
    '릴 존': 'explosive crunk-pioneer male rapper with earth-shaking club energy and powerful vocals',
    '릴존': 'explosive crunk-pioneer male rapper with earth-shaking club energy and powerful vocals',
    '쥬시 제이': 'underground legend male rapper who built southern hardcore trap skeletal framework',
    '쥬시제이': 'underground legend male rapper who built southern hardcore trap skeletal framework',
    '미스틱칼': 'thunderous military-grade male rapper with overwhelming volume and raw southern energy',
    '투 체인즈': 'addictive punchline male rapper with playful charisma and infectious trap mastery',
    '투체인즈': 'addictive punchline male rapper with playful charisma and infectious trap mastery',
    '지 지': 'husky gravelly male rapper delivering authentic Atlanta street narratives with grit',
    '지지': 'husky gravelly male rapper delivering authentic Atlanta street narratives with grit',
    '릴 바비': 'precision-engineered modern male rapper with relentless continuous trap flow dominance',
    '릴바비': 'precision-engineered modern male rapper with relentless continuous trap flow dominance',
    '건나': 'silky sliding male rapper perfecting melodic trap with fluid effortless delivery',
    '맥클모어': 'accessible narrative male rapper layering popular storytelling over southern-style production',
    '21 새비지': 'ice-cold monotone male rapper embodying modern dark trap with deadpan delivery',
    '21새비지': 'ice-cold monotone male rapper embodying modern dark trap with deadpan delivery',
    # ── 💫 21세기 하이브리드 & 얼터너티브 (Alternative / Hybrid) ──
    '카니예 웨스트': 'paradigm-shifting male rapper-producer and hip-hop genre greatest sonic innovator ever',
    '카니예웨스트': 'paradigm-shifting male rapper-producer and hip-hop genre greatest sonic innovator ever',
    '드레이크': 'chart-dominating male rapper-singer who demolished the rap-R&B boundary worldwide',
    '키드 커디': 'dreamy alternative male rapper who implanted psychedelic rock sensibility into hip-hop',
    '키드커디': 'dreamy alternative male rapper who implanted psychedelic rock sensibility into hip-hop',
    '릴 우지 버트': 'emo-rock infused male rapper fusing emotional intensity with rapid hi-hat trap',
    '릴우지버트': 'emo-rock infused male rapper fusing emotional intensity with rapid hi-hat trap',
    '주스 월드': 'heart-wrenching melodic male rapper who epitomized emo-rap with devastating melodies',
    '주스월드': 'heart-wrenching melodic male rapper who epitomized emo-rap with devastating melodies',
    '릴 피프': 'punk-rock crossover male rapper who perfectly blended hardcore punk with trap',
    '릴피프': 'punk-rock crossover male rapper who perfectly blended hardcore punk with trap',
    '타일러 더 크리에이터': 'eccentric brilliant male rapper commanding neo-soul alternative hip-hop with vision',
    '타일러더크리에이터': 'eccentric brilliant male rapper commanding neo-soul alternative hip-hop with vision',
    '에이셉 퍼그': 'energetic flashy male rapper with wild hybrid flow over hardcore club beats',
    '에이셉퍼그': 'energetic flashy male rapper with wild hybrid flow over hardcore club beats',
    '차일디시 감비노': 'multi-talented male rapper-actor delivering socially charged alternative hip-hop masterfully',
    '차일디시감비노': 'multi-talented male rapper-actor delivering socially charged alternative hip-hop masterfully',
    '포스트 말론': 'genre-blending male vocalist fusing rock, country, and trap with healing timbre',
    '포스트말론': 'genre-blending male vocalist fusing rock, country, and trap with healing timbre',
    '엑스엑스엑스텐타시온': 'versatile raw male rapper spanning distorted lo-fi beats to tender acoustic rap',
    '플레이보이 카티': 'hypnotic baby-voice male rapper commanding rave-trap with addictive minimalist flow',
    '플레이보이카티': 'hypnotic baby-voice male rapper commanding rave-trap with addictive minimalist flow',
    '맥 밀러': 'soulful jazzy male rapper weaving indie rock and jazz into hip-hop beautifully',
    '맥밀러': 'soulful jazzy male rapper weaving indie rock and jazz into hip-hop beautifully',
    '로직': 'rapid-fire technical male rapper balancing speed with accessible pop-ballad sensibility',
    '지이지': 'polished male rapper with clean jazz-synthpop beats and sophisticated hybrid delivery',
    '와이클리프 장': 'Caribbean-fusion male rapper crossing reggae, Latin, and hip-hop with 90s mastery',
    '와이클리프장': 'Caribbean-fusion male rapper crossing reggae, Latin, and hip-hop with 90s mastery',
    '비오비': 'pop-rock acoustic male rapper who conquered Billboard with accessible crossover sound',
    '자 룰': 'husky passionate male rapper who pioneered love-rap duets with R&B singers globally',
    '자룰': 'husky passionate male rapper who pioneered love-rap duets with R&B singers globally',
    '플로 라이다': 'EDM-fused male rapper dominating global clubs with electronic dance-rap energy',
    '플로라이다': 'EDM-fused male rapper dominating global clubs with electronic dance-rap energy',
    '위즈 칼리파': 'mellow melodic male rapper with addictive hooks and relaxed hybrid ballad delivery',
    '위즈칼리파': 'mellow melodic male rapper with addictive hooks and relaxed hybrid ballad delivery',
    # ── 🇬🇧 글로벌 영미권 & 그라임/드릴 (UK / Drill / Global) ──
    '팝 스모크': 'thunderous deep-cave male rapper who exploded Brooklyn drill onto the global stage',
    '팝스모크': 'thunderous deep-cave male rapper who exploded Brooklyn drill onto the global stage',
    '스켑타': 'razor-sharp UK grime male rapper representing London streets with global authority',
    '스톰지': 'powerful UK national male rapper fusing grime with classic soul harmonics brilliantly',
    '센트럴 씨': 'trend-setting UK drill male rapper dominating global shorts and reels effortlessly',
    '센트럴씨': 'trend-setting UK drill male rapper dominating global shorts and reels effortlessly',
    '제이훕': 'Afrobeat-drill fusion male rapper who pioneered Afroswing genre with infectious energy',
    '데이브': 'genius lyricist UK male rapper delivering profound narratives over piano-driven beats',
    '긱스': 'slow heavyweight UK underground male rapper with iconic deep bass flow delivery',
    '헤디 원': 'precisely polished UK drill male rapper with the most refined rhythmic mastery',
    '헤디원': 'precisely polished UK drill male rapper with the most refined rhythmic mastery',
    '디지 래스칼': 'revolutionary UK grime male rapper who first brought grime to mainstream success',
    '디지래스칼': 'revolutionary UK grime male rapper who first brought grime to mainstream success',
    '크로프트': 'wordplay-brilliant UK duo male rappers with pop-friendly drill beat chemistry',
    '케난': 'Somali-born male rapper fusing African traditional rhythms with uplifting hip-hop spirit',
    '티제이': 'hard-hitting slide-drill male rapper with powerful 808 bass-riding technique',
    '토니 레인즈': 'explosive Canadian male rapper-singer with 80s-90s sampling and dynamic versatility',
    '토니레인즈': 'explosive Canadian male rapper-singer with 80s-90s sampling and dynamic versatility',
    '나브': 'atmospheric Canadian male rapper crafting dreamy synth-trap soundscapes with soft delivery',
    '아웃로우즈': 'gangster-crew male rappers carrying west-coast and global rap legacy with authority',
    '빅 주': 'heavyweight drum-and-bass male rapper with signature UK grime flow mastery',
    '빅주': 'heavyweight drum-and-bass male rapper with signature UK grime flow mastery',
    '아론 스미스': 'electronic-trap crossover male rapper blending synths with global hybrid rap delivery',
    '아론스미스': 'electronic-trap crossover male rapper blending synths with global hybrid rap delivery',
    '제이 콜': 'wise philosophical male rapper layering classic boom-bap soul with modern narratives',
    '제이콜': 'wise philosophical male rapper layering classic boom-bap soul with modern narratives',
    # ═══════════════════════════════════════════════════════════
    # cgo-383: 힙합 여자 보컬 100명
    # ═══════════════════════════════════════════════════════════
    # ── 🥁 정통 붐뱁 & 올드스쿨 전설 (Boom-Bap / Old-School) ──
    '미시 에일리엇': 'innovative female rapper-producer with revolutionary visual and sonic hip-hop mastery',
    '미시에일리엇': 'innovative female rapper-producer with revolutionary visual and sonic hip-hop mastery',
    '퀸 라티파': 'commanding female rapper who elevated hip-hop with social messages and artistic gravitas',
    '퀸라티파': 'commanding female rapper who elevated hip-hop with social messages and artistic gravitas',
    '엠씨 라이트': 'pioneering boom-bap female rapper who achieved the first solo gold album for women',
    '엠씨라이트': 'pioneering boom-bap female rapper who achieved the first solo gold album for women',
    '솔트': 'legendary 80s female rapper who spearheaded hip-hop mainstream with infectious energy',
    '페파': 'charismatic bold female rapper who transformed hip-hop genetics with playful authority',
    '록샌 섕테': 'battle-rap legend female rapper with unmatched improvisational freestyle brilliance',
    '록샌섕테': 'battle-rap legend female rapper with unmatched improvisational freestyle brilliance',
    '백시 미터': 'understated monotone boom-bap female rapper dominating 90s Philadelphia underground',
    '백시미터': 'understated monotone boom-bap female rapper dominating 90s Philadelphia underground',
    '모니 러브': 'witty transatlantic female rapper with clever flow who defined an era gracefully',
    '모니러브': 'witty transatlantic female rapper with clever flow who defined an era gracefully',
    '폭시 브라운': 'fierce sharp-tongued New York female rapper with aggressive boom-bap delivery',
    '폭시브라운': 'fierce sharp-tongued New York female rapper with aggressive boom-bap delivery',
    '라 비아': 'heavyweight hardcore boom-bap female rapper with solid powerful projection and grit',
    '라비아': 'heavyweight hardcore boom-bap female rapper with solid powerful projection and grit',
    '레이디 오브 레이지': 'explosive hardcore female rapper who shredded 90s Death Row beats with fury',
    '레이디오브레이지': 'explosive hardcore female rapper who shredded 90s Death Row beats with fury',
    '이브': 'fierce female rapper from Ruff Ryders dominating 2000s with aggressive powerful flow',
    '리사 로페즈': 'legendary high-tone female rapper with rhythmic agility and iconic vocal presence',
    '리사로페즈': 'legendary high-tone female rapper with rhythmic agility and iconic vocal presence',
    '챰 브레이클리': 'intense 90s New York hardcore female rapper with striking lyrical craftsmanship',
    '챰브레이클리': 'intense 90s New York hardcore female rapper with striking lyrical craftsmanship',
    '진 그레이': 'underground technical female rapper with the most intricate rhyme architecture',
    '진그레이': 'underground technical female rapper with the most intricate rhyme architecture',
    '트리나': 'Miami hardcore female rapper who built southern rap foundation with bold delivery',
    '갱스타 부': 'legendary Three 6 Mafia female rapper ruling southern underground with fierce authority',
    '갱스타부': 'legendary Three 6 Mafia female rapper ruling southern underground with fierce authority',
    '레이디 럭': 'skilled battle-rap female rapper who shook early 2000s New York underground scene',
    '레이디럭': 'skilled battle-rap female rapper who shook early 2000s New York underground scene',
    '랩소디': 'modern boom-bap female rapper praised by legends for supreme lyrical wisdom and craft',
    # ── 👑 빌보드 지배자 & 하드코어 퀸 (Hardcore & Billboard Queens) ──
    '니키 미나즈': 'rapid-fire versatile female rapper with unmatched diction and genre-defining delivery',
    '니키미나즈': 'rapid-fire versatile female rapper with unmatched diction and genre-defining delivery',
    '카디 비': 'raw powerful female rapper conquering Billboard with aggressive unfiltered street energy',
    '카디비': 'raw powerful female rapper conquering Billboard with aggressive unfiltered street energy',
    '메간 디 스탈리온': 'relentless southern female rapper with heavyweight flow pounding hard-hitting beats',
    '메간디스탈리온': 'relentless southern female rapper with heavyweight flow pounding hard-hitting beats',
    '릴 킴': 'trailblazing New York female rapper who set fashion and lyrical standards for women',
    '릴킴': 'trailblazing New York female rapper who set fashion and lyrical standards for women',
    '라토': 'polished Atlanta trap female rapper with sophisticated hooks and confident swagger',
    '글로릴라': 'deep-voiced southern female rapper with heavy 808 impact and raw visceral power',
    '아이스 스파이스': 'Bronx drill female rapper who conquered global shorts with irresistible trend-setting',
    '아이스스파이스': 'Bronx drill female rapper who conquered global shorts with irresistible trend-setting',
    '글로리아 마르티네즈': 'Latin Afro-beat female rapper who shook Grammy stages with hardcore fusion vocals',
    '글로리아마르티네즈': 'Latin Afro-beat female rapper who shook Grammy stages with hardcore fusion vocals',
    '레미 마': 'Terror Squad pride female rapper with authentic New York hardcore hip-hop dignity',
    '레미마': 'Terror Squad pride female rapper with authentic New York hardcore hip-hop dignity',
    '아잘리아 뱅크스': 'technically brilliant female rapper who perfectly crosses house-EDM with hardcore rap',
    '아잘리아뱅크스': 'technically brilliant female rapper who perfectly crosses house-EDM with hardcore rap',
    '이고 아잘리아': 'Australian-born female rapper who topped Billboard hip-hop charts with crossover hits',
    '이고아잘리아': 'Australian-born female rapper who topped Billboard hip-hop charts with crossover hits',
    '제이티': 'fierce Miami trap duo female rapper with unrestrained lyrics and infectious groove',
    '융 마이애미': 'bouncy southern trap female rapper optimized for hi-hat-driven hardcore delivery',
    '융마이애미': 'bouncy southern trap female rapper optimized for hi-hat-driven hardcore delivery',
    '빅 보스 벨라': 'viral hook-machine female rapper crafting addictive trap hooks for social media',
    '빅보스벨라': 'viral hook-machine female rapper crafting addictive trap hooks for social media',
    '소이티': 'glamorous west-coast female rapper delivering lifestyle bars over stylish sampled beats',
    '도치': 'next-generation hardcore female rapper excelling at rap, singing, and stage performance',
    '플로 밀리': 'sharp high-tone female rapper with clever off-beat flow and chart-penetrating diction',
    '플로밀리': 'sharp high-tone female rapper with clever off-beat flow and chart-penetrating diction',
    '티아코라': 'anime-aesthetic female rapper blending quirky visuals with unique trap beat choices',
    '티에라 왁': 'inventive creative female rapper who stunned music world with one-minute track concepts',
    '티에라왁': 'inventive creative female rapper who stunned music world with one-minute track concepts',
    '코이 르레이': 'light floating female rapper with melodic singing-rap over 808 trap beats uniquely',
    '코이르레이': 'light floating female rapper with melodic singing-rap over 808 trap beats uniquely',
    # ── 🔥 트랩 & 멜로딕 힙합 (Trap & Melodic Drill) ──
    '도자 캣': 'genre-crossing female artist seamlessly switching between pop, R&B, and hardcore trap',
    '도자캣': 'genre-crossing female artist seamlessly switching between pop, R&B, and hardcore trap',
    '리틀 심즈': 'globally acclaimed UK female rapper with orchestral trap narratives and depth',
    '리틀심즈': 'globally acclaimed UK female rapper with orchestral trap narratives and depth',
    '아쿠아 나루': 'poetic jazz-harmony female rapper fusing intricate trap beats with literary grace',
    '아쿠아나루': 'poetic jazz-harmony female rapper fusing intricate trap beats with literary grace',
    '캄린': '90s G-Funk revivalist female rapper reinterpreting west-coast vibes through modern trap',
    '리코 네스티': 'punk-rage female rapper pioneering trap-metal by fusing rock fury with hardcore beats',
    '리코네스티': 'punk-rage female rapper pioneering trap-metal by fusing rock fury with hardcore beats',
    '영 마': 'Brooklyn deep-voiced female rapper dominating trap beats with cold mid-bass authority',
    '영마': 'Brooklyn deep-voiced female rapper dominating trap beats with cold mid-bass authority',
    '노네임': 'whisper-soft literary female rapper floating poetically over jazz and lo-fi trap beats',
    '덱 로프': 'gentle melodic trap female rapper who pioneered smooth singing-rap trends in 2010s',
    '덱로프': 'gentle melodic trap female rapper who pioneered smooth singing-rap trends in 2010s',
    '비비머타': 'sharp organic indie-trap female rapper creating the rawest underground groove patterns',
    '샤이걸': 'UK club hyperpop female rapper crossing electronic beats with melodic trap seamlessly',
    '아르마니 시저': 'Griselda Records queen female rapper moving between boom-bap and dark trap mastery',
    '아르마니시저': 'Griselda Records queen female rapper moving between boom-bap and dark trap mastery',
    '레이디 레셔': 'ultra-fast UK grime female rapper with blazing speed over trap and drill beats',
    '레이디레셔': 'ultra-fast UK grime female rapper with blazing speed over trap and drill beats',
    '스테플론 돈': 'dancehall-reggae female rapper fusing Caribbean rhythms with UK drill trap globally',
    '스테플론돈': 'dancehall-reggae female rapper fusing Caribbean rhythms with UK drill trap globally',
    '쉔시아': 'Caribbean-flavored female rapper who effortlessly rides trendy American trap production',
    '프린세스 노키아': 'NYC underground queen female rapper embodying alternative trap with raw authenticity',
    '프린세스노키아': 'NYC underground queen female rapper embodying alternative trap with raw authenticity',
    '레일라': 'dark European drill female rapper commanding heavy trap beats with ominous presence',
    '글로벌 믹스 래퍼': 'Eastern-melodic female rapper crossing Asian tonality with dark trap production',
    '글로벌믹스래퍼': 'Eastern-melodic female rapper crossing Asian tonality with dark trap production',
    '엠아이에이': 'revolutionary female rapper fusing third-world percussion with electronic trap radically',
    '산티골드': 'new-wave rock female rapper who shattered boundaries between rock and hybrid trap',
    '나오': 'sophisticated tension-chord female vocalist with distinctive falsetto over refined trap beats',
    # ── 🇬🇧 글로벌 영미권 & 그라임/드릴 (UK / Drill / Global) ──
    '에니': 'London-born healing female rapper with jazz-hip-hop warmth over refined drill beats',
    '아이라 스타': 'Afrobeat-drill hybrid female rapper setting global rhythm trends with infectious energy',
    '아이라스타': 'Afrobeat-drill hybrid female rapper setting global rhythm trends with infectious energy',
    '이브스 투모어': 'radically alternative female rapper fusing avant-garde sound with trap production',
    '이브스투모어': 'radically alternative female rapper fusing avant-garde sound with trap production',
    '토미 제네시스': 'Canadian dark-aesthetic female rapper with unique fetish-rap style over trap beats',
    '토미제네시스': 'Canadian dark-aesthetic female rapper with unique fetish-rap style over trap beats',
    '수가 티': 'historic west-coast crew female rapper with distinctive original flow and presence',
    '수가티': 'historic west-coast crew female rapper with distinctive original flow and presence',
    '미즈 다이너마이트': 'pioneering UK grime-garage female rapper who opened mainstream doors in 2000s',
    '미즈다이너마이트': 'pioneering UK grime-garage female rapper who opened mainstream doors in 2000s',
    '나바': 'microtonal Arab-maqam female rapper crossing Middle-Eastern melody with drill beats',
    '엠씨 멜로디': 'Dutch boom-bap female rapper who captivated all of Europe with classic flow',
    '엠씨멜로디': 'Dutch boom-bap female rapper who captivated all of Europe with classic flow',
    '니나 디아즈': 'Latin rock-drill crossover female rapper with hybrid trap energy and raw power',
    '니나디아즈': 'Latin rock-drill crossover female rapper with hybrid trap energy and raw power',
    '디암스': 'France greatest-selling female rapper with epic narrative storytelling and authority',
    '제니': 'globally verified K-pop female rapper with trendy English trap flow and stage power',
    '드리지': 'Chicago hardcore drill female rapper with precise punchlines and aggressive delivery',
    '차이나': 'Philadelphia dark-cloud trap female rapper with haunting atmospheric lo-fi mastery',
    '스노우 더 프로덕트': 'world-class speed-rap female rapper switching effortlessly between English and Spanish',
    '스노우더프로덕트': 'world-class speed-rap female rapper switching effortlessly between English and Spanish',
    '가비': 'Latin reggaeton-drill crossover female rapper connecting Caribbean beats with US trap',
    '소피아 블랙': 'R&B-infused female rapper riding 808 glide bass drill with smooth vocal elegance',
    '소피아블랙': 'R&B-infused female rapper riding 808 glide bass drill with smooth vocal elegance',
    '비비 부렐리': 'hit-songwriter female rapper with raw soulful trap vocals and creative genius',
    '비비부렐리': 'hit-songwriter female rapper with raw soulful trap vocals and creative genius',
    '하비아 마이티': 'Polaris-winning Canadian female rapper with hardcore drill mastery and intelligence',
    '하비아마이티': 'Polaris-winning Canadian female rapper with hardcore drill mastery and intelligence',
    '칼리 우치스': 'dreamy Latin-pop female rapper weaving ethereal harmonics with southern trap groove',
    '칼리우치스': 'dreamy Latin-pop female rapper weaving ethereal harmonics with southern trap groove',
    '티나셰': 'lethal off-beat female rapper-singer delivering devastating flow over syncopated hi-hats',
    # ── 🇰🇷 대한민국 최고의 여성 래퍼 (K-HipHop Queens) ──
    '씨엘': '2NE1 hardcore K-pop female rapper who pioneered Billboard and global fashion-music fusion',
    '제시': 'explosive raspy female rapper with southern trap energy and stage-commanding Korean power',
    '치타': 'razor-precise Korean female rapper devouring boom-bap and trap with lethal punchlines',
    '이영지': 'deep baritone-grade Korean female rapper bombing modern drill and trap beats powerfully',
    '미란이': 'addictive melodic Korean female rapper comforting audiences with catchy singing-rap hooks',
    '신스': 'hardcore boom-bap Korean female rapper filling beats with intense life-story narratives',
    '키디비': 'technically versatile Korean female rapper freely riding R&B harmonics and boom-bap rhymes',
    '길미': 'original all-rounder Korean female rapper with rapid-fire delivery and powerful singing',
    '캠보': 'emerging Korean drill female rapper commanding heavy 808 beats with bold presence',
    '전소연': 'genius K-pop producing female rapper who shatters idol limits with trap mastery',
    '유빈': 'charming mid-low Korean female rapper with attractive husky tone on boom-bap beats',
    '지민': 'distinctive ultra-high Korean female rapper crafting ear-catching trap hooks with precision',
    '카디': 'underground Korean female rapper crossing hardcore rock sound with trap production',
    '엑시': 'competition-bred Korean female rapper with solid vocal foundation and rhythmic precision',
    '문별': 'bold thick-toned Korean female rapper anchoring songs with signature mid-low delivery',
    '최예나': 'pop-punk Korean female rapper harmoniously mixing cute trap flow with bubbly energy',
    '리사': 'global Billboard-hitting female rapper with Thai-international swagger and trap queen energy',


    # ═══════════════════════════════════════════════════════════
    # cgo-384: 트로트 남자 보컬 50명
    # ═══════════════════════════════════════════════════════════
    # ── 🎵 전통 트로트 개척자 & 전설 ──
    '남인수': 'crystalline pure-toned male trot tenor revered as the emperor of classic Korean enka',
    '고복수': 'plaintive gentle male trot vocal soothing homesick hearts with simple heartfelt melody',
    '백년설': 'rich earthy male trot baritone comforting working-class souls with rustic warmth',
    '현인': 'pioneering male trot vocalist with signature vibrato who opened Korean popular music',
    '박재홍': 'powerful open-throated male trot singer belting folk sorrows with piercing clarity',
    '진방남': 'sorrowful bending-note master male trot vocalist with deeply mournful delivery',
    '이인권': 'warm low-register male trot vocalist evoking hometown nostalgia with gentle phrasing',
    '도미': 'sophisticated mid-century male trot vocalist bridging modern melody with traditional roots',
    '배호': 'immortal deep baritone male trot vocalist who elevated the genre with noble dignity',
    '한복남': 'humorous witty male trot vocalist who popularized comedic storytelling with catchy groove',
    # ── 👑 트로트 황금기 & 양대산맥 레전드 ──
    '남진': 'Elvis-inspired charismatic male trot vocalist who pioneered dance-trot with stage magnetism',
    '최희준': 'elegant baritone male trot vocalist singing life melancholy with refined literary grace',
    '태진아': 'hook-driven addictive male trot vocalist dominating with catchy refrains and deep emotion',
    '송대관': 'quintessentially Korean optimistic male trot vocalist with earthy rustic warmth and joy',
    '설운도': 'genius singer-songwriter male trot vocalist blending samba and twist rhythms creatively',
    '현철': 'uniquely flavored male trot vocalist with signature nasal bending-note technique mastery',
    '조항조': 'mournful mid-bass male trot vocalist commanding orchestral-scale grand ballad narratives',
    # ── ⚡ 파워 락·댄스 & 뉴웨이브 ──
    '강진': 'rhythmic groove master male trot vocalist who electrified all generations with one hit',
    '박현빈': 'classically trained powerful high-note male trot vocalist who launched power-trot era',
    '신유': 'sweet romantic tenor male trot vocalist with handsome appeal and lyrical sensitivity',
    '진성': 'explosive raspy male trot vocalist with gut-wrenching sorrow and raw emotional power',
    '박상철': 'brass-backed powerhouse male dance-trot vocalist with commanding stage energy',
    '영탁': 'rhythmic all-rounder male trot vocalist with powerful diction and stage-breaking energy',
    '장민호': 'refined groovy male trot vocalist with idol-trained polish and solid vocal technique',
    '이찬원': 'traditional bending-note technician male trot vocalist with earthy fermented-bean voice',
    '김호중': 'operatic tenor male trot vocalist completing orchestral-scale power with massive volume',
    '김수찬': 'flashy showman male new-wave dance-trot vocalist with infectious entertainment energy',
    # ── 🪕 감성 서정 & 포크 융합 ──
    '김희재': 'polished hybrid male trot vocalist blending precise choreography with sweet light tenor',
    '오승근': 'folk-rooted gentle male trot vocalist comforting the nation with plain warm delivery',
    '진시몬': 'folk-ballad optimized male trot vocalist with sweet sentimental melodic craftsmanship',
    '나태주': 'clear steady male pop-trot vocalist hiding deep lyricism behind flashy performance',
    '배일호': 'earthy rustic male trot vocalist combining rural folk sentiment with trot tradition',
    '김용필': 'dignified low-tone male trot vocalist who sings like reciting poetry with gravitas',
    '안성훈': 'pristine clean high-note male healing-ballad trot vocalist with flawless precision',
    '박지현': 'bright energetic male trot vocalist radiating vitality with open airy tenor delivery',
    '최수호': 'minimalist acoustic male trot vocalist with deep resonance on simple folk melodies',
    # ── 🎻 국악 크로스오버 & 시네마틱 ──
    '민수현': 'gentle refined male trot vocalist delivering traditional depth with elegant composure',
    '최재명': 'pansori-infused cinematic male trot vocalist with elaborate melodic architecture and grand projection',
    '정동원': 'prodigy male trot vocalist mastering saxophone to orchestra with epic narrative depth',
    '손태진': 'classical crossover male trot vocalist harmonizing operatic power with grand orchestral scale',
    '최우진': 'stable soaring high-note male trot vocalist riding grand traditional Korean melodies',
    '박서진': 'percussion-performing male trot vocalist with deeply sorrowful han-infused vocal power',
    '강태관': 'pansori-based male trot vocalist showing textbook traditional crossover with profound depth',
    '고영열': 'master-architect male trot vocalist radically mixing pansori, piano and trot harmonics',
    '조명섭': 'bel-canto male trot vocalist creating cinematic time-slip narratives with unique timbre',
    '영광': 'rugged bending-note male trot vocalist cutting through grand horn and string ensembles',
    '남승민': 'cinematic male trot vocalist riding large string sections with emotional stability and depth',

    # ═══════════════════════════════════════════════════════════
    # cgo-384: 트로트 여자 보컬 50명
    # ═══════════════════════════════════════════════════════════
    # ── 🎵 전통 트로트 여류 전설 ──
    '황금심': 'crystalline nightingale female trot vocalist who dominated early classic Korean trot',
    '이난영': 'legendary nasal-melody female trot vocalist who comforted a colonized nation with sorrow',
    '심연옥': 'deep resonant female trot vocalist who tenderly soothed wartime refugees with warmth',
    '박재란': 'brilliant nightingale female trot vocalist celebrated as the golden voice of the 50s-60s',
    '백설희': 'hauntingly beautiful female trot vocalist who captured Korean han in exquisite melody',
    '박애경': 'legendary harmony female trot vocalist showcasing textbook traditional duet vocal mastery',
    '김향미': 'rustic mid-low bending-note female trot vocalist anchoring legendary harmony foundations',
    '지화자': 'stable powerhouse female trot vocalist with the most reliable pentatonic vocal delivery',
    '안다성': 'elegant 60s female trot vocalist layering sophisticated arrangements over traditional melody',
    '이미자': 'the living goddess of Korean trot elegy with immortal bending-note vocal mythology',
    # ── 👑 트로트 황금기 여제들 ──
    '하춘화': 'textbook female trot vocalist with decades of live performance forging unshakeable technique',
    '김연자': 'enka-trot queen female vocalist who conquered both Japan and Korea with explosive power',
    '김수희': 'powerful pansori-toned female trot vocalist who made the whole nation cry and laugh',
    '주현미': 'pharmacist-turned female trot vocalist with crystalline falsetto and delicate high bending',
    '문희옥': 'textbook traditional female trot vocalist with the most flavorful classic bending delivery',
    '현숙': 'positive upbeat female dance-trot vocalist who pioneered rhythmic trot with infectious joy',
    '최진희': 'pop-ballad crossover female trot vocalist perfectly blending Western harmony with sorrow',
    '방실이': 'powerhouse big-voiced female dance-trot vocalist who pioneered energetic trot showmanship',
    '한혜진': 'husky deep mid-low female trot vocalist adding mature depth to adult contemporary trot',
    '한복희': 'stage-dominating female trot vocalist with addictive groove and magnetic crowd control',
    # ── ⚡ 파워 댄스 & 뉴웨이브 ──
    '장윤정': 'genre-reshaping female trot queen who single-handedly revived trot for a new generation',
    '홍진영': 'cute nasally charming female electronic dance-trot vocalist with modern pop appeal',
    '김혜연': 'powerful venue-shaking female dance-trot vocalist commanding massive event stages',
    '서지오': 'rapper-style female dance-trot vocalist maintaining rock-solid technique through choreography',
    '은가은': 'musical-theater trained female power-trot vocalist with soaring high-note stage presence',
    '황우림': 'idol-trained groovy female new-wave trot vocalist incorporating Latin rhythms creatively',
    '별사랑': 'all-range female trot technician vocalist spanning deep bass to soaring high notes',
    '허찬미': 'idol-crossover female trot vocalist with trendy beat-riding ability and sharp performance',
    '요요미': 'cute bright female trot vocalist dominating highway-groove beats with adorable charm',
    '강혜연': 'girl-group trained female trot vocalist hiding solid traditional vocal power behind charm',
    # ── 🪕 감성 서정 & 포크 융합 ──
    '홍자': 'thick soulful female trot vocalist showing peak sorrowful delivery with gomtang warmth',
    '우연이': 'folk-rock gentle female trot vocalist comforting the nation with plain heartfelt warmth',
    '금잔디': 'highway queen female trot vocalist with tender yet smooth voice soothing working hearts',
    '정다경': 'dancer-trained graceful female trot vocalist with clear deep lyrical vocal delivery',
    '김나희': 'crystal-clear healing female trot vocalist who shattered comedian-to-singer stereotypes',
    '강예슬': 'pure refreshing female trot vocalist providing emotional calm with angelic bright tone',
    '마리아': 'first foreign trot champion female vocalist who mastered Korean bending-note technique',
    '김다현': 'young prodigy female trot vocalist narrating deep life stories with mature emotional arc',
    '김태연': 'pansori-master young female trot vocalist melting fierce traditional soul into acoustic folk',
    '윤태화': 'solid expressive female trot vocalist carrying traditional lyrical depth with steady power',
    # ── 🎻 국악 크로스오버 & 시네마틱 ──
    '송가인': 'Miss Trot champion female vocalist with overwhelming pansori-based power shattering Korean han',
    '양지은': 'pansori-certified female crossover trot vocalist singing miracles over grand orchestrations',
    '홍지윤': 'doll-faced female trot vocalist hiding explosive pansori-scaled cinematic high-note power',
    '김의영': 'spicy capsaicin-sharp female trot vocalist with traditional bending and pansori flair',
    '전유진': 'prodigious genius female cinematic trot vocalist effortlessly riding symphonic waves',
    '오유진': 'gayageum-playing female hybrid trot vocalist bridging traditional Korean music and trot',
    '최향': 'rich-volume female cinematic trot vocalist standing firm within grand horn ensembles',
    '풍금': 'refined female trot vocalist who distills deep traditional han into cinematic film-scale delivery',
    '신미래': 'dreamy atmospheric female trot vocalist reinterpreting 30s-40s Korean folk with ethereal tone',
    '나영': 'explosive next-generation female cinematic trot vocalist with overwhelming pansori projection',

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
    '플로스트라다무스': 'hip-hop trap EDM pioneer, swagger-heavy beats with electronic drops',
    '젯츠 데드': 'old-school dubstep, dark hip-hop bass, Toronto underground bass music',
    '젯츠데드': 'old-school dubstep, dark hip-hop bass, Toronto underground bass music',
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
    '토키몬스타': 'alternative techno beat-making, Grammy-nominated Korean-American electronic genius',
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
    '위프트 크림': 'dark hip-hop trap with cinematic dubstep, aggressive bass queen',
    '위프트크림': 'dark hip-hop trap with cinematic dubstep, aggressive bass queen',
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
    '디제이 미아': 'powerful K-club electro house, Korean club scene dominating big room beats',
    '디제이미아': 'powerful K-club electro house, Korean club scene dominating big room beats',
    '디제이 바나': 'sophisticated progressive house toplines, Korean techno scene technician',
    '디제이바나': 'sophisticated progressive house toplines, Korean techno scene technician',
    '수라': 'trendy neo-dance K-EDM hybrid, flashy showmanship, pan-Asian festival vocal',
    '디제이 소다': 'hybrid future pop bass, global SNS pioneer, Asian festival market leader',
    '디제이소다': 'hybrid future pop bass, global SNS pioneer, Asian festival market leader',
    '디제이 나리': 'solid four-on-the-floor club synths, polished Korean club groove master',
    '디제이나리': 'solid four-on-the-floor club synths, polished Korean club groove master',
    '안예은': 'Korean traditional gugak-techno fusion, unprecedented sonic cultural crossover',
    '선우정아': 'jazz-pop-electronic boundary-breaking, sophisticated harmonic backend vocal',
    '림 킴': 'mysterious Eastern pentatonic dark trap bass, revolutionary K-electronic fusion',
    '림킴': 'mysterious Eastern pentatonic dark trap bass, revolutionary K-electronic fusion',
    '전소연': 'K-pop boundary-breaking, direct EDM house-techno drop production genius',
    '씨엘': 'Daft Punk-style french touch house, iconic global trap rap flow powerhouse',
    '제니': 'trendy electronic pop bass, Coachella main stage K-EDM pop fairy vocal',
    '빌리 에일리시': 'dreamy retro synthwave, cinematic industrial techno, whispery dark vocal',
    '빌리에일리시': 'dreamy retro synthwave, cinematic industrial techno, whispery dark vocal',
}


# ═══ CGO 보컬 믹서 1,000 프리셋 (cgo-390) ═══
# cat: A=강화형 B=충돌형 C=크로스 D=시대믹스
# tag: 사용자에게 표시되는 특성 라벨 (아티스트명 비노출)
# w: 3인 가중치 [%,%,%]
# prompt: Suno API에 전달되는 합성 보컬 프롬프트
VOICE_MIX: Dict[str, dict] = {
    'VM0001': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, gravelly uniquely husky deep male jazz vocal, one, heavy gravelly vocal'},
    'VM0002': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, laid-back mellow baritone, polished swinging male'},
    'VM0003': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, sticky deep husky Korean female hip-hop R&B, immortal deep baritone male'},
    'VM0004': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, sticky deep husky Korean female hip-hop R&B, smoky low alto'},
    'VM0005': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'husky deep mid-low female trot vocalist adding mature depth to adult contemporary trot, gravelly uniquely husky deep male jazz vocal, one, pop-future bass fairy,'},
    'VM0006': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, sorrowful bending-note master, bold deep contralto'},
    'VM0007': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, charming deep baritone, velvety deep crooning'},
    'VM0008': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'immortal deep baritone male trot vocalist who elevated the genre with noble dignity, mellow melodic male rapper with addictive hooks and, heavy gravelly vocal'},
    'VM0009': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, mellow melodic male rapper with addictive hooks and, polished swinging male'},
    'VM0010': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'dark European drill female rapper commanding heavy trap, storytelling piano male vocal with warm gritty, slow heavyweight UK underground'},
    'VM0011': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, deep baritone male alternative rock vocal, brass-backed powerhouse male'},
    'VM0012': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'luxurious deep soulful baritone with rich gospel-touched warm resonance, dark European drill female rapper commanding heavy trap, barefoot diva, deeply appealing'},
    'VM0013': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, heavyweight drum-and-bass male, intimate whispery'},
    'VM0014': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'deep heavy contralto female vocal singing the, deep baritone male alternative rock vocal, commanding hip-hop soul'},
    'VM0015': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'barefoot diva, deeply appealing Korean female vocal drawn from the depths of the heart, commanding hip-hop soul, storytelling piano male'},
    'VM0016': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, brass-backed powerhouse male dance-trot vocalist with, heavyweight drum-and-bass male'},
    'VM0017': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'deep-voiced southern female rapper with heavy 808 impact and raw visceral power, slow heavyweight UK underground, deep velvety'},
    'VM0018': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'husky theatrical baritone with bold unique projection and dramatic flair, most sophisticated calm sensual mid-low female, relaxed mellow baritone'},
    'VM0019': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, velvety deep crooning, clear steady male'},
    'VM0020': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'husky theatrical baritone with bold unique projection and dramatic flair, pop-future bass fairy,, bold deep contralto'},
    'VM0021': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep-voiced male rapper-producer who powered Death Row Records golden era sound, deep heavy charismatic low male vocal, commanding hip-hop soul'},
    'VM0022': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavyweight commanding male rapper with flawless flow and deep groove mastery, bold thick-toned Korean female rapper anchoring songs, husky theatrical baritone'},
    'VM0023': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, heavyweight commanding male, rustic mid-low bending-note'},
    'VM0024': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'brass-backed powerhouse male dance-trot vocalist with commanding stage energy, resonant deep baritone with dramatic anthemic, honest deep tenor'},
    'VM0025': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'whispery intimate ASMR vocal with dark, pop-future bass fairy, AlunaGeorge vocal pixie,, refined female trot vocalist'},
    'VM0026': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'immortal deep baritone male trot vocalist who elevated the genre with noble dignity, bold thick-toned Korean female rapper anchoring songs, heavyweight drum-and-bass male'},
    'VM0027': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'laid-back mellow baritone with serene breezy minimal acoustic calm, all-range female trot technician vocalist spanning deep bass, rustic mid-low bending-note'},
    'VM0028': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'deep literary lyrical male Korean pop ballad vocal with quiet resonance, timeless elegant male, suave romantic baritone'},
    'VM0029': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, immortal deep baritone male, deep heavy contralto'},
    'VM0030': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'dark European drill female rapper commanding heavy trap, clear steady male pop-trot vocalist hiding deep, young prodigy female trot'},
    'VM0031': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, theatrical mysterious mid-low, timeless elegant male'},
    'VM0032': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'refined female trot vocalist who distills deep traditional han into cinematic film-scale delivery, husky theatrical baritone with bold unique, mellow melodic male rapper'},
    'VM0033': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, elegant baritone male trot, velvety deep crooning'},
    'VM0034': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'charming deep baritone male vocal, the king of rock and roll, honest deep tenor with raw sincerity and, husky deep mid-low female'},
    'VM0035': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'husky deep mid-low female trot vocalist adding mature depth to adult contemporary trot, polished swinging male, stable rich baritone'},
    'VM0036': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'husky deep mid-low female trot vocalist adding mature, bold theatrical baritone with brassy big, suave romantic baritone'},
    'VM0037': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, heavy dubstep-trap hybrid, dark aggressive bass,, young prodigy female trot'},
    'VM0038': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep-voiced male rapper-producer who powered Death Row Records golden era sound, percussion-performing male trot vocalist with deeply, smooth classic'},
    'VM0039': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavyweight drum-and-bass male rapper with signature UK grime flow mastery, deep soul-laden mezzo-alto with mature tone, commanding dramatic diva'},
    'VM0040': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, barefoot diva, deeply appealing Korean female vocal drawn from, percussion-performing male trot'},
    'VM0041': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'refined female trot vocalist who distills deep traditional han into cinematic film-scale delivery, commanding heavy baritone with powerful sensual, dignified low-tone male trot'},
    'VM0042': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'relaxed mellow baritone with comfortable lush string-backed vocal ease, honest deep tenor with raw sincerity and, young prodigy female trot'},
    'VM0043': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'immortal deep baritone male trot vocalist who elevated the genre with noble dignity, smoky low alto, heavy dubstep-trap hybrid,'},
    'VM0044': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'bold theatrical baritone with brassy big band showman vocal energy, deep soul-laden mezzo-alto, bold thick-toned Korean'},
    'VM0045': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep-voiced southern female rapper with heavy 808 impact and raw visceral power, hook-driven addictive male trot, dancer-trained graceful female'},
    'VM0046': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'luxurious deep soulful baritone with rich gospel-touched warm resonance, whispery intimate ASMR vocal with dark, bold deep contralto'},
    'VM0047': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'mellow melodic male rapper with addictive hooks and relaxed hybrid ballad delivery, smooth classic baritone male crooner, dreamy atmospheric'},
    'VM0048': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, charming deep baritone, most sophisticated calm'},
    'VM0049': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, commanding hip-hop soul alto with passionate, deep heavy contralto'},
    'VM0050': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'heavyweight deep-bass male rapper dominating grandiose trap beats with boss authority, most sophisticated calm sensual mid-low female, all-range female trot technician'},
    'VM0051': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, percussion-performing male trot vocalist with deeply, smoky low alto'},
    'VM0052': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'commanding heavy baritone with powerful sensual deep soul growl, dignified low-tone male trot, whispery intimate ASMR'},
    'VM0053': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'rich earthy male trot baritone comforting working-class souls with rustic warmth, intimate whispery baritone with atmospheric, storytelling piano male'},
    'VM0054': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'elegant baritone male trot vocalist singing life melancholy with refined literary grace, slow heavyweight UK underground, immortal deep baritone male'},
    'VM0055': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'clear steady male pop-trot vocalist hiding deep lyricism behind flashy performance, deep literary lyrical male Korean pop ballad, storytelling piano male'},
    'VM0056': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'bold thick-toned Korean female rapper anchoring songs with signature mid-low delivery, mellow melodic male rapper with addictive hooks and, polished swinging male'},
    'VM0057': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, refined female trot vocalist who distills deep traditional, mournful mid-bass male'},
    'VM0058': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'immortal deep baritone male trot vocalist who elevated the genre with noble dignity, radiant smooth tenor, heavy gravelly vocal'},
    'VM0059': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, most sophisticated calm sensual mid-low female, hook-driven addictive male trot'},
    'VM0060': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, deep literary lyrical male Korean pop ballad, heavy dubstep-trap hybrid,'},
    'VM0061': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'sticky deep husky Korean female hip-hop R&B vocal at the pinnacle, commanding dramatic diva, relaxed mellow baritone'},
    'VM0062': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'percussion-performing male trot vocalist with deeply sorrowful han-infused vocal power, minimalist acoustic male trot, smoky low alto'},
    'VM0063': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, relaxed mellow baritone with comfortable lush, dark European drill female'},
    'VM0064': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'R&B-infused female rapper riding 808 glide bass drill, honest deep tenor with raw sincerity and, brass-backed powerhouse male'},
    'VM0065': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'immortal deep baritone male trot vocalist who elevated the genre with noble dignity, pop-future bass fairy, AlunaGeorge vocal pixie,, percussion-performing male trot'},
    'VM0066': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, romantic aged baritone with weathered folk, smoky low alto'},
    'VM0067': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, deep heavy charismatic low male vocal, heavyweight commanding male'},
    'VM0068': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep literary lyrical male Korean pop ballad vocal with quiet resonance, minimalist acoustic male trot vocalist with deep resonance, husky deep mid-low female'},
    'VM0069': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, deep heavy charismatic low male vocal, all-range female trot technician'},
    'VM0070': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, deep heavy contralto, commanding dramatic diva'},
    'VM0071': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'luxurious deep soulful baritone with rich gospel-touched warm resonance, whispery intimate ASMR vocal with dark, deep baritone male'},
    'VM0072': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'barefoot diva, deeply appealing Korean female vocal drawn from, sorrowful bending-note master male trot vocalist, dancer-trained graceful female'},
    'VM0073': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, deep velvety contralto with smoldering, heavyweight commanding male'},
    'VM0074': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, emerging Korean drill female, commanding heavy baritone'},
    'VM0075': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, brass-backed powerhouse male dance-trot vocalist with, heavyweight commanding male'},
    'VM0076': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, brass-backed powerhouse male dance-trot vocalist with, storytelling piano male'},
    'VM0077': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'refined female trot vocalist who distills deep traditional, smoky low alto with intimate atmospheric, husky deep mid-low female'},
    'VM0078': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'commanding heavy baritone with powerful sensual deep soul growl, thunderous deep-cave male rapper who exploded Brooklyn drill, deep heavy contralto'},
    'VM0079': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep-voiced southern female rapper with heavy 808 impact and raw visceral power, rich deep mid-low, velvety deep crooning'},
    'VM0080': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, stable rich baritone with sweeping epic phrasing, young prodigy female trot'},
    'VM0081': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, stable rich baritone with sweeping epic phrasing, polished swinging male'},
    'VM0082': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, timeless elegant male, polished swinging male'},
    'VM0083': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'dreamy atmospheric electronic vocal, deep, suave romantic baritone with elegant continental, husky theatrical baritone'},
    'VM0084': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep velvety contralto with smoldering sultry low-register warmth, mellow melodic male rapper with addictive hooks and, honest deep tenor'},
    'VM0085': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'percussion-performing male trot vocalist with deeply sorrowful han-infused vocal power, deep grand baritone with rich harmonic resonance, deep literary lyrical'},
    'VM0086': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'immortal deep baritone male trot vocalist who elevated the genre with noble dignity, heavy gravelly vocal with raw hard-rock intensity, deep baritone male'},
    'VM0087': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'heavyweight deep-bass male rapper dominating grandiose trap beats with boss authority, commanding heavy baritone with powerful sensual, honest deep tenor'},
    'VM0088': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'hook-driven addictive male trot vocalist dominating with catchy refrains and deep emotion, dreamy atmospheric electronic vocal, deep, immortal deep baritone male'},
    'VM0089': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, timeless elegant male, most sophisticated calm'},
    'VM0090': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'suave romantic baritone with elegant continental, heavy dubstep-trap hybrid, dark aggressive bass,, sticky deep neo-soul'},
    'VM0091': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, resonant deep baritone with dramatic anthemic, elegant baritone male trot'},
    'VM0092': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, deep soul-laden mezzo-alto with mature tone, heavyweight drum-and-bass male'},
    'VM0093': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'slow heavyweight UK underground male rapper with iconic deep bass flow delivery, deep velvety, whispery intimate ASMR'},
    'VM0094': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, slow heavyweight UK underground, pop-future bass fairy,'},
    'VM0095': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, commanding dramatic diva vocal with explosive, velvety deep crooning'},
    'VM0096': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'rich earthy male trot baritone comforting working-class souls with rustic warmth, gravelly uniquely husky deep, warm charismatic'},
    'VM0097': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'heavyweight deep-bass male rapper dominating grandiose trap, rich earthy male trot baritone comforting working-class, whispery intimate ASMR'},
    'VM0098': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, polished swinging male crooner vocal with, dark European drill female'},
    'VM0099': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'dignified low-tone male trot vocalist who sings like reciting poetry with gravitas, bold deep contralto, bold theatrical baritone'},
    'VM0100': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'thunderous deep-cave male rapper who exploded Brooklyn drill onto the global stage, deep soul-laden mezzo-alto with mature tone, commanding hip-hop soul'},
    'VM0101': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one, intimate whispery baritone with atmospheric, bold thick-toned Korean'},
    'VM0102': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, deep heavy contralto, thunderous deep-cave male rapper'},
    'VM0103': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'all-range female trot technician vocalist spanning deep bass, bold thick-toned Korean female rapper anchoring songs, refined female trot vocalist'},
    'VM0104': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, timeless elegant male jazz crooner vocal, honest deep tenor'},
    'VM0105': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, sticky deep neo-soul male vocal that, polished swinging male'},
    'VM0106': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'elegant baritone male trot vocalist singing life melancholy with refined literary grace, thunderous deep-cave male rapper, heavyweight commanding male'},
    'VM0107': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'all-range female trot technician vocalist spanning deep bass to soaring high notes, timeless elegant male jazz crooner vocal, R&B-infused female rapper riding'},
    'VM0108': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, deep grand baritone, deep baritone male'},
    'VM0109': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'elegant baritone male trot vocalist singing life melancholy with refined literary grace, charming deep baritone male vocal, the king, dancer-trained graceful female'},
    'VM0110': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'bold theatrical baritone with brassy big band showman vocal energy, heavyweight drum-and-bass male rapper with signature, elegant baritone male trot'},
    'VM0111': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavyweight deep-bass male rapper dominating grandiose trap beats with boss authority, heavyweight commanding male, luxurious deep soulful'},
    'VM0112': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, husky deep mid-low female, deep soul-laden mezzo-alto'},
    'VM0113': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'rich earthy male trot baritone comforting working-class souls with rustic warmth, deep-voiced male rapper-producer, bold thick-toned Korean'},
    'VM0114': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, dreamy atmospheric electronic vocal, deep, commanding heavy baritone'},
    'VM0115': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'sorrowful bending-note master male trot vocalist with deeply mournful delivery, dark European drill female rapper commanding heavy trap, warm charismatic'},
    'VM0116': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, bold thick-toned Korean female rapper anchoring songs, dreamy atmospheric'},
    'VM0117': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, rich deep mid-low, deep velvety'},
    'VM0118': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep-voiced male rapper-producer who powered Death Row Records golden era sound, clear steady male, hook-driven addictive male trot'},
    'VM0119': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, all-range female trot technician, dignified low-tone male trot'},
    'VM0120': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'relaxed mellow baritone with comfortable lush string-backed vocal ease, barefoot diva, deeply appealing Korean female vocal drawn from, rich deep mid-low'},
    'VM0121': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavyweight deep-bass male rapper dominating grandiose trap beats with boss authority, sticky deep husky, commanding dramatic diva'},
    'VM0122': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'deep soul-laden mezzo-alto with mature tone and moody R&B depth, bold deep contralto with distinctive vibrato, bold theatrical baritone'},
    'VM0123': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, deep baritone male, whispery intimate ASMR'},
    'VM0124': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'most sophisticated calm sensual mid-low female vocal with luxury tone, brass-backed powerhouse male, sorrowful bending-note master'},
    'VM0125': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, R&B-infused female rapper riding, rustic mid-low bending-note'},
    'VM0126': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, resonant deep baritone with dramatic anthemic, charming deep baritone'},
    'VM0127': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'immortal deep baritone male trot vocalist who elevated the genre with noble dignity, bold deep contralto, warm charismatic'},
    'VM0128': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'dancer-trained graceful female trot vocalist with clear deep lyrical vocal delivery, clear steady male pop-trot vocalist hiding deep, deep literary lyrical'},
    'VM0129': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, luxurious deep soulful baritone with rich, brass-backed powerhouse male'},
    'VM0130': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'clear steady male pop-trot vocalist hiding deep lyricism behind flashy performance, young prodigy female trot vocalist narrating deep life, pop-future bass fairy,'},
    'VM0131': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'heavyweight commanding male rapper with flawless flow and deep groove mastery, hook-driven addictive male trot, husky deep mid-low female'},
    'VM0132': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, mournful mid-bass male, velvety deep crooning'},
    'VM0133': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'dignified low-tone male trot vocalist who sings like reciting poetry with gravitas, mournful mid-bass male trot vocalist commanding, honest deep tenor'},
    'VM0134': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, dancer-trained graceful female, percussion-performing male trot'},
    'VM0135': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, heavy gravelly vocal with raw hard-rock intensity, hook-driven addictive male trot'},
    'VM0136': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'rich deep mid-low voice with velvety soul, commanding dramatic diva vocal with explosive, husky theatrical baritone'},
    'VM0137': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'heavyweight commanding male rapper with flawless flow and deep groove mastery, heavy dubstep-trap hybrid, dark aggressive bass,, barefoot diva, deeply appealing'},
    'VM0138': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, commanding hip-hop soul, R&B-infused female rapper riding'},
    'VM0139': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, smoky low alto, deep heavy contralto'},
    'VM0140': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'rich earthy male trot baritone comforting working-class souls with rustic warmth, heavyweight deep-bass male, heavy dubstep-trap hybrid,'},
    'VM0141': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'sorrowful bending-note master male trot vocalist with deeply mournful delivery, deep heavy charismatic, deep velvety'},
    'VM0142': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'R&B-infused female rapper riding 808 glide bass drill with smooth vocal elegance, smooth classic, dark European drill female'},
    'VM0143': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'charming deep baritone male vocal, the king of rock and roll, slow heavyweight UK underground, stable rich baritone'},
    'VM0144': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'sorrowful bending-note master male trot vocalist with deeply mournful delivery, refined female trot vocalist who distills deep traditional, heavy dubstep-trap hybrid,'},
    'VM0145': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, percussion-performing male trot, husky theatrical baritone'},
    'VM0146': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'percussion-performing male trot vocalist with deeply, gravelly uniquely husky deep male jazz vocal, one, R&B-infused female rapper riding'},
    'VM0147': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'bold thick-toned Korean female rapper anchoring songs with signature mid-low delivery, relaxed mellow baritone with comfortable lush, clear steady male'},
    'VM0148': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'refined female trot vocalist who distills deep traditional han into cinematic film-scale delivery, gravelly uniquely husky deep male jazz vocal, one, emerging Korean drill female'},
    'VM0149': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, barefoot diva, deeply appealing Korean female vocal drawn from, immortal deep baritone male'},
    'VM0150': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'clear steady male pop-trot vocalist hiding deep, radiant smooth tenor with luminous Latin, deep literary lyrical'},
    'VM0151': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'dark European drill female rapper commanding heavy trap beats with ominous presence, deep-voiced male rapper-producer, smoky low alto'},
    'VM0152': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'deep-voiced male rapper-producer who powered Death Row, mellow melodic male rapper with addictive hooks and, dark European drill female'},
    'VM0153': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'sorrowful bending-note master male trot vocalist with deeply mournful delivery, deep grand baritone, barefoot diva, deeply appealing'},
    'VM0154': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'smooth classic baritone male crooner, heavy dubstep-trap hybrid, dark aggressive bass,, deep literary lyrical'},
    'VM0155': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, deep heavy contralto female vocal singing the, deep baritone male'},
    'VM0156': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, bold theatrical baritone with brassy big, smooth classic'},
    'VM0157': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, commanding dramatic diva, commanding heavy baritone'},
    'VM0158': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, smoky low alto, clear steady male'},
    'VM0159': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, deep-voiced southern female rapper with heavy 808 impact, relaxed mellow baritone'},
    'VM0160': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, velvety deep crooning baritone with effortless, deep soul-laden mezzo-alto'},
    'VM0161': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, deep soul-laden mezzo-alto, dark European drill female'},
    'VM0162': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'bold theatrical baritone with brassy big band showman vocal energy, most sophisticated calm sensual mid-low female, whispery intimate ASMR'},
    'VM0163': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'stable rich baritone with sweeping epic phrasing, luxurious deep soulful baritone with rich, deep baritone male'},
    'VM0164': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'refined female trot vocalist who distills deep traditional han into cinematic film-scale delivery, immortal deep baritone male trot vocalist who elevated, deep-voiced southern female rapper'},
    'VM0165': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'husky deep mid-low female trot vocalist adding mature depth to adult contemporary trot, mellow melodic male rapper, theatrical mysterious mid-low'},
    'VM0166': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'percussion-performing male trot vocalist with deeply, polished swinging male crooner vocal with, elegant baritone male trot'},
    'VM0167': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'brass-backed powerhouse male dance-trot vocalist with commanding stage energy, suave romantic baritone with elegant continental, elegant baritone male trot'},
    'VM0168': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'dancer-trained graceful female trot vocalist with clear deep lyrical vocal delivery, husky theatrical baritone, refined female trot vocalist'},
    'VM0169': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'clear steady male pop-trot vocalist hiding deep, elegant baritone male trot vocalist singing life melancholy, husky theatrical baritone'},
    'VM0170': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, sticky deep husky, heavy gravelly vocal'},
    'VM0171': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'whispery intimate ASMR vocal with dark, clear steady male pop-trot vocalist hiding deep, dark European drill female'},
    'VM0172': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, rustic mid-low bending-note female trot vocalist, deep-voiced male rapper-producer'},
    'VM0173': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'percussion-performing male trot vocalist with deeply, smoky low alto with intimate atmospheric, slow heavyweight UK underground'},
    'VM0174': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'mellow melodic male rapper with addictive hooks and relaxed hybrid ballad delivery, heavy gravelly vocal with raw hard-rock intensity, pop-future bass fairy,'},
    'VM0175': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'dancer-trained graceful female trot vocalist with clear deep lyrical vocal delivery, intimate whispery baritone with atmospheric, barefoot diva, deeply appealing'},
    'VM0176': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'rich earthy male trot baritone comforting working-class souls with rustic warmth, percussion-performing male trot vocalist with deeply, pop-future bass fairy,'},
    'VM0177': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'dignified low-tone male trot vocalist who sings like reciting poetry with gravitas, percussion-performing male trot vocalist with deeply, mournful mid-bass male'},
    'VM0178': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'sticky deep husky Korean female hip-hop R&B vocal at the pinnacle, romantic aged baritone, bold deep contralto'},
    'VM0179': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, deep soul-laden mezzo-alto, deep-voiced southern female rapper'},
    'VM0180': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, young prodigy female trot, thunderous deep-cave male rapper'},
    'VM0181': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'dreamy atmospheric electronic vocal, deep lingering emotional tone, heavyweight deep-bass male, slow heavyweight UK underground'},
    'VM0182': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'brass-backed powerhouse male dance-trot vocalist with, husky theatrical baritone with bold unique, deep soul-laden mezzo-alto'},
    'VM0183': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'commanding hip-hop soul alto with passionate, luxurious deep soulful baritone with rich, intimate whispery'},
    'VM0184': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'luxurious deep soulful baritone with rich gospel-touched warm resonance, refined female trot vocalist who distills deep traditional, radiant smooth tenor'},
    'VM0185': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, minimalist acoustic male trot, heavy gravelly vocal'},
    'VM0186': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'young prodigy female trot vocalist narrating deep life stories with mature emotional arc, rich deep mid-low voice with velvety soul, storytelling piano male'},
    'VM0187': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, commanding hip-hop soul alto with passionate, heavyweight drum-and-bass male'},
    'VM0188': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, theatrical mysterious mid-low male vocal with, refined female trot vocalist'},
    'VM0189': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, deep baritone male alternative rock vocal, deep literary lyrical'},
    'VM0190': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, commanding heavy baritone with powerful sensual, percussion-performing male trot'},
    'VM0191': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, rich deep mid-low, dignified low-tone male trot'},
    'VM0192': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'hook-driven addictive male trot vocalist dominating with catchy refrains and deep emotion, mellow melodic male rapper with addictive hooks and, radiant smooth tenor'},
    'VM0193': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, minimalist acoustic male trot vocalist with deep resonance, timeless elegant male'},
    'VM0194': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'bold thick-toned Korean female rapper anchoring songs with signature mid-low delivery, percussion-performing male trot vocalist with deeply, most sophisticated calm'},
    'VM0195': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'deep literary lyrical male Korean pop ballad vocal with quiet resonance, suave romantic baritone with elegant continental, stable rich baritone'},
    'VM0196': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, luxurious deep soulful, pop-future bass fairy,'},
    'VM0197': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, dancer-trained graceful female, dark European drill female'},
    'VM0198': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'hook-driven addictive male trot vocalist dominating with catchy, stable rich baritone with sweeping epic phrasing, immortal deep baritone male'},
    'VM0199': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, storytelling piano male, deep velvety'},
    'VM0200': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, honest deep tenor with raw sincerity and, barefoot diva, deeply appealing'},
    'VM0201': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, warm charismatic baritone with calypso-tinged, dreamy atmospheric'},
    'VM0202': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, charming deep baritone, rich earthy male'},
    'VM0203': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'dignified low-tone male trot vocalist who sings like reciting poetry with gravitas, romantic aged baritone with weathered folk, sticky deep neo-soul'},
    'VM0204': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, brass-backed powerhouse male, most sophisticated calm'},
    'VM0205': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'commanding hip-hop soul alto with passionate 90s ballad grit, honest deep tenor, bold theatrical baritone'},
    'VM0206': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, heavy dubstep-trap hybrid,, bold deep contralto'},
    'VM0207': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, husky theatrical baritone with bold unique, resonant deep baritone'},
    'VM0208': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, relaxed mellow baritone with comfortable lush, deep soul-laden mezzo-alto'},
    'VM0209': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, hook-driven addictive male trot, luxurious deep soulful'},
    'VM0210': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'stable rich baritone with sweeping epic phrasing and cinematic vocal depth, bold thick-toned Korean female rapper anchoring songs, husky theatrical baritone'},
    'VM0211': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, rustic mid-low bending-note female trot vocalist, hook-driven addictive male trot'},
    'VM0212': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, elegant baritone male trot vocalist singing life melancholy, deep-voiced male rapper-producer'},
    'VM0213': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, gravelly uniquely husky deep male jazz vocal, one, sticky deep husky'},
    'VM0214': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'smooth classic baritone male crooner jazz pop vocal, husky theatrical baritone with bold unique, deep baritone male'},
    'VM0215': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'dignified low-tone male trot vocalist who sings like, dancer-trained graceful female trot vocalist with clear, hook-driven addictive male trot'},
    'VM0216': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, elegant baritone male trot vocalist singing life melancholy, percussion-performing male trot'},
    'VM0217': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'dignified low-tone male trot vocalist who sings like, thunderous deep-cave male rapper who exploded Brooklyn drill, rich earthy male'},
    'VM0218': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'deep-voiced male rapper-producer who powered Death Row, polished swinging male crooner vocal with, velvety deep crooning'},
    'VM0219': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'rich earthy male trot baritone comforting working-class souls with rustic warmth, rustic mid-low bending-note, rich deep mid-low'},
    'VM0220': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'bold thick-toned Korean female rapper anchoring songs with signature mid-low delivery, dreamy atmospheric, sticky deep neo-soul'},
    'VM0221': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'sticky deep husky Korean female hip-hop R&B, heavyweight commanding male rapper with flawless flow, deep grand baritone'},
    'VM0222': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, bold thick-toned Korean female rapper anchoring songs, all-range female trot technician'},
    'VM0223': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, clear steady male pop-trot vocalist hiding deep, rich earthy male'},
    'VM0224': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, dancer-trained graceful female trot vocalist with clear, heavyweight drum-and-bass male'},
    'VM0225': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'rustic mid-low bending-note female trot vocalist anchoring legendary harmony foundations, hook-driven addictive male trot vocalist dominating with catchy, deep literary lyrical'},
    'VM0226': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'husky theatrical baritone with bold unique, dreamy atmospheric electronic vocal, deep, commanding heavy baritone'},
    'VM0227': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'hook-driven addictive male trot vocalist dominating with catchy refrains and deep emotion, percussion-performing male trot vocalist with deeply, luxurious deep soulful'},
    'VM0228': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, velvety deep crooning baritone with effortless, deep soul-laden mezzo-alto'},
    'VM0229': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, dreamy atmospheric electronic vocal, deep, rich earthy male'},
    'VM0230': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, timeless elegant male jazz crooner vocal, clear steady male'},
    'VM0231': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, rustic mid-low bending-note female trot vocalist, romantic aged baritone'},
    'VM0232': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'husky theatrical baritone with bold unique projection and dramatic flair, sorrowful bending-note master, minimalist acoustic male trot'},
    'VM0233': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'whispery intimate ASMR vocal with dark minimal cinematic atmosphere, brass-backed powerhouse male dance-trot vocalist with, dreamy atmospheric'},
    'VM0234': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'R&B-infused female rapper riding 808 glide bass drill with smooth vocal elegance, intimate whispery, all-range female trot technician'},
    'VM0235': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'laid-back mellow baritone with serene breezy minimal acoustic calm, deep heavy contralto, heavy gravelly vocal'},
    'VM0236': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'mellow melodic male rapper with addictive hooks and relaxed hybrid ballad delivery, husky deep mid-low female trot vocalist adding mature, husky theatrical baritone'},
    'VM0237': {'cat':'A','tag':'중저음 매력','w':[50, 30, 20],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, stable rich baritone with sweeping epic phrasing, percussion-performing male trot'},
    'VM0238': {'cat':'A','tag':'중저음 매력','w':[40, 40, 20],'prompt':'mournful mid-bass male trot vocalist commanding, refined female trot vocalist who distills deep traditional, minimalist acoustic male trot'},
    'VM0239': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'heavyweight deep-bass male rapper dominating grandiose trap beats with boss authority, dreamy atmospheric, heavy dubstep-trap hybrid,'},
    'VM0240': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'brass-backed powerhouse male dance-trot vocalist with commanding stage energy, deep-voiced male rapper-producer, intimate whispery'},
    'VM0241': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, R&B-infused female rapper riding, commanding hip-hop soul'},
    'VM0242': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, heavyweight commanding male, dignified low-tone male trot'},
    'VM0243': {'cat':'A','tag':'중저음 매력','w':[70, 20, 10],'prompt':'dark European drill female rapper commanding heavy trap beats with ominous presence, R&B-infused female rapper riding, suave romantic baritone'},
    'VM0244': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'sticky deep neo-soul male vocal that melts the heart, commanding heavy baritone with powerful sensual, husky deep mid-low female'},
    'VM0245': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavyweight deep-bass male rapper dominating grandiose trap beats with boss authority, storytelling piano male vocal with warm gritty, hook-driven addictive male trot'},
    'VM0246': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'young prodigy female trot vocalist narrating deep life stories with mature emotional arc, brass-backed powerhouse male, theatrical mysterious mid-low'},
    'VM0247': {'cat':'A','tag':'중저음 매력','w':[60, 30, 10],'prompt':'resonant deep baritone with dramatic anthemic folk pop vocal projection, rich earthy male trot baritone comforting working-class, suave romantic baritone'},
    'VM0248': {'cat':'A','tag':'중저음 매력','w':[50, 40, 10],'prompt':'dignified low-tone male trot vocalist who sings like reciting poetry with gravitas, honest deep tenor with raw sincerity and, mellow melodic male rapper'},
    'VM0249': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'emerging Korean drill female rapper commanding heavy 808 beats with bold presence, barefoot diva, deeply appealing, heavyweight deep-bass male'},
    'VM0250': {'cat':'A','tag':'중저음 매력','w':[60, 20, 20],'prompt':'refined female trot vocalist who distills deep traditional han into cinematic film-scale delivery, dark European drill female, heavyweight commanding male'},
    'VM0251': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'percussion-performing male trot vocalist with deeply sorrowful han-infused vocal power, beast-like husky male vocal with Korean emotional, globally verified K-pop female'},
    'VM0252': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'soaring rock soprano with legendary high-range stadium power, angsty alternative mezzo with confessional, warm low-register male'},
    'VM0253': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, most destructive female rock vocal in, elegant 60s female'},
    'VM0254': {'cat':'B','tag':'중저음 매력','w':[40, 40, 20],'prompt':'stable rich baritone with sweeping epic phrasing, genius male vocal combining demonic growling, authoritative smooth male'},
    'VM0255': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, ethereal crystalline soprano with gentle, gangster-crew male rappers'},
    'VM0256': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'resonant deep baritone with dramatic anthemic folk pop vocal projection, plaintive soaring falsetto with vulnerable intimate, master-architect male trot'},
    'VM0257': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, raw convulsive gravelly male vocal wringing every, commanding male rapper'},
    'VM0258': {'cat':'B','tag':'압도적 고음','w':[50, 30, 20],'prompt':'perfectionist tenor with powerful live projection and orchestral vocal precision, refreshing bright rock soprano with crisp attack, charming mid-low Korean female'},
    'VM0259': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'resonant deep baritone with dramatic anthemic folk pop vocal projection, gentle meditative tenor, idol-trained groovy female'},
    'VM0260': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'brass-backed powerhouse male dance-trot vocalist with commanding stage energy, rich commanding contralto with majestic, cute warm soprano with'},
    'VM0261': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'thick soulful female trot vocalist showing peak sorrowful delivery with gomtang warmth, warm earthy alto with tender, genius K-pop producing female'},
    'VM0262': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'crystal clear yet steel-strong female belting high vocal filling stadiums, elegant 60s female trot vocalist layering sophisticated, clear pristine soprano'},
    'VM0263': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'immortal deep baritone male trot vocalist who elevated, modern layered R&B alto with dense, pristine classical'},
    'VM0264': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'global trendy tenor with cinematic pop polish and youthful dynamic range, spicy capsaicin-sharp female trot vocalist with traditional, warm intimate male'},
    'VM0265': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'slow heavyweight UK underground male rapper with iconic deep bass flow delivery, massive operatic soprano with stadium-shaking, raw desperate soprano'},
    'VM0266': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'earthy rustic male trot vocalist combining rural folk sentiment with trot tradition, tender longing falsetto carrying sorrowful romantic, explosive female disco'},
    'VM0267': {'cat':'B','tag':'허스키 감성','w':[70, 20, 10],'prompt':'the king of Korean pop, versatile male vocal covering rock ballad and folk, creative fusion soprano, idol-crossover female trot'},
    'VM0268': {'cat':'B','tag':'허스키 감성','w':[50, 30, 20],'prompt':'modern breathy alternative R&B vocal with trendy melodic sensibility, gentle refined male trot vocalist delivering traditional, smooth R&B singing over'},
    'VM0269': {'cat':'B','tag':'압도적 고음','w':[50, 30, 20],'prompt':'romantic emotional tenor with soaring rock-ballad phrasing and soft power, pristine smooth falsetto with effortless high, Latin reggaeton-drill crossover'},
    'VM0270': {'cat':'B','tag':'압도적 고음','w':[60, 20, 20],'prompt':'majestic operatic tenor with soaring classical purity and grand resonance, bright pure indie, dreamy atmospheric house,'},
    'VM0271': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, refreshing clear soprano, authoritative male rapper'},
    'VM0272': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, trendy urban mezzo, Grammy-winning EDM topline'},
    'VM0273': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'heavyweight drum-and-bass male rapper with signature, explosive power from small frame, timeless clear, unique delicate Korean female'},
    'VM0274': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'sorrowful bending-note master male trot vocalist with deeply mournful delivery, sandpaper-rough charming male vocal with loose, punchy dynamic male'},
    'VM0275': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'pioneering male trot vocalist with signature vibrato who, pure crystalline tenor with emotionally transparent, inventive creative female rapper'},
    'VM0276': {'cat':'B','tag':'허스키 감성','w':[70, 20, 10],'prompt':'gritty warm male keyboard-soul vocal with bluesy rasp and heartfelt punch, honest unpretentious warm, mysterious Eastern pentatonic'},
    'VM0277': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'raw desperate soprano with unfiltered emotional intensity and urgent vocal power, warm low-register male trot vocalist evoking hometown, revolutionary UK grime male'},
    'VM0278': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'most sophisticated calm sensual mid-low female vocal with luxury tone, ultimate Korean female, rough raspy alto'},
    'VM0279': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'emerging Korean drill female rapper commanding heavy 808 beats with bold presence, pioneering male trot vocalist, trendy stylish male'},
    'VM0280': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, dance anthem, warm intimate male'},
    'VM0281': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'honest deep tenor with raw sincerity and quietly compelling melodic appeal, explosive raspy male trot, crystal-clear healing female'},
    'VM0282': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, edgy youthful alto with emotional, angelic Irish ensemble'},
    'VM0283': {'cat':'B','tag':'압도적 고음','w':[60, 20, 20],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, inventive jazzy soprano, underground legend male'},
    'VM0284': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'clear steady male pop-trot vocalist hiding deep lyricism behind flashy performance, husky powerful, ethereal breathtaking soprano'},
    'VM0285': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'laid-back mellow baritone with serene breezy, stadium-filling resonant male vocal with, genius sensual male vocal'},
    'VM0286': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, fire-breathing piercing metallic high female, gritty soulful'},
    'VM0287': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'powerful husky high tenor with dramatic intensity and piercing climactic notes, textbook traditional female trot, polished velvety'},
    'VM0288': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, rugged bending-note male trot, radically alternative female'},
    'VM0289': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, bel-canto male trot vocalist creating cinematic time-slip, explosive raspy female rapper'},
    'VM0290': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'gayageum-playing female hybrid trot vocalist bridging traditional, inventive jazzy soprano with playful harmonic twists, fierce female rapper from'},
    'VM0291': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'deep-voiced southern female rapper with heavy 808 impact and raw visceral power, clear earnest tenor, soft dreamy'},
    'VM0292': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'versatile clear tenor with explosive high notes, unique sophisticated neo-soul queen female vocal, sweet lyrical soprano'},
    'VM0293': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'transcendent tenor with flawless breath control, nervous yet beautiful dreamy falsetto male, witty transatlantic female rapper'},
    'VM0294': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'mysterious powerful gothic female rock vocal piercing through dark orchestral sound, textbook traditional female trot, unique delicate Korean female'},
    'VM0295': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'dramatic operatic male vocal with soaring, the king of Korean pop, versatile male vocal, crystalline nightingale female'},
    'VM0296': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'world-class 5-octave powerful Korean female vocal with dramatic high notes and pop diva power, gravelly soulful midrange with Celtic, warm folk acoustic'},
    'VM0297': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'clear earnest tenor with sweeping poetic folk storytelling grandeur, edgy youthful alto with emotional, delicate airy'},
    'VM0298': {'cat':'B','tag':'허스키 감성','w':[50, 30, 20],'prompt':'raw rough soul-shaking male rock vocal with emotional honesty, tender longing falsetto carrying sorrowful romantic, hard-hitting gangster male'},
    'VM0299': {'cat':'B','tag':'압도적 고음','w':[60, 30, 10],'prompt':'majestic operatic tenor with soaring classical purity and grand resonance, pharmacist-turned female trot vocalist with crystalline falsetto, commanding male rapper'},
    'VM0300': {'cat':'B','tag':'중저음 매력','w':[40, 40, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one, deep husky soulful Korean female ballad vocal, rhythmic male pop'},
    'VM0301': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'gritty soulful groove vocal with British white-soul rasp, pure clean soprano capturing quiet depth, trend-setting UK drill'},
    'VM0302': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'warm robust tenor with grand sweeping romantic pop balladry and passion, rough torn raspy, soft breathy warm'},
    'VM0303': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'young prodigy female trot vocalist narrating deep life stories with mature emotional arc, classical soprano, flawless classic female'},
    'VM0304': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'heavyweight drum-and-bass male rapper with signature UK grime flow mastery, doll-faced female trot, bouncy yet heartfelt'},
    'VM0305': {'cat':'B','tag':'압도적 고음','w':[60, 20, 20],'prompt':'raw fragile tenor building from whisper to intense acoustic crescendo, inventive jazzy soprano, authoritative smooth male'},
    'VM0306': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'soulful mezzo-soprano with flawless R&B scale technique and rich harmony, pristine classical crossover soprano with, dreamy Latin-pop female'},
    'VM0307': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'minimalist acoustic male trot vocalist with deep resonance on simple folk melodies, multi-genre soprano with, smooth modern country'},
    'VM0308': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'percussion-performing male trot vocalist with deeply sorrowful han-infused vocal power, raw powerful female rapper conquering Billboard with, quiet warm soothing'},
    'VM0309': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'resonant deep baritone with dramatic anthemic, flawless classic female vocal mastering Broadway and, soft breathy warm'},
    'VM0310': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, pure crystal-clear folk soprano with gentle, historic west-coast crew'},
    'VM0311': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'brass-backed powerhouse male dance-trot vocalist with commanding stage energy, cinematic dubstep, creative fusion soprano'},
    'VM0312': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, clear earnest tenor, silvery celestial country'},
    'VM0313': {'cat':'B','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'world-class soprano with soaring operatic power and pristine cinematic projection, devastating power-ballad soprano, crystal-clear healing female'},
    'VM0314': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'sorrowful bending-note master male trot vocalist with deeply mournful delivery, fire-breathing piercing, globally distinctive soprano'},
    'VM0315': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, genius singer-songwriter male, bright cheerful male'},
    'VM0316': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'deep heavy contralto female vocal singing the Black soul with gravitas, queen of soul, gospel-based explosive powerful female vocal, rebellious melancholic raw retro'},
    'VM0317': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, distinctive nasal indie tenor with quirky charm, deep husky soulful'},
    'VM0318': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'pop-future bass fairy, AlunaGeorge vocal pixie, light sparkling tone, lyrical light tenor with airy French-pop-influenced, cute warm soprano with'},
    'VM0319': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'slow heavyweight UK underground male rapper with iconic, powerfully raspy male rapper with aggressive west-coast, rough torn raspy'},
    'VM0320': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'soaring popera soprano with theatrical cinematic grandeur and power, polished warm soprano with elegant 60s, modern male rapper'},
    'VM0321': {'cat':'B','tag':'압도적 고음','w':[70, 20, 10],'prompt':'agile scatting tenor blending jazz improvisation with smooth pop finesse, bright cheerful male, razor-precise Korean female'},
    'VM0322': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'sad sharp Irish traditional female vocal with sorrowful piercing tone, underground gritty male rapper with raw boom-bap, emotional healing trance,'},
    'VM0323': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, sophisticated mid-range vocal with refined phrasing, flawless crystal clear'},
    'VM0324': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'legendary nasal-melody female trot vocalist who comforted a colonized nation with sorrow, emotional healing trance, angelic vocal melodies,, NYC underground queen'},
    'VM0325': {'cat':'B','tag':'압도적 고음','w':[60, 20, 20],'prompt':'heart-tearing sorrowful explosive male soul vocal, creative fusion soprano, razor-precise Korean female'},
    'VM0326': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, broken sobbing male, bright pure tenor'},
    'VM0327': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'trendy urban mezzo with tension-filled chord sensibility and sultry phrasing, warm versatile mezzo with theatrical, explosive Canadian male'},
    'VM0328': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'ultimate Korean female high-note queen with blade-sharp piercing rapid vocal delivery, pansori-infused cinematic male trot vocalist with elaborate melodic, warm folk acoustic'},
    'VM0329': {'cat':'B','tag':'중저음 매력','w':[50, 40, 10],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, husky soulful female vocal fusing hip-hop and, cute bright female'},
    'VM0330': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'sweeping dramatic tenor with lush symphonic, rugged bending-note male trot vocalist cutting through grand, pristine classical'},
    'VM0331': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'distinctive nasal indie tenor with quirky charm, trendy urban mezzo with tension-filled chord, massive cinematic soprano'},
    'VM0332': {'cat':'B','tag':'허스키 감성','w':[50, 30, 20],'prompt':'rebellious melancholic raw retro soul jazz female vocal, one of a kind tone, angelic fragile yet devastating falsetto male, technically gifted male'},
    'VM0333': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'the king of Korean pop, versatile male vocal, bright pure indie soprano with cheerful Hongdae, pioneering male rapper'},
    'VM0334': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'immortal deep baritone male trot vocalist who elevated the genre with noble dignity, enka-trot queen female vocalist, elegant 60s female'},
    'VM0335': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'smoky low alto with intimate atmospheric jazz vocal phrasing, androgynous cold urban, clear pristine soprano'},
    'VM0336': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, enka-trot queen female vocalist, pure crystalline tenor'},
    'VM0337': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, refined French chanteuse with classic, delicate yodeling folk'},
    'VM0338': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'charming deep baritone male vocal, the king of rock and roll, refined velvety tenor with elegant soaring, bright youthful'},
    'VM0339': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'deep classic husky female vocal with, folk-rooted gentle male trot vocalist comforting the nation, dreamy alternative male'},
    'VM0340': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'luxurious deep soulful baritone with rich gospel-touched warm resonance, genius rhythmic soulful, haunting atmospheric'},
    'VM0341': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'warm charismatic baritone with calypso-tinged folk orchestral humanity, explosive power from small frame, timeless clear, explosive raspy male trot'},
    'VM0342': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'commanding hip-hop soul alto with passionate 90s ballad grit, trance vocal queen,, NYC underground queen'},
    'VM0343': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'rugged male rapper blending gritty tone with gangster balladry and west-coast soul, quiet warm soothing, witty transatlantic female rapper'},
    'VM0344': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'elegant baritone male trot vocalist singing life melancholy with refined literary grace, powerful heartfelt classic pop male vocal, pristine classical'},
    'VM0345': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'globally distinctive soprano with unique nasal R&B color and emotional crack, flawless crystal clear tenor cutting through, legendary Three 6 Mafia'},
    'VM0346': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'sorrowful bending-note master male trot vocalist with deeply mournful delivery, piercing powerful, androgynous cold urban'},
    'VM0347': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep-voiced male rapper-producer who powered Death Row Records golden era sound, clear bright female pop vocal, pharmacist-turned female trot'},
    'VM0348': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'heavyweight drum-and-bass male rapper with signature UK grime flow mastery, clear bright female pop vocal with ultra-high technique, hauntingly beautiful female trot'},
    'VM0349': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'hard-hitting slide-drill male rapper with powerful 808 bass-riding technique, pansori-infused cinematic male trot vocalist with elaborate melodic, pure crystalline tenor'},
    'VM0350': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'deep soul-laden mezzo-alto with mature tone, screaming high tenor with razor-sharp power, refined French'},
    'VM0351': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'raw convulsive gravelly male vocal wringing every note with blues agony, quiet warm soothing, globally acclaimed UK'},
    'VM0352': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'young prodigy female trot vocalist narrating deep life, Miss Trot champion female vocalist with overwhelming pansori-based, airy ethereal male'},
    'VM0353': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'resonant deep baritone with dramatic anthemic folk pop vocal projection, gritty soulful groove vocal with, warm intimate male'},
    'VM0354': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'bold theatrical baritone with brassy big band showman vocal energy, sandpaper-rough charming male vocal with loose, creative fusion soprano'},
    'VM0355': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, transparent fragile male, power pop-rock EDM'},
    'VM0356': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'clear steady male pop-trot vocalist hiding deep lyricism behind flashy performance, original all-rounder Korean, rhythmic powerhouse'},
    'VM0357': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, legendary harmony female, warm folk acoustic'},
    'VM0358': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'stable rich baritone with sweeping epic phrasing, blended operatic tenor ensemble with lush, deep classic husky female'},
    'VM0359': {'cat':'B','tag':'압도적 고음','w':[70, 20, 10],'prompt':'crystal clear yet steel-strong female belting high vocal filling stadiums, intense dramatic, Terror Squad pride female'},
    'VM0360': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'deep heavy charismatic low male vocal, distinctive nasal indie tenor with quirky charm, pure refreshing female trot'},
    'VM0361': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'sticky deep husky Korean female hip-hop R&B vocal at the pinnacle, solid expressive female trot, transparent dewdrop-clear soprano'},
    'VM0362': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'explosive operatic metal male vocal like a human air raid siren, layered multitrack choral vocal creating vast, global EDM hit'},
    'VM0363': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'intimate whispery baritone with atmospheric mysterious cinematic texture, rhythmic powerhouse, microtonal Arab-maqam female'},
    'VM0364': {'cat':'B','tag':'압도적 고음','w':[60, 30, 10],'prompt':'The Voice, perfect female vocal with flawless power pitch and emotion, nervous yet beautiful dreamy falsetto male, polished Atlanta trap'},
    'VM0365': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'emotionally charged soprano with blockbuster string-ballad intensity and raw feeling, street-style diva soprano with massive volume and, relentless rapid-fire male'},
    'VM0366': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'warm charismatic baritone with calypso-tinged folk orchestral humanity, flawless technique male, paradigm-shifting male rapper-producer'},
    'VM0367': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'hook-driven addictive male trot vocalist dominating with catchy, anthem trance, heart-wrenching melodies, powerful emotional, lush romantic male'},
    'VM0368': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'rich deep mid-low voice with velvety soul, traditional bending-note technician male trot vocalist, bright narrative acoustic'},
    'VM0369': {'cat':'B','tag':'중저음 매력','w':[40, 40, 20],'prompt':'deep literary lyrical male Korean pop ballad, devastating power-ballad soprano tearing through lush, funky freewheeling male'},
    'VM0370': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'pop-future bass fairy, AlunaGeorge vocal pixie,, global trendy tenor with cinematic pop polish, warm earthy'},
    'VM0371': {'cat':'B','tag':'중저음 매력','w':[50, 40, 10],'prompt':'suave romantic baritone with elegant continental orchestral pop charm, devastating power-ballad soprano tearing through lush, pioneering male rapper'},
    'VM0372': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'resonant deep baritone with dramatic anthemic folk pop vocal projection, underground gritty male, husky passionate male rapper'},
    'VM0373': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'emerging Korean drill female rapper commanding heavy 808 beats with bold presence, deep husky soulful, historic west-coast crew'},
    'VM0374': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'deep soul-laden mezzo-alto with mature tone, sweeping dramatic tenor with lush symphonic, unique bright indie'},
    'VM0375': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'Korean R&B fairy female vocal with perfect breath control and brilliant melisma, cute warm soprano with, global Billboard-hitting female'},
    'VM0376': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'young prodigy female trot vocalist narrating deep life stories with mature emotional arc, husky soulful female, lethal off-beat female'},
    'VM0377': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'all-range female trot technician vocalist spanning deep bass, versatile clear tenor with explosive high notes, velvety smooth perfect'},
    'VM0378': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, raw angular mezzo, warm versatile'},
    'VM0379': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'easygoing sunny tenor with playful organic folk pop vocal charm, deep classic husky female vocal with, bright cheerful male'},
    'VM0380': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, cinematic dubstep orchestral vocal, movie-score, pristine clean high-note'},
    'VM0381': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'beast-like husky male vocal with Korean emotional sorrow and raw power, serene quiet soprano with gentle indie folk healing, revolutionary UK grime male'},
    'VM0382': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'sweeping dramatic tenor with lush symphonic phrasing and soaring crescendos, raw aching male piano vocal that erupts from, gentle wistful male'},
    'VM0383': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'gritty warm male keyboard-soul vocal with bluesy rasp and heartfelt punch, massive cinematic soprano dominating choir and symphony, rapid-fire versatile female'},
    'VM0384': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'heavyweight drum-and-bass male rapper with signature UK grime flow mastery, feathery high tenor with breezy soft, elegant 60s female'},
    'VM0385': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'mournful mid-bass male trot vocalist commanding orchestral-scale grand ballad narratives, edgy youthful alto with emotional, calm low mid-range'},
    'VM0386': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'dignified low-tone male trot vocalist who sings like reciting poetry with gravitas, deeply emotive soulful, polished velvety'},
    'VM0387': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'heavyweight hardcore boom-bap female rapper with solid, crystalline narrative soprano with warm, French electroclash legend,'},
    'VM0388': {'cat':'B','tag':'허스키 감성','w':[70, 20, 10],'prompt':'explosive raspy male trot vocalist with gut-wrenching sorrow and raw emotional power, ethereal Nordic, whisper-soft literary female rapper'},
    'VM0389': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'emotive building male vocal from restrained verse, deep classic husky female vocal with, celestial melodic bass'},
    'VM0390': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'minimalist acoustic male trot vocalist with deep resonance on simple folk melodies, distinctive nasal indie tenor with quirky charm, natural conversational mid-range'},
    'VM0391': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'dancer-trained graceful female trot vocalist with clear deep lyrical vocal delivery, husky heartbreak-filled Korean female OST, versatile raw male rapper'},
    'VM0392': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'trembling soulful male vocal with aching falsetto and vulnerable emotional depth, bright pure tenor with wholesome nature-inspired, polished Atlanta trap'},
    'VM0393': {'cat':'B','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'hard-hitting slide-drill male rapper with powerful 808 bass-riding technique, husky soulful female, pristine smooth falsetto'},
    'VM0394': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'gravelly uniquely husky deep male jazz vocal, one of a kind tone, bel-canto male trot, angelic fragile yet'},
    'VM0395': {'cat':'B','tag':'압도적 고음','w':[60, 30, 10],'prompt':'distinctive nasal indie tenor with quirky charm and acoustic pop character, airy ethereal falsetto with raw sensual gospel, charismatic bold female'},
    'VM0396': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'bold deep contralto with distinctive vibrato and rock-pop grit, genius male vocal combining demonic growling, quirky playful'},
    'VM0397': {'cat':'B','tag':'허스키 감성','w':[50, 30, 20],'prompt':'devastating power-ballad soprano tearing through lush string arrangements emotionally, deep resonant female trot vocalist who tenderly soothed, hard-hitting gangster male'},
    'VM0398': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, bright energetic male trot vocalist radiating vitality with, minimal clear soprano'},
    'VM0399': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'first foreign trot champion female vocalist who mastered Korean bending-note technique, layered multitrack choral vocal creating vast, France greatest-selling female'},
    'VM0400': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'smooth classic baritone male crooner jazz pop vocal, powerful rich baritone-tenor with sweeping orchestral, sky-high angelic male'},
    'VM0401': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'Belgian festival vocal performance, hype crowd, smooth modern country soprano with dreamy, passionate revolutionary male'},
    'VM0402': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'husky powerful soprano with overwhelming melismatic soul technique, husky heartbreak-filled Korean female OST, angelic fragile yet'},
    'VM0403': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'brass-backed powerhouse male dance-trot vocalist with commanding stage energy, The Voice, perfect female, ice-cold sad rebellious'},
    'VM0404': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'dance anthem powerhouse, Tiesto collaboration, high-energy pop-EDM vocal, clear smooth pure falsetto, world-class speed-rap female'},
    'VM0405': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'delicate lyrical tenor with poetic graceful, street-style diva soprano with massive volume and, nasal high-pitched male'},
    'VM0406': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'charming deep baritone male vocal, the king of rock and roll, romantic emotional tenor with soaring rock-ballad, sophisticated silky'},
    'VM0407': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, raw powerful black-soul-based female belting vocal, hit-songwriter female rapper'},
    'VM0408': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'powerful gritty vocal on hardcore bass, chest-voice distortion ballad power, ethereal Nordic soprano with nature-inspired, Grammy-winning EDM topline'},
    'VM0409': {'cat':'B','tag':'압도적 고음','w':[60, 30, 10],'prompt':'dramatic operatic male vocal with soaring sorrowful high notes, transparent dewdrop-clear soprano with pristine folk, Australian-born female rapper'},
    'VM0410': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'R&B-infused female rapper riding 808 glide bass drill with smooth vocal elegance, genius singer-songwriter male, chart-dominating male rapper-singer'},
    'VM0411': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'slow heavyweight UK underground male rapper with iconic deep bass flow delivery, 5-octave female vocal with dolphin whistle register, clear pristine soprano'},
    'VM0412': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, devastating power-ballad soprano tearing through lush, smooth romantic male rapper'},
    'VM0413': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, devastating power-ballad soprano, highway queen female trot'},
    'VM0414': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'rhythmic all-rounder male trot vocalist with powerful diction and stage-breaking energy, pansori-certified female crossover trot vocalist singing, refreshing clear soprano'},
    'VM0415': {'cat':'B','tag':'허스키 감성','w':[70, 20, 10],'prompt':'legendary nasal sorrowful uniquely toned Korean trot folk female vocal soaking the soul, warm folk acoustic, flashy showman male'},
    'VM0416': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'romantic aged baritone with weathered folk warmth and nostalgic resonance, emotive building male vocal from restrained verse, dreamy sophisticated falsetto'},
    'VM0417': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'heavyweight commanding male rapper with flawless flow and deep groove mastery, soaring popera soprano, soft gentle tenor'},
    'VM0418': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'pansori-master young female trot vocalist melting fierce traditional soul into acoustic folk, honest unpretentious warm, natural nasal-toned mezzo'},
    'VM0419': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'soaring rock soprano with legendary high-range stadium power, husky gravelly male rapper delivering authentic Atlanta, Nordic crystal-clear'},
    'VM0420': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'refined effortless alto with minimal urban folk, bright pure tenor with wholesome nature-inspired, aggressive hard-hitting male'},
    'VM0421': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'bold thick-toned Korean female rapper anchoring songs, heavyweight hardcore boom-bap female rapper with solid, refined groovy male trot'},
    'VM0422': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'musical-theater trained female power-trot vocalist with soaring high-note stage presence, emotional vocal trance, heart-purifying chord sequences,, polished velvety'},
    'VM0423': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'barefoot diva, deeply appealing Korean female vocal drawn from the depths of the heart, cinematic male trot vocalist, ethereal theatrical falsetto'},
    'VM0424': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'soaring male rock ballad vocal with polished, earnest warm tenor with pure heartfelt delivery, rapid-fire technical male'},
    'VM0425': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'barefoot diva, deeply appealing Korean female vocal drawn from, warm classic tenor with smooth legato phrasing, serene quiet soprano with'},
    'VM0426': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'mournful mid-bass male trot vocalist commanding orchestral-scale grand ballad narratives, legendary nasal-melody female trot, feathery soft tender'},
    'VM0427': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'thunderous deep-cave male rapper who exploded Brooklyn drill onto the global stage, original all-rounder Korean, flawless technique male'},
    'VM0428': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'rugged male rapper blending gritty tone with gangster, inventive jazzy soprano with playful harmonic twists, modern boom-bap female rapper'},
    'VM0429': {'cat':'B','tag':'중저음 매력','w':[60, 20, 20],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, saddest tone in jazz, hardcore boom-bap Korean'},
    'VM0430': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'heavyweight drum-and-bass male rapper with signature UK grime flow mastery, doll-faced female trot vocalist hiding explosive, lush romantic male'},
    'VM0431': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'heavyweight hardcore boom-bap female rapper with solid powerful projection and grit, arrogant cynical distinctive male britpop vocal, soft breathy warm'},
    'VM0432': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'elegant refined tenor with classically elevated harmonic vocal phrasing, hit-songwriter female rapper with raw soulful trap, transparent dewdrop-clear soprano'},
    'VM0433': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'suave romantic baritone with elegant continental, powerful open-throated male trot singer belting folk, polished velvety'},
    'VM0434': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'most sophisticated calm sensual mid-low female vocal with luxury tone, gentle acoustic to soaring, pioneering male trot vocalist'},
    'VM0435': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'heavy gravelly vocal with raw hard-rock intensity and deep emotional grit, modern layered R&B alto with dense, genius sensual male'},
    'VM0436': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'radiant smooth tenor with luminous Latin pop brass-backed vocal glow, emotional vocal trance,, 90s G-Funk revivalist'},
    'VM0437': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'refreshing powerful female country-pop, gentle soothing baritone with the warmest, Brazilian folk-house hybrid,'},
    'VM0438': {'cat':'B','tag':'중저음 매력','w':[40, 40, 20],'prompt':'mournful mid-bass male trot vocalist commanding, raw angular mezzo with unconventional phrasing, Latin rock-drill crossover female'},
    'VM0439': {'cat':'B','tag':'중저음 매력','w':[50, 30, 20],'prompt':'percussion-performing male trot vocalist with deeply sorrowful han-infused vocal power, passionate tender tenor with soulful Latin, aggressive hard-hitting male'},
    'VM0440': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'sorrowful French chanson female vocal pouring raw life pain like a violin, layered multitrack choral, commanding male rapper'},
    'VM0441': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'sticky deep husky Korean female hip-hop R&B vocal at the pinnacle, delicate symphonic metal, raw gravelly male rapper'},
    'VM0442': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'dreamy atmospheric electronic vocal, deep, precise rhythmic Swedish diva, Clean Bandit, ethereal crystalline'},
    'VM0443': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'lush romantic male vocal blending classical piano grandeur with pop yearning, street-style diva soprano, Elvis-inspired charismatic male'},
    'VM0444': {'cat':'B','tag':'중저음 매력','w':[50, 30, 20],'prompt':'warm charismatic baritone with calypso-tinged folk orchestral humanity, hit-songwriter female rapper with raw soulful trap, globally acclaimed UK'},
    'VM0445': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'bold theatrical baritone with brassy big band showman vocal energy, gentle acoustic to soaring, airy ethereal falsetto'},
    'VM0446': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'heavyweight drum-and-bass male rapper with signature UK grime flow mastery, distinctive nasal indie tenor with quirky charm, crystalline narrative'},
    'VM0447': {'cat':'B','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'clear bright female pop vocal with ultra-high technique continuing the Mariah Carey legacy, flawless classic female, intense dramatic'},
    'VM0448': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'minimalist acoustic male trot vocalist with deep resonance on simple folk melodies, master-architect male trot, French electroclash legend,'},
    'VM0449': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'sorrowful French chanson female vocal pouring raw life pain like a violin, soft breathy warm, razor-sharp UK grime'},
    'VM0450': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'mournful mid-bass male trot vocalist commanding orchestral-scale grand ballad narratives, operatic tenor male trot vocalist completing orchestral-scale, sweet lyrical soprano'},
    'VM0451': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'commanding heavy baritone with powerful sensual deep soul growl, broken sobbing male vocal pouring desperate modern, pristine clean high-note'},
    'VM0452': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'husky deep mid-low female trot vocalist adding mature depth to adult contemporary trot, overwhelming falsetto high male Korean ballad, husky heartbreak-filled'},
    'VM0453': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'husky powerful soprano with overwhelming, gritty warm male keyboard-soul vocal with bluesy, gentle wistful male'},
    'VM0454': {'cat':'B','tag':'중저음 매력','w':[40, 40, 20],'prompt':'bold deep contralto with distinctive vibrato, raw gravelly male rapper with ferocious barking energy, 90s G-Funk revivalist'},
    'VM0455': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'first lady of jazz, perfect pitch rhythm and freestyle scat female vocal, honest unpretentious warm, rapid-fire southern male'},
    'VM0456': {'cat':'B','tag':'압도적 고음','w':[40, 40, 20],'prompt':'blended operatic tenor ensemble with lush, soft breathy warm female pop vocal, Philadelphia dark-cloud trap'},
    'VM0457': {'cat':'B','tag':'압도적 고음','w':[60, 20, 20],'prompt':'Australian twin melody master, powerful vocal-driven progressive house, unique delicate Korean female, inventive creative female rapper'},
    'VM0458': {'cat':'B','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'5-octave female vocal with dolphin whistle register and R&B melisma, raw convulsive gravelly male vocal wringing every, serene quiet soprano with'},
    'VM0459': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'mellow melodic male rapper with addictive hooks and relaxed hybrid ballad delivery, flawless technique male Korean vocal mastering, mysterious Eastern pentatonic'},
    'VM0460': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, warm robust tenor with grand sweeping romantic, ethereal breathtaking soprano'},
    'VM0461': {'cat':'B','tag':'허스키 감성','w':[60, 30, 10],'prompt':'bel-canto male trot vocalist creating cinematic time-slip narratives with unique timbre, refreshing bright rock soprano with crisp attack, Polaris-winning Canadian female'},
    'VM0462': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'heavy dubstep-trap hybrid, dark aggressive bass, intense festival energy, piercing powerful, soulful mezzo-soprano with'},
    'VM0463': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'monster-vocal soprano with devastating power and next-generation explosive technique, authentic R&B alto with golden-era groove and, pristine classical'},
    'VM0464': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, lush romantic male, rapid-fire versatile female'},
    'VM0465': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'clear steady male pop-trot vocalist hiding deep lyricism behind flashy performance, refined velvety tenor with elegant soaring, emotionally charged soprano'},
    'VM0466': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'emotionally explosive belting soprano with dramatic, genre-bending refined alto with jazz-soul sophistication, crystalline pure-toned male trot'},
    'VM0467': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'commanding heavy baritone with powerful sensual, distinctive nasal indie tenor with quirky charm, bouncy yet heartfelt'},
    'VM0468': {'cat':'B','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'operatic tenor male trot vocalist completing orchestral-scale power with massive volume, husky heartbreak-filled, bright pure tenor'},
    'VM0469': {'cat':'B','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'ultimate Korean female high-note queen with blade-sharp, husky intelligent female R&B vocal with, angelic fragile yet'},
    'VM0470': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'sorrowful bending-note master male trot vocalist, youthful clear tenor with earnest modern, lush romantic male'},
    'VM0471': {'cat':'B','tag':'압도적 고음','w':[50, 30, 20],'prompt':'emotionally explosive belting soprano with dramatic orchestral climax power, airy ethereal male falsetto vocal building, hip-hop trap EDM'},
    'VM0472': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'all-range female trot technician vocalist spanning deep bass, operatic tenor male trot vocalist completing orchestral-scale, emotional vocal trance,'},
    'VM0473': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'smooth classic baritone male crooner jazz pop vocal, queen of soul, gospel-based explosive powerful female vocal, folk-ballad optimized male'},
    'VM0474': {'cat':'B','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'commanding dramatic diva vocal with explosive big-band cinematic delivery, clear earnest tenor with sweeping poetic, rugged male rapper blending'},
    'VM0475': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'rich deep mid-low voice with velvety soul resonance and warm delivery, powerful venue-shaking female, youthful crystalline soprano'},
    'VM0476': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'dignified low-tone male trot vocalist who sings like reciting poetry with gravitas, modern breathy alternative R&B vocal with, sharp organic indie-trap'},
    'VM0477': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'bouncy yet heartfelt country female vocal with, silvery celestial country harmony soprano with, smooth romantic male rapper'},
    'VM0478': {'cat':'B','tag':'중저음 매력','w':[60, 30, 10],'prompt':'heavyweight drum-and-bass male rapper with signature UK grime flow mastery, prodigy male trot vocalist mastering saxophone to orchestra, wise philosophical male'},
    'VM0479': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep literary lyrical male Korean pop ballad vocal with quiet resonance, crystalline soaring high male tenor with, hit-songwriter female rapper'},
    'VM0480': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'most sophisticated calm sensual mid-low female vocal with luxury tone, refined groovy male trot, ethereal Nordic'},
    'VM0481': {'cat':'B','tag':'허스키 감성','w':[60, 20, 20],'prompt':'sophisticated mid-range vocal with refined phrasing and understated elegance, crystal-clear healing female, Chicago hardcore drill'},
    'VM0482': {'cat':'B','tag':'압도적 고음','w':[70, 20, 10],'prompt':'refreshing powerful female country-pop crossover vocal, soft breathy warm, authoritative male rapper'},
    'VM0483': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'mellow melodic male rapper with addictive hooks and relaxed hybrid ballad delivery, pioneering male trot vocalist with signature vibrato who, quintessentially Korean optimistic male'},
    'VM0484': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'Miss Trot champion female vocalist with overwhelming pansori-based power shattering Korean han, pansori-infused cinematic male trot vocalist with elaborate melodic, ethereal Nordic'},
    'VM0485': {'cat':'B','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'sticky deep neo-soul male vocal that, gentle acoustic to soaring high notes, clear storytelling Korean, street-style diva soprano'},
    'VM0486': {'cat':'B','tag':'허스키 감성','w':[50, 30, 20],'prompt':'pansori-master young female trot vocalist melting fierce traditional soul into acoustic folk, comforting warm baritone with steady timeless, Bronx drill female'},
    'VM0487': {'cat':'B','tag':'압도적 고음','w':[50, 40, 10],'prompt':'doll-faced female trot vocalist hiding explosive pansori-scaled cinematic high-note power, smooth silky male R&B piano vocal, Afrobeat-drill fusion male'},
    'VM0488': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, folk-ballad optimized male, inventive jazzy soprano'},
    'VM0489': {'cat':'B','tag':'허스키 감성','w':[50, 40, 10],'prompt':'authentic R&B alto with golden-era groove and smooth soulful vocal runs, pure clean soprano capturing quiet depth, revolutionary UK grime male'},
    'VM0490': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'gayageum-playing female hybrid trot vocalist bridging traditional, pristine clear soprano with ethereal purity suited, Eastern-melodic female rapper'},
    'VM0491': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'smooth classic baritone male crooner jazz pop vocal, distinctive nasal indie, refined effortless alto'},
    'VM0492': {'cat':'B','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'sorrowful bending-note master male trot vocalist with deeply mournful delivery, sophisticated folk soprano, deep resonant female trot'},
    'VM0493': {'cat':'B','tag':'허스키 감성','w':[40, 40, 20],'prompt':'scratched wounded raspy male grunge vocal, youthful crystalline soprano with emotionally transparent, fierce Miami trap duo'},
    'VM0494': {'cat':'B','tag':'중저음 매력','w':[70, 20, 10],'prompt':'commanding hip-hop soul alto with passionate 90s ballad grit, rough torn raspy, Afrobeat-drill hybrid female'},
    'VM0495': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'anthem trance, heart-wrenching melodies, powerful emotional vocal trance hooks, explosive raspy male trot, precise pitch-perfect male pop'},
    'VM0496': {'cat':'B','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'sorrowful falsetto transitioning to angry melodic screaming male vocal, ice-cold sad rebellious female vocal with, airy ethereal male'},
    'VM0497': {'cat':'B','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'heavyweight commanding male rapper with flawless flow and deep groove mastery, distinctive high-pitched male, earnest narrative male'},
    'VM0498': {'cat':'B','tag':'중저음 매력','w':[50, 40, 10],'prompt':'deep velvety contralto with smoldering sultry low-register warmth, master-architect male trot vocalist radically mixing pansori,, inventive creative female rapper'},
    'VM0499': {'cat':'B','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, gentle acoustic to soaring high notes, clear storytelling Korean, passionate raspy tenor'},
    'VM0500': {'cat':'B','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'minimalist acoustic male trot vocalist with deep resonance on simple folk melodies, uniquely flavored male trot vocalist with signature, quirky playful'},
    'VM0501': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'distinctive high-pitched male rapper who pioneered gangsta, dreamy Latin-pop female rapper weaving ethereal harmonics, explosive raspy female rapper'},
    'VM0502': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'razor-sharp UK grime male rapper representing London streets with global authority, prodigious genius female, legendary nasal-melody female trot'},
    'VM0503': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'dramatic powerful male rock vocal with, legendary 80s female rapper who spearheaded hip-hop, addictive melodic Korean'},
    'VM0504': {'cat':'C','tag':'일렉트로닉','w':[50, 30, 20],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, legendary Three 6 Mafia female rapper ruling southern, idol-crossover female trot'},
    'VM0505': {'cat':'C','tag':'천상의 목소리','w':[40, 40, 20],'prompt':'nervous yet beautiful dreamy falsetto male, emerging Korean drill female rapper commanding heavy 808, 90s G-Funk revivalist'},
    'VM0506': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'cinematic male trot vocalist riding large string sections with emotional stability and depth, dark European drill female, pansori-master young female trot'},
    'VM0507': {'cat':'C','tag':'크리스탈 톤','w':[60, 20, 20],'prompt':'genius sensual male vocal switching between falsetto and chest voice, 5-octave female vocal, hardcore boom-bap Korean'},
    'VM0508': {'cat':'C','tag':'펑키 그루브','w':[50, 30, 20],'prompt':'versatile male vocal from soft falsetto to rock screaming, powerhouse big-voiced female dance-trot vocalist who, hypnotic baby-voice male'},
    'VM0509': {'cat':'C','tag':'허스키 감성','w':[60, 20, 20],'prompt':'genius rhythmic soulful male vocal with brilliant melisma technique, most destructive female, deep-voiced southern female rapper'},
    'VM0510': {'cat':'C','tag':'굵은 바리톤','w':[70, 20, 10],'prompt':'heavyweight deep-bass male rapper dominating grandiose trap beats with boss authority, explosive power from, pharmacist-turned female trot'},
    'VM0511': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'nasal high-pitched male rapper who pioneered, powerful pansori-toned female trot vocalist who made the, pioneering male trot vocalist'},
    'VM0512': {'cat':'C','tag':'허스키 매력','w':[60, 30, 10],'prompt':'rough textured male vocal with gritty emotional rock-ballad appeal and edge, intense 90s New York hardcore female rapper, pansori-based male trot'},
    'VM0513': {'cat':'C','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'polished male rapper with clean jazz-synthpop beats and sophisticated hybrid delivery, emerging Korean drill female, revolutionary male rapper who'},
    'VM0514': {'cat':'C','tag':'그루브 보컬','w':[50, 30, 20],'prompt':'atmospheric Canadian male rapper crafting dreamy synth-trap soundscapes with soft delivery, explosive energy rough husky female rock and roll, authoritative male rapper'},
    'VM0515': {'cat':'C','tag':'깊은 울림','w':[60, 30, 10],'prompt':'deep-voiced male rapper-producer who powered Death Row Records golden era sound, gentle acoustic to soaring high notes, clear storytelling Korean, explosive energy rough husky'},
    'VM0516': {'cat':'C','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'pristine smooth male easy-listening vocal with serene velvety mid-range tone, new-wave rock female rapper who shattered boundaries between, refreshing powerful'},
    'VM0517': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'smooth romantic male rapper delivering the most mellow G-Funk with laid-back flow, heavyweight hardcore boom-bap female rapper with solid, rhythmic groove master male'},
    'VM0518': {'cat':'C','tag':'깊은 울림','w':[50, 40, 10],'prompt':'thunderous deep-cave male rapper who exploded Brooklyn drill onto the global stage, Miss Trot champion female vocalist with overwhelming pansori-based, rough torn raspy'},
    'VM0519': {'cat':'C','tag':'깊은 베이스','w':[70, 20, 10],'prompt':'mellow melodic male rapper with addictive hooks and relaxed hybrid ballad delivery, quiet warm soothing, arrogant cynical distinctive'},
    'VM0520': {'cat':'C','tag':'허스키 매력','w':[70, 20, 10],'prompt':'rough textured male vocal with gritty emotional rock-ballad appeal and edge, legendary 80s female, raw aching male piano'},
    'VM0521': {'cat':'C','tag':'깊은 베이스','w':[40, 40, 20],'prompt':'polished swinging male crooner vocal with, Eastern-melodic female rapper crossing Asian tonality, deep-voiced southern female rapper'},
    'VM0522': {'cat':'C','tag':'그루브 보컬','w':[60, 20, 20],'prompt':'precision-engineered modern male rapper with relentless continuous trap flow dominance, highway queen female trot, raw aching male piano'},
    'VM0523': {'cat':'C','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'authoritative smooth male rapper with business-mogul swagger and effortless delivery, musical-theater trained female, soulful male vocal with'},
    'VM0524': {'cat':'C','tag':'크리스탈 톤','w':[70, 20, 10],'prompt':'angelic fragile yet devastating falsetto male vocal with soul-shaking emotion, Afrobeat-drill hybrid female, explosive power from'},
    'VM0525': {'cat':'C','tag':'천상의 목소리','w':[50, 40, 10],'prompt':'angelic fragile yet devastating falsetto male vocal with soul-shaking emotion, elegant 60s female trot vocalist layering sophisticated, next-generation hardcore female'},
    'VM0526': {'cat':'C','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'flawless technique male Korean vocal mastering every emotion perfectly, new-wave rock female rapper, sharp organic indie-trap'},
    'VM0527': {'cat':'C','tag':'깊은 베이스','w':[50, 30, 20],'prompt':'heavyweight deep-bass male rapper dominating grandiose trap beats with boss authority, radically alternative female rapper fusing avant-garde, sorrowful bending-note master'},
    'VM0528': {'cat':'C','tag':'허스키 매력','w':[50, 40, 10],'prompt':'bel-canto male trot vocalist creating cinematic time-slip narratives with unique timbre, technically versatile Korean female rapper freely riding R&B, soulful male vocal with'},
    'VM0529': {'cat':'C','tag':'일렉트로닉','w':[50, 40, 10],'prompt':'percussion-performing male trot vocalist with deeply sorrowful han-infused vocal power, Chicago hardcore drill female rapper with precise, thunderous military-grade male'},
    'VM0530': {'cat':'C','tag':'중저음 매력','w':[70, 20, 10],'prompt':'deep literary lyrical male Korean pop ballad vocal with quiet resonance, Korean R&B fairy female, R&B-infused female rapper riding'},
    'VM0531': {'cat':'C','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, cute nasally charming female electronic dance-trot vocalist, hard-hitting slide-drill male'},
    'VM0532': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'technically gifted male rapper with extraordinary rhyme arrangement on west-coast beats, unique sophisticated neo-soul, saddest tone in jazz'},
    'VM0533': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'explosive operatic metal male vocal like a human air raid siren, world-class speed-rap female rapper switching effortlessly, new-wave rock female rapper'},
    'VM0534': {'cat':'C','tag':'허스키 감성','w':[40, 40, 20],'prompt':'silky smooth sensual male, hardcore boom-bap Korean female rapper filling beats, husky deep mid-low female'},
    'VM0535': {'cat':'C','tag':'그루브 보컬','w':[60, 30, 10],'prompt':'atmospheric Canadian male rapper crafting dreamy synth-trap soundscapes with soft delivery, iconic pop female vocal with unique tone that, battle-rap legend female'},
    'VM0536': {'cat':'C','tag':'투명한 음색','w':[70, 20, 10],'prompt':'crystalline pure-toned male trot tenor revered as the emperor of classic Korean enka, deep powerful female, raw rough soul-shaking'},
    'VM0537': {'cat':'C','tag':'천상의 목소리','w':[60, 30, 10],'prompt':'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes, all-range female trot technician vocalist spanning deep bass, wildly innovative southern male'},
    'VM0538': {'cat':'C','tag':'시원한 고음','w':[50, 30, 20],'prompt':'husky powerful male belting vocal with soul-drenched rock ballad intensity, whisper-soft literary female rapper floating poetically over jazz, warm low-register male'},
    'VM0539': {'cat':'C','tag':'천상의 목소리','w':[70, 20, 10],'prompt':'warm low-register male trot vocalist evoking hometown nostalgia with gentle phrasing, textbook female trot vocalist, sharp high-tone female'},
    'VM0540': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'powerful UK national male rapper fusing grime with, positive upbeat female dance-trot vocalist who pioneered rhythmic, Caribbean-fusion male rapper'},
    'VM0541': {'cat':'C','tag':'크리스탈 톤','w':[50, 30, 20],'prompt':'angelic fragile yet devastating falsetto male vocal with soul-shaking emotion, explosive powerful energetic Korean female vocal mastering, Polaris-winning Canadian female'},
    'VM0542': {'cat':'C','tag':'그루브 보컬','w':[60, 30, 10],'prompt':'gangster-crew male rappers carrying west-coast and global rap legacy with authority, soft breathy warm female pop vocal, pioneering male trot vocalist'},
    'VM0543': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'raspy warm male rock vocal with anthemic sing-along ballad grit, London-born healing female, dreamy Latin-pop female'},
    'VM0544': {'cat':'C','tag':'크리스탈 톤','w':[60, 20, 20],'prompt':'bright energetic male trot vocalist radiating vitality with open airy tenor delivery, highway queen female trot, perfect vocal technique'},
    'VM0545': {'cat':'C','tag':'감성 폭발','w':[50, 30, 20],'prompt':'underground gritty male rapper with raw boom-bap sensibility and street authenticity, deep resonant female trot vocalist who tenderly soothed, theatrical sweeping male'},
    'VM0546': {'cat':'C','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'clear smooth pure falsetto male R&B vocal, most destructive female rock vocal in, aggressive hard-hitting male'},
    'VM0547': {'cat':'C','tag':'깊은 베이스','w':[60, 20, 20],'prompt':'thunderous deep-cave male rapper who exploded Brooklyn drill onto the global stage, idol-crossover female trot, timeless elegant male'},
    'VM0548': {'cat':'C','tag':'허스키 매력','w':[50, 40, 10],'prompt':'raw rough soul-shaking male rock vocal with emotional honesty, punk-rage female rapper pioneering trap-metal by fusing rock, pioneering fierce female'},
    'VM0549': {'cat':'C','tag':'투명한 음색','w':[50, 40, 10],'prompt':'warm intimate male folk pop vocal with gentle rasp, Miss Trot champion female vocalist with overwhelming pansori-based, arrogant cynical distinctive'},
    'VM0550': {'cat':'C','tag':'시원한 고음','w':[50, 40, 10],'prompt':'soaring male rock ballad vocal with polished tenor and guitar-driven passion, pioneering fierce female rock alto with, genius K-pop producing female'},
    'VM0551': {'cat':'C','tag':'허스키 감성','w':[40, 40, 20],'prompt':'sophisticated mid-century male trot vocalist bridging modern, husky deep mid-low female trot vocalist adding mature, innovative female rapper-producer'},
    'VM0552': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'raspy warm male rock vocal with anthemic sing-along ballad grit, explosive hardcore female rapper who shredded 90s Death, technically sharp male'},
    'VM0553': {'cat':'C','tag':'깊은 베이스','w':[60, 20, 20],'prompt':'minimalist acoustic male trot vocalist with deep resonance on simple folk melodies, pure refreshing female trot, energetic flashy male rapper'},
    'VM0554': {'cat':'C','tag':'투명한 음색','w':[50, 40, 10],'prompt':'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes, textbook female trot vocalist with decades of live, dreamy atmospheric female trot'},
    'VM0555': {'cat':'C','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep literary lyrical male Korean pop ballad vocal with quiet resonance, legendary Three 6 Mafia female rapper ruling southern, gentle acoustic to soaring'},
    'VM0556': {'cat':'C','tag':'허스키 감성','w':[70, 20, 10],'prompt':'folk-ballad optimized male trot vocalist with sweet sentimental melodic craftsmanship, cute bright female, mournful mid-bass male'},
    'VM0557': {'cat':'C','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, battle-rap legend female, wailing blues-rock male'},
    'VM0558': {'cat':'C','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'genius sensual male vocal switching between falsetto and chest voice with incredible range, revolutionary female rapper fusing third-world percussion, unique sophisticated neo-soul'},
    'VM0559': {'cat':'C','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'atmospheric Canadian male rapper crafting dreamy synth-trap soundscapes with soft delivery, legendary nasal sorrowful uniquely, feathery soft tender'},
    'VM0560': {'cat':'C','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'thunderous deep-cave male rapper who exploded Brooklyn drill onto the global stage, global Billboard-hitting female rapper with Thai-international swagger, pristine clean high-note'},
    'VM0561': {'cat':'C','tag':'깊은 베이스','w':[50, 30, 20],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, pansori-certified female crossover trot vocalist singing, nasal high-pitched male'},
    'VM0562': {'cat':'C','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'dignified low-tone male trot vocalist who sings like reciting poetry with gravitas, Bronx drill female rapper who conquered global, deep classic husky female'},
    'VM0563': {'cat':'C','tag':'굵은 바리톤','w':[70, 20, 10],'prompt':'deep heavy charismatic low male vocal carrying philosophical messages, first foreign trot, plaintive gentle male trot'},
    'VM0564': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'bel-canto male trot vocalist creating cinematic time-slip narratives with unique timbre, pioneering boom-bap female rapper who achieved the first, legendary storytelling male'},
    'VM0565': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'raw aching male piano vocal that erupts from whisper to desperate wail, viral hook-machine female rapper crafting addictive trap, Afrobeat-drill hybrid female'},
    'VM0566': {'cat':'C','tag':'감성 폭발','w':[40, 40, 20],'prompt':'uniquely flavored male trot vocalist with signature, folk-rock gentle female trot vocalist comforting the nation, pansori-certified female crossover'},
    'VM0567': {'cat':'C','tag':'폭발 에너지','w':[70, 20, 10],'prompt':'bel-canto male trot vocalist creating cinematic time-slip narratives with unique timbre, crystal clear yet, raw powerful female'},
    'VM0568': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'raw rough soul-shaking male rock vocal with emotional honesty, technically versatile Korean female rapper freely riding R&B, wise philosophical male'},
    'VM0569': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'tireless iron-throated high male rock vocal with unique groove, understated monotone boom-bap female rapper dominating, Miami hardcore female rapper'},
    'VM0570': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'rhythmic male pop vocal freely switching between falsetto and chest voice, deep-voiced southern female rapper with heavy 808 impact, distinctive husky low-tone'},
    'VM0571': {'cat':'C','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'refined groovy male trot vocalist with idol-trained polish and solid vocal technique, pioneering boom-bap female rapper, razor-precise Korean female'},
    'VM0572': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'explosive raspy male trot vocalist with gut-wrenching sorrow and raw emotional power, Dutch boom-bap female rapper who captivated all of, sophisticated tension-chord female'},
    'VM0573': {'cat':'C','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'heavyweight drum-and-bass male rapper with signature UK grime flow mastery, fierce Miami trap duo, arrogant cynical distinctive'},
    'VM0574': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'pansori-infused cinematic male trot vocalist with elaborate melodic architecture and grand projection, 2NE1 hardcore K-pop female rapper who pioneered Billboard, autotune-wielding male rapper'},
    'VM0575': {'cat':'C','tag':'중저음 매력','w':[60, 30, 10],'prompt':'traditional bending-note technician male trot vocalist with earthy fermented-bean voice, most sophisticated calm sensual mid-low female, sorrowful bending-note master'},
    'VM0576': {'cat':'C','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'trendy stylish male rapper layering fashion-forward aesthetics over New York boom-bap, androgynous cold urban, clear smooth'},
    'VM0577': {'cat':'C','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'deep literary lyrical male Korean pop ballad vocal with quiet resonance, crystal clear yet, versatile male vocalist seamlessly'},
    'VM0578': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'multi-talented male rapper-producer with refined west-coast lyricism and groove mastery, perfect powerful female R&B pop vocal with flawless, revolutionary male rapper who'},
    'VM0579': {'cat':'C','tag':'그루브 보컬','w':[60, 20, 20],'prompt':'trendy stylish male rapper layering fashion-forward aesthetics over New York boom-bap, deep classic husky female, psychedelic male rapper'},
    'VM0580': {'cat':'C','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'transparent fragile male vocal with crystalline sad tone and quiet intensity, lethal off-beat female rapper-singer delivering devastating, perfect powerful female R&B'},
    'VM0581': {'cat':'C','tag':'압도적 고음','w':[60, 30, 10],'prompt':'classically trained powerful high-note male trot vocalist who launched power-trot era, idol-trained groovy female new-wave trot vocalist, explosive powerful energetic'},
    'VM0582': {'cat':'C','tag':'감성 폭발','w':[50, 30, 20],'prompt':'rugged bending-note male trot vocalist cutting through grand horn and string ensembles, quiet warm soothing female jazz folk vocal, theatrical sweeping male'},
    'VM0583': {'cat':'C','tag':'그루브 보컬','w':[40, 40, 20],'prompt':'punk-rock crossover male rapper who perfectly blended, hit-songwriter female rapper with raw soulful trap, fierce Miami trap duo'},
    'VM0584': {'cat':'C','tag':'크리스탈 톤','w':[70, 20, 10],'prompt':'quintessentially Korean optimistic male trot vocalist with earthy rustic warmth and joy, 90s G-Funk revivalist, raw gravelly male rapper'},
    'VM0585': {'cat':'C','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'distinctive high-pitched male rapper who pioneered gangsta rap with piercing tone, technically versatile Korean female rapper freely riding R&B, slow heavyweight UK underground'},
    'VM0586': {'cat':'C','tag':'감성 폭발','w':[70, 20, 10],'prompt':'folk-ballad optimized male trot vocalist with sweet sentimental melodic craftsmanship, powerful pansori-toned female trot, solid high male'},
    'VM0587': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'hypnotic baby-voice male rapper commanding rave-trap with addictive minimalist flow, crystal clear yet, velvety smooth perfect'},
    'VM0588': {'cat':'C','tag':'투명한 음색','w':[50, 30, 20],'prompt':'folk-ballad optimized male trot vocalist with sweet sentimental melodic craftsmanship, quiet warm soothing female jazz folk vocal, smooth silky male'},
    'VM0589': {'cat':'C','tag':'허스키 감성','w':[50, 40, 10],'prompt':'raw rough soul-shaking male rock vocal with emotional honesty, most sophisticated calm sensual mid-low female, explosive raspy female rapper'},
    'VM0590': {'cat':'C','tag':'감성 폭발','w':[50, 30, 20],'prompt':'raw convulsive gravelly male vocal wringing every note with blues agony, perfect powerful female R&B pop vocal with flawless, silky sliding male'},
    'VM0591': {'cat':'C','tag':'굵은 바리톤','w':[40, 40, 20],'prompt':'rich earthy male trot baritone comforting working-class, girl-group trained female trot vocalist hiding solid traditional, most sophisticated calm'},
    'VM0592': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'desperately emotional high-pitched male Korean vocal that makes everyone cry, technically versatile Korean female, underground legend male'},
    'VM0593': {'cat':'C','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'Afrobeat-drill fusion male rapper who pioneered Afroswing genre with infectious energy, highway queen female trot, gritty groovy modern'},
    'VM0594': {'cat':'C','tag':'천상의 목소리','w':[50, 30, 20],'prompt':'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes, sorrowful French chanson female vocal pouring raw life, passionate revolutionary male'},
    'VM0595': {'cat':'C','tag':'허스키 감성','w':[50, 40, 10],'prompt':'traditional bending-note technician male trot vocalist with earthy fermented-bean voice, most sophisticated calm sensual mid-low female, global Billboard-hitting female'},
    'VM0596': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'versatile raw male rapper spanning distorted lo-fi beats to tender acoustic rap, heavyweight hardcore boom-bap female rapper with solid, Dutch boom-bap female rapper'},
    'VM0597': {'cat':'C','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'pinnacle of Korean R&B soul male vocal with perfect high-tone technique, cute nasally charming female electronic dance-trot vocalist, husky soulful female'},
    'VM0598': {'cat':'C','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'legendary pure high male tenor with effortless, legendary 80s female rapper who spearheaded hip-hop, velvety smooth perfect'},
    'VM0599': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'flawless technique male Korean vocal mastering every emotion perfectly, inventive creative female rapper, wise philosophical male'},
    'VM0600': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'sandpaper-rough charming male vocal with loose swaggering rock-ballad phrasing, viral hook-machine female rapper crafting addictive trap, solid high male'},
    'VM0601': {'cat':'C','tag':'폭발 에너지','w':[40, 40, 20],'prompt':'gritty charismatic male stadium-rock vocal with, dark European drill female rapper commanding heavy trap, pansori-certified female crossover'},
    'VM0602': {'cat':'C','tag':'압도적 고음','w':[70, 20, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, ice-cold sad rebellious, pinnacle of Korean'},
    'VM0603': {'cat':'C','tag':'깊은 베이스','w':[60, 30, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, husky intelligent female R&B vocal with, deep baritone-grade Korean female'},
    'VM0604': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'sharp nervous ultra-high screaming male rock vocal with wide range, underground technical female rapper with the, globally acclaimed UK'},
    'VM0605': {'cat':'C','tag':'감성 보컬','w':[70, 20, 10],'prompt':'deep-voiced male rapper-producer who powered Death Row Records golden era sound, punk rock godmother, the king of Korean'},
    'VM0606': {'cat':'C','tag':'펑키 그루브','w':[60, 30, 10],'prompt':'transparent fragile male vocal with crystalline sad tone and quiet intensity, UK club hyperpop female rapper crossing electronic beats, pioneering UK grime-garage'},
    'VM0607': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'ice-cold monotone male rapper embodying modern dark trap with deadpan delivery, refreshing powerful female country-pop, trailblazing New York female'},
    'VM0608': {'cat':'C','tag':'압도적 고음','w':[50, 30, 20],'prompt':'genius lyricist UK male rapper delivering profound narratives over piano-driven beats, 5-octave female vocal with dolphin whistle, raw powerful female'},
    'VM0609': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'genius lyricist UK male rapper delivering profound narratives over piano-driven beats, stable powerhouse female trot vocalist with the most, genius K-pop producing female'},
    'VM0610': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'broken sobbing male vocal pouring desperate modern heartbreak with Scottish rasp, anime-aesthetic female rapper, dreamy atmospheric female trot'},
    'VM0611': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'pioneering male rapper who defined modern rhyme schemes with meticulous cadence, bouncy yet heartfelt country female vocal with, accessible narrative male'},
    'VM0612': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'overwhelming falsetto high male Korean ballad vocal dominating karaoke, cute nasally charming, gangster-crew male rappers'},
    'VM0613': {'cat':'C','tag':'그루브 보컬','w':[60, 20, 20],'prompt':'wordplay-brilliant UK duo male rappers with pop-friendly drill beat chemistry, deep classic husky, underground technical female'},
    'VM0614': {'cat':'C','tag':'크리스탈 톤','w':[60, 20, 20],'prompt':'clear smooth pure falsetto male R&B vocal, whisper-soft literary female rapper, androgynous cold urban'},
    'VM0615': {'cat':'C','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'crystalline soaring high male tenor with smooth Chicago soft-rock shimmer, commanding female rapper who elevated hip-hop with social, textbook traditional female trot'},
    'VM0616': {'cat':'C','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'hard-hitting slide-drill male rapper with powerful 808 bass-riding technique, Griselda Records queen female, powerful rough soulful'},
    'VM0617': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'versatile male vocalist seamlessly fusing rap and R&B over laid-back west-coast beats, musical-theater trained female power-trot vocalist with, explosive raspy female rapper'},
    'VM0618': {'cat':'C','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'hook-driven addictive male trot vocalist dominating with catchy refrains and deep emotion, viral hook-machine female rapper crafting addictive trap, sorrowful falsetto transitioning'},
    'VM0619': {'cat':'C','tag':'거친 소울','w':[60, 20, 20],'prompt':'pop-rock acoustic male rapper who conquered Billboard with accessible crossover sound, pop-ballad crossover female, Korean R&B fairy female'},
    'VM0620': {'cat':'C','tag':'압도적 고음','w':[40, 40, 20],'prompt':'emo-rock infused male rapper fusing emotional intensity, clear bright female pop vocal with ultra-high technique, solid expressive female trot'},
    'VM0621': {'cat':'C','tag':'크리스탈 톤','w':[60, 30, 10],'prompt':'quintessentially Korean optimistic male trot vocalist with earthy rustic warmth and joy, dancehall-reggae female rapper fusing Caribbean rhythms with, high-pitched screaming male'},
    'VM0622': {'cat':'C','tag':'감성 폭발','w':[40, 40, 20],'prompt':'raw aching male piano vocal that erupts from, clear bright female pop vocal, aggressive hard-hitting male'},
    'VM0623': {'cat':'C','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, pioneering UK grime-garage, legendary pure high'},
    'VM0624': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'gangster-crew male rappers carrying west-coast and global, mysterious powerful gothic female rock vocal piercing, rapid-fire versatile female'},
    'VM0625': {'cat':'C','tag':'허스키 감성','w':[70, 20, 10],'prompt':'rugged male rapper blending gritty tone with gangster balladry and west-coast soul, refined female trot vocalist, inventive genius male'},
    'VM0626': {'cat':'C','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'sky-high angelic male falsetto vocal soaring through romantic soft-rock climaxes, Canadian dark-aesthetic female, silky laid-back male'},
    'VM0627': {'cat':'C','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'slow heavyweight UK underground male rapper with iconic, revolutionary female rapper fusing third-world percussion, rebellious melancholic raw retro'},
    'VM0628': {'cat':'C','tag':'허스키 감성','w':[40, 40, 20],'prompt':'raw rough soul-shaking male rock vocal, cute bright female trot vocalist dominating highway-groove, charming deep baritone'},
    'VM0629': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'rapid-fire technical male rapper balancing speed with accessible pop-ballad sensibility, sad sharp Irish traditional female vocal, wise philosophical male'},
    'VM0630': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'foundational male rapper who architected modern trap hip-hop culture and sound, barefoot diva, deeply appealing Korean female vocal drawn from, wailing blues-rock male'},
    'VM0631': {'cat':'C','tag':'리듬 보컬','w':[60, 20, 20],'prompt':'relentless rapid-fire male rapper with massive projection and boom-bap dominance, rustic mid-low bending-note, pinnacle of Korean'},
    'VM0632': {'cat':'C','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'electronic-trap crossover male rapper blending synths with global hybrid rap delivery, deep resonant female trot vocalist who tenderly soothed, virtuoso male rapper'},
    'VM0633': {'cat':'C','tag':'파워 보컬','w':[60, 20, 20],'prompt':'genius sensual male vocal switching between falsetto and chest voice, crystal clear yet, solid expressive female trot'},
    'VM0634': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'rhythmic groove master male trot vocalist who electrified all generations with one hit, deep heavy contralto female vocal singing the, genius singer-songwriter male'},
    'VM0635': {'cat':'C','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'pinnacle of Korean R&B soul male vocal with perfect high-tone technique, glamorous west-coast female, The Voice, perfect'},
    'VM0636': {'cat':'C','tag':'일렉트로닉','w':[60, 20, 20],'prompt':'deep baritone male alternative rock vocal icon with emotional depth, witty transatlantic female rapper, Polaris-winning Canadian female'},
    'VM0637': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'raw convulsive gravelly male vocal wringing every note with blues agony, bold thick-toned Korean, legendary nasal-melody female trot'},
    'VM0638': {'cat':'C','tag':'그루브 보컬','w':[70, 20, 10],'prompt':'silky sliding male rapper perfecting melodic trap with fluid effortless delivery, bouncy yet heartfelt, fierce Miami trap duo'},
    'VM0639': {'cat':'C','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'punk-rock crossover male rapper who perfectly blended hardcore punk with trap, barefoot diva, deeply appealing Korean female vocal drawn from, deep resonant female trot'},
    'VM0640': {'cat':'C','tag':'거친 소울','w':[60, 30, 10],'prompt':'velvety smooth perfect male R&B ballad vocal with full volume, husky deep mid-low female trot vocalist adding mature, raw convulsive gravelly'},
    'VM0641': {'cat':'C','tag':'허스키 감성','w':[40, 40, 20],'prompt':'theatrical sweeping male piano ballad vocal, modern boom-bap female rapper praised by legends for, overwhelming falsetto high'},
    'VM0642': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'genius lyricist UK male rapper delivering profound narratives over piano-driven beats, mysterious powerful gothic female rock vocal piercing, UK club hyperpop female'},
    'VM0643': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'arrogant cynical distinctive male britpop vocal that defined an era, genius K-pop producing female rapper who shatters idol, Griselda Records queen female'},
    'VM0644': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'multi-talented male rapper-producer with refined west-coast lyricism and groove mastery, stable powerhouse female trot vocalist with the most, technically sharp male'},
    'VM0645': {'cat':'C','tag':'허스키 매력','w':[40, 40, 20],'prompt':'raw convulsive gravelly male vocal wringing every, revolutionary female rapper fusing third-world percussion, earthy rustic male trot'},
    'VM0646': {'cat':'C','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'high-pitched screaming male rock vocal, textbook hard rock tenor, microtonal Arab-maqam female, dramatic powerful male'},
    'VM0647': {'cat':'C','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'operatic tenor male trot vocalist completing orchestral-scale power with massive volume, dark European drill female rapper commanding heavy trap, Korean R&B fairy female'},
    'VM0648': {'cat':'C','tag':'펑키 그루브','w':[40, 40, 20],'prompt':'perfect vocal technique with appealing sweet, explosive hardcore female rapper who shredded 90s Death, rapid-fire southern male'},
    'VM0649': {'cat':'C','tag':'천상의 목소리','w':[60, 20, 20],'prompt':'warm folk acoustic Korean male vocal with heartfelt lonely storytelling, rich-volume female cinematic, technically versatile Korean female'},
    'VM0650': {'cat':'C','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'lightning-fast male rapper layering angelic melodies over rapid-fire delivery uniquely, otherworldly bizarre yet beautiful avant-garde female, feathery soft tender'},
    'VM0651': {'cat':'C','tag':'허스키 감성','w':[60, 20, 20],'prompt':'explosive raspy male trot vocalist with gut-wrenching sorrow and raw emotional power, glamorous west-coast female, powerful open-throated male'},
    'VM0652': {'cat':'C','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, pure refreshing female trot vocalist providing emotional calm, raspy warm male'},
    'VM0653': {'cat':'C','tag':'맑은 감성','w':[60, 20, 20],'prompt':'silky laid-back male rapper with signature drawl and effortless west-coast groove, ethereal theatrical falsetto, warm intimate male'},
    'VM0654': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'emo-rock infused male rapper fusing emotional intensity with rapid hi-hat trap, rustic mid-low bending-note female trot vocalist, Korean R&B fairy female'},
    'VM0655': {'cat':'C','tag':'허스키 감성','w':[50, 30, 20],'prompt':'arrogant cynical distinctive male britpop vocal that defined an era, refined female trot vocalist who distills deep traditional, classically trained powerful'},
    'VM0656': {'cat':'C','tag':'깊은 베이스','w':[40, 40, 20],'prompt':'mournful mid-bass male trot vocalist commanding, first lady of jazz, perfect pitch rhythm and, dreamy Latin-pop female'},
    'VM0657': {'cat':'C','tag':'크리스탈 톤','w':[70, 20, 10],'prompt':'feathery soft tender male vocal with airy gentle folk-pop delivery, world-class speed-rap female, genius male vocal'},
    'VM0658': {'cat':'C','tag':'압도적 고음','w':[50, 30, 20],'prompt':'sacred powerful metal male vocal with commanding volume from small frame, all-range female trot technician vocalist spanning deep bass, bright cheerful male'},
    'VM0659': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'operatic tenor male trot vocalist completing orchestral-scale power with massive volume, stage-dominating female trot vocalist with addictive groove, Dutch boom-bap female rapper'},
    'VM0660': {'cat':'C','tag':'그루브 보컬','w':[70, 20, 10],'prompt':'addictive punchline male rapper with playful charisma and infectious trap mastery, rustic mid-low bending-note, legendary harmony female'},
    'VM0661': {'cat':'C','tag':'천상의 목소리','w':[50, 40, 10],'prompt':'quintessentially Korean optimistic male trot vocalist with earthy rustic warmth and joy, spicy capsaicin-sharp female trot vocalist with traditional, relentless rapid-fire male'},
    'VM0662': {'cat':'C','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'plaintive gentle male trot vocal soothing homesick hearts with simple heartfelt melody, fierce female rapper from, new-wave rock female rapper'},
    'VM0663': {'cat':'C','tag':'허스키 감성','w':[60, 20, 20],'prompt':'broken sobbing male vocal pouring desperate modern heartbreak with Scottish rasp, charming mid-low Korean female, dignified low-tone male trot'},
    'VM0664': {'cat':'C','tag':'리듬 보컬','w':[60, 20, 20],'prompt':'solid high male vocal with retro and modern groove, energetic performer, crystal-clear healing female, stable powerhouse female trot'},
    'VM0665': {'cat':'C','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'slow heavyweight UK underground male rapper with iconic deep bass flow delivery, bouncy southern trap female rapper optimized, quintessentially Korean optimistic male'},
    'VM0666': {'cat':'C','tag':'압도적 고음','w':[70, 20, 10],'prompt':'legendary storytelling male rapper with unique accent and theatrical narrative flow, deep powerful female, perfect powerful female'},
    'VM0667': {'cat':'C','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'feathery soft tender male vocal with airy gentle folk-pop delivery, trailblazing New York female rapper who set fashion, Latin Afro-beat female rapper'},
    'VM0668': {'cat':'C','tag':'펑키 그루브','w':[40, 40, 20],'prompt':'addictive punchline male rapper with playful charisma, ice-cold sad rebellious female vocal with, dreamy atmospheric female trot'},
    'VM0669': {'cat':'C','tag':'허스키 매력','w':[60, 30, 10],'prompt':'prodigy male trot vocalist mastering saxophone to orchestra with epic narrative depth, cute nasally charming female electronic dance-trot vocalist, raw rough soul-shaking'},
    'VM0670': {'cat':'C','tag':'리듬 보컬','w':[60, 30, 10],'prompt':'energetic flashy male rapper with wild hybrid flow over hardcore club beats, refined female trot vocalist who distills deep traditional, perfect powerful female R&B'},
    'VM0671': {'cat':'C','tag':'감성 폭발','w':[60, 30, 10],'prompt':'folk-ballad optimized male trot vocalist with sweet sentimental melodic craftsmanship, explosive power from small frame, timeless clear, new-wave rock female rapper'},
    'VM0672': {'cat':'C','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'modern male rapper perfectly reviving 90s golden-era New York boom-bap aesthetics, 5-octave female vocal, minimalist acoustic male trot'},
    'VM0673': {'cat':'C','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'legendary storytelling male rapper with unique accent and theatrical narrative flow, legendary harmony female, highway queen female trot'},
    'VM0674': {'cat':'C','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'genius sensual male vocal switching between falsetto and chest voice, historic west-coast crew female rapper with distinctive, explosive operatic metal'},
    'VM0675': {'cat':'C','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'storytelling piano male vocal with warm gritty New York baritone charm, Latin reggaeton-drill crossover female rapper connecting Caribbean, crystal clear yet'},
    'VM0676': {'cat':'C','tag':'깊은 베이스','w':[70, 20, 10],'prompt':'slow heavyweight UK underground male rapper with iconic deep bass flow delivery, Terror Squad pride female, clear steady male'},
    'VM0677': {'cat':'C','tag':'거친 소울','w':[50, 30, 20],'prompt':'genius lyricist UK male rapper delivering profound narratives over piano-driven beats, ice-cold sad rebellious female vocal with, powerful rough soulful'},
    'VM0678': {'cat':'C','tag':'펑키 그루브','w':[40, 40, 20],'prompt':'funky freewheeling male rapper with raw unfiltered, husky intelligent female R&B vocal with, most devastating powerful'},
    'VM0679': {'cat':'C','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'immortal deep baritone male trot vocalist who elevated the genre with noble dignity, solid expressive female trot, soulful jazzy male rapper'},
    'VM0680': {'cat':'C','tag':'시원한 고음','w':[60, 20, 20],'prompt':'stadium-filling resonant male vocal with powerful message delivery, rebellious melancholic raw retro, emo-rock infused male'},
    'VM0681': {'cat':'C','tag':'거친 소울','w':[50, 30, 20],'prompt':'the king of Korean pop, versatile male vocal covering rock ballad and folk, NYC underground queen female rapper embodying alternative, folk-rock gentle female trot'},
    'VM0682': {'cat':'C','tag':'거친 소울','w':[70, 20, 10],'prompt':'energetic flashy male rapper with wild hybrid flow over hardcore club beats, pansori-master young female trot, rugged bending-note male trot'},
    'VM0683': {'cat':'C','tag':'압도적 고음','w':[60, 30, 10],'prompt':'heart-wrenching melodic male rapper who epitomized emo-rap with devastating melodies, powerful venue-shaking female dance-trot vocalist commanding, deep powerful female'},
    'VM0684': {'cat':'C','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'nervous yet beautiful dreamy falsetto male vocal, ethereal and haunting, underground Korean female, Australian-born female rapper'},
    'VM0685': {'cat':'C','tag':'감성 폭발','w':[40, 40, 20],'prompt':'raw convulsive gravelly male vocal wringing every, explosive next-generation female cinematic trot vocalist, bright cheerful male'},
    'VM0686': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'rapid-fire technical male rapper balancing speed with accessible pop-ballad sensibility, husky soulful female vocal fusing hip-hop and, triplet-flow male rapper'},
    'VM0687': {'cat':'C','tag':'그루브 보컬','w':[60, 20, 20],'prompt':'paradigm-shifting male rapper-producer and hip-hop genre greatest sonic innovator ever, folk-rock gentle female trot, rich-volume female cinematic'},
    'VM0688': {'cat':'C','tag':'깊은 울림','w':[50, 40, 10],'prompt':'theatrical mysterious mid-low male vocal with glam rock charisma, textbook traditional female trot vocalist with the most, crystal clear yet'},
    'VM0689': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'classical crossover male trot vocalist harmonizing operatic power with grand orchestral scale, commanding female rapper who elevated hip-hop with social, light floating female rapper'},
    'VM0690': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'theatrical sweeping male piano ballad vocal with dramatic crescendo delivery, France greatest-selling female rapper with epic, psychedelic male rapper'},
    'VM0691': {'cat':'C','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'clear steady male pop-trot vocalist hiding deep lyricism behind flashy performance, original all-rounder Korean female rapper with rapid-fire, glamorous west-coast female'},
    'VM0692': {'cat':'C','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'powerful UK national male rapper fusing grime with classic soul harmonics brilliantly, barefoot diva, deeply appealing Korean female vocal drawn from, prodigious genius female'},
    'VM0693': {'cat':'C','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'tireless iron-throated high male rock vocal, underground Korean female rapper crossing hardcore rock, sharp nervous ultra-high'},
    'VM0694': {'cat':'C','tag':'투명한 음색','w':[70, 20, 10],'prompt':'warm intimate male folk pop vocal with gentle rasp, raw powerful black-soul-based, hook-driven addictive male trot'},
    'VM0695': {'cat':'C','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'velvety smooth perfect male R&B ballad vocal with full volume, rapid-fire versatile female rapper with unmatched, lightning-fast male rapper'},
    'VM0696': {'cat':'C','tag':'깊은 베이스','w':[50, 40, 10],'prompt':'mournful mid-bass male trot vocalist commanding orchestral-scale grand ballad narratives, unique delicate Korean female vocal representing a generation, poetic jazz-harmony female'},
    'VM0697': {'cat':'C','tag':'압도적 고음','w':[60, 20, 20],'prompt':'versatile raw male rapper spanning distorted lo-fi beats to tender acoustic rap, doll-faced female trot, emotive building male'},
    'VM0698': {'cat':'C','tag':'그루브 보컬','w':[50, 30, 20],'prompt':'passionate revolutionary male rapper with soul-stirring delivery and poetic intensity, dancer-trained graceful female trot vocalist with clear, silky smooth'},
    'VM0699': {'cat':'C','tag':'허스키 매력','w':[50, 40, 10],'prompt':'overwhelming male R&B lead vocal with rich harmonics, Caribbean-flavored female rapper who effortlessly rides, rugged male rapper blending'},
    'VM0700': {'cat':'C','tag':'거친 소울','w':[60, 20, 20],'prompt':'dreamy alternative male rapper who implanted psychedelic rock sensibility into hip-hop, saddest tone in jazz, pop-ballad crossover female'},
    'VM0701': {'cat':'C','tag':'펑키 그루브','w':[60, 30, 10],'prompt':'pop-rock acoustic male rapper who conquered Billboard with accessible crossover sound, first foreign trot champion female vocalist who, angelic fragile yet'},
    'VM0702': {'cat':'C','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'wordplay-brilliant UK duo male rappers with pop-friendly drill beat chemistry, androgynous cold urban, desperately emotional high-pitched'},
    'VM0703': {'cat':'C','tag':'투명한 음색','w':[40, 40, 20],'prompt':'sky-high angelic male falsetto vocal soaring, R&B-infused female rapper riding 808 glide bass drill, Korean R&B fairy female'},
    'VM0704': {'cat':'C','tag':'압도적 고음','w':[60, 30, 10],'prompt':'sacred powerful metal male vocal with commanding volume from small frame, pansori-master young female trot vocalist melting fierce traditional, crystal-clear healing female'},
    'VM0705': {'cat':'C','tag':'감성 폭발','w':[40, 40, 20],'prompt':'sophisticated mid-century male trot vocalist bridging modern, deep powerful female vocal consuming jazz rock and, ultra-fast UK grime female'},
    'VM0706': {'cat':'C','tag':'리듬 보컬','w':[50, 40, 10],'prompt':'autotune-wielding male rapper who perfected modern melodic trap with hypnotic delivery, deep heavy contralto female vocal singing the, girl-group trained female trot'},
    'VM0707': {'cat':'C','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'energetic flashy male rapper with wild hybrid flow over hardcore club beats, most sophisticated calm sensual mid-low female, new-wave rock female rapper'},
    'VM0708': {'cat':'C','tag':'펑키 그루브','w':[50, 30, 20],'prompt':'authoritative smooth male rapper with business-mogul swagger and effortless delivery, saddest tone in jazz history, wounded soul female, original all-rounder Korean'},
    'VM0709': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'explosive rapid-fire male rapper with razor-sharp diction and unmatched global impact, explosive power from small frame, timeless clear, prodigious genius female'},
    'VM0710': {'cat':'C','tag':'감성 폭발','w':[60, 20, 20],'prompt':'prodigy male trot vocalist mastering saxophone to orchestra with epic narrative depth, emerging Korean drill female, smooth silky male'},
    'VM0711': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'lightning-fast male rapper layering angelic melodies over rapid-fire delivery uniquely, The Voice, perfect female vocal with flawless power pitch, folk-ballad optimized male'},
    'VM0712': {'cat':'C','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, clear bright female pop, glamorous west-coast female'},
    'VM0713': {'cat':'C','tag':'거친 소울','w':[60, 30, 10],'prompt':'rugged bending-note male trot vocalist cutting through grand horn and string ensembles, world-class 5-octave powerful Korean female vocal with dramatic high, folk-ballad optimized male'},
    'VM0714': {'cat':'C','tag':'리듬 보컬','w':[40, 40, 20],'prompt':'triplet-flow male rapper who rewrote global trap, girl-group trained female trot vocalist hiding solid traditional, soft breathy warm'},
    'VM0715': {'cat':'C','tag':'크리스탈 톤','w':[60, 30, 10],'prompt':'bright cheerful male pop vocal with catchy melodic 60s piano flair, sharp organic indie-trap female rapper creating the, heavyweight hardcore boom-bap'},
    'VM0716': {'cat':'C','tag':'중저음 매력','w':[50, 40, 10],'prompt':'thunderous deep-cave male rapper who exploded Brooklyn drill onto the global stage, sorrowful French chanson female vocal pouring raw life, storytelling piano male'},
    'VM0717': {'cat':'C','tag':'허스키 감성','w':[70, 20, 10],'prompt':'raw rough soul-shaking male rock vocal with emotional honesty, punk-rage female rapper pioneering, bright energetic male trot'},
    'VM0718': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'earthy rustic male trot vocalist combining rural folk sentiment with trot tradition, legendary high-tone female rapper with rhythmic agility, polished Atlanta trap'},
    'VM0719': {'cat':'C','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'stable soaring high-note male trot vocalist riding grand traditional Korean melodies, explosive raspy female rapper with southern trap energy, deep literary lyrical'},
    'VM0720': {'cat':'C','tag':'그루브 보컬','w':[40, 40, 20],'prompt':'fleet-footed male rapper with dazzling speed and showmanship, flawless classic female vocal mastering Broadway and, wildly innovative southern male'},
    'VM0721': {'cat':'C','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'versatile raw male rapper spanning distorted lo-fi beats to tender acoustic rap, musical-theater trained female power-trot vocalist with, passionate climbing male'},
    'VM0722': {'cat':'C','tag':'압도적 고음','w':[60, 20, 20],'prompt':'powerful open-throated male trot singer belting folk sorrows with piercing clarity, textbook traditional female trot, pure refreshing female trot'},
    'VM0723': {'cat':'C','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, UK club hyperpop female rapper crossing electronic beats, first lady of jazz,'},
    'VM0724': {'cat':'C','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'classical crossover male trot vocalist harmonizing operatic power with grand orchestral scale, sharp high-tone female rapper with clever off-beat, pansori-master young female trot'},
    'VM0725': {'cat':'C','tag':'허스키 매력','w':[70, 20, 10],'prompt':'husky gravelly male rapper delivering authentic Atlanta street narratives with grit, quiet warm soothing, hypnotic baby-voice male'},
    'VM0726': {'cat':'C','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'silky warm healing gentle male vocal like velvet, competition-bred Korean female, passionate revolutionary male'},
    'VM0727': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'explosive operatic metal male vocal like a, innovative female rapper-producer with revolutionary visual, understated monotone boom-bap'},
    'VM0728': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'uniquely flavored male trot vocalist with signature nasal bending-note technique mastery, poetic jazz-harmony female, relentless southern female'},
    'VM0729': {'cat':'C','tag':'시원한 고음','w':[40, 40, 20],'prompt':'sorrowful falsetto transitioning to angry melodic, Korean R&B fairy female vocal with perfect breath, Afrobeat-drill hybrid female'},
    'VM0730': {'cat':'C','tag':'그루브 보컬','w':[50, 30, 20],'prompt':'fleet-footed male rapper with dazzling speed and showmanship from the golden era, ethereal theatrical falsetto female vocal with, prodigious genius female'},
    'VM0731': {'cat':'C','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'mournful mid-bass male trot vocalist commanding orchestral-scale grand ballad narratives, powerful venue-shaking female, idol-crossover female trot'},
    'VM0732': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'chart-dominating male rapper-singer who demolished the, raw powerful black-soul-based female belting vocal, refined groovy male trot'},
    'VM0733': {'cat':'C','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'raspy warm male rock vocal with anthemic sing-along ballad grit, understated monotone boom-bap, smooth classic'},
    'VM0734': {'cat':'C','tag':'맑은 감성','w':[70, 20, 10],'prompt':'folk-rooted gentle male trot vocalist comforting the nation with plain warm delivery, revolutionary female rapper, dreamy sophisticated falsetto'},
    'VM0735': {'cat':'C','tag':'압도적 고음','w':[60, 30, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, androgynous cold urban charismatic female vocal, sharp nervous ultra-high'},
    'VM0736': {'cat':'C','tag':'압도적 고음','w':[50, 30, 20],'prompt':'rhythmic all-rounder male trot vocalist with powerful diction and stage-breaking energy, rich-volume female cinematic trot vocalist standing firm, explosive operatic metal'},
    'VM0737': {'cat':'C','tag':'허스키 감성','w':[60, 20, 20],'prompt':'explosive raspy male trot vocalist with gut-wrenching sorrow and raw emotional power, punk-rage female rapper pioneering, powerful venue-shaking female'},
    'VM0738': {'cat':'C','tag':'일렉트로닉','w':[60, 30, 10],'prompt':'rapid-fire technical male rapper balancing speed with accessible pop-ballad sensibility, husky soulful female vocal fusing hip-hop and, most sophisticated calm'},
    'VM0739': {'cat':'C','tag':'리듬 보컬','w':[70, 20, 10],'prompt':'solid high male vocal with retro and modern groove, energetic performer, deep-voiced southern female rapper, dramatic operatic male'},
    'VM0740': {'cat':'C','tag':'투명한 음색','w':[70, 20, 10],'prompt':'versatile male vocal from soft falsetto to rock screaming, deep husky soulful, highway queen female trot'},
    'VM0741': {'cat':'C','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'cold atmospheric male rapper delivering chilling street narratives with west-coast cool, deep powerful female vocal, cute nasally charming'},
    'VM0742': {'cat':'C','tag':'허스키 매력','w':[50, 30, 20],'prompt':'rough torn raspy male vocal pouring soul until the last breath, fierce Miami trap duo female rapper with unrestrained, pansori-based male trot'},
    'VM0743': {'cat':'C','tag':'크리스탈 톤','w':[60, 30, 10],'prompt':'transparent fragile male vocal with crystalline sad tone and quiet intensity, fierce female rapper from Ruff Ryders dominating 2000s, world-class 5-octave powerful Korean'},
    'VM0744': {'cat':'C','tag':'그루브 보컬','w':[40, 40, 20],'prompt':'heart-wrenching melodic male rapper who epitomized, brilliant nightingale female trot vocalist celebrated as the, rapid-fire technical male'},
    'VM0745': {'cat':'C','tag':'허스키 매력','w':[70, 20, 10],'prompt':'overwhelming male R&B lead vocal with rich harmonics, fierce female rapper from, androgynous cold urban'},
    'VM0746': {'cat':'C','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'solid high male vocal with retro and modern groove, textbook female trot vocalist with decades of live, multi-talented male rapper-actor'},
    'VM0747': {'cat':'C','tag':'투명한 음색','w':[60, 20, 20],'prompt':'quintessentially Korean optimistic male trot vocalist with earthy rustic warmth and joy, perfect powerful female R&B, refined female trot vocalist'},
    'VM0748': {'cat':'C','tag':'댄스 보컬','w':[40, 40, 20],'prompt':'master-architect male trot vocalist radically mixing pansori,, technically versatile Korean female rapper freely riding R&B, globally acclaimed UK'},
    'VM0749': {'cat':'C','tag':'파워 보컬','w':[50, 30, 20],'prompt':'overwhelming falsetto high male Korean ballad vocal dominating karaoke, flawless classic female vocal mastering Broadway and, bold thick-toned Korean'},
    'VM0750': {'cat':'C','tag':'감성 보컬','w':[70, 20, 10],'prompt':'smooth classic baritone male crooner jazz pop vocal, rich-volume female cinematic, saddest tone in jazz'},
    'VM0751': {'cat':'D','tag':'맑은 감성','w':[50, 30, 20],'prompt':'crystalline pure-toned male trot tenor revered as the emperor of classic Korean enka, eccentric brilliant male rapper commanding neo-soul, husky deep mid-low female'},
    'VM0752': {'cat':'D','tag':'감성 보컬','w':[60, 20, 20],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, raw aching male piano, delicate symphonic metal'},
    'VM0753': {'cat':'D','tag':'감성 폭발','w':[60, 20, 20],'prompt':'lush romantic male vocal blending classical piano grandeur with pop yearning, soulful mezzo-soprano with, quintessentially Korean optimistic male'},
    'VM0754': {'cat':'D','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'solid high male vocal with retro and modern groove, energetic performer, soft gentle tenor, the living goddess of'},
    'VM0755': {'cat':'D','tag':'맑은 감성','w':[50, 30, 20],'prompt':'pristine classical crossover soprano with angelic symphonic serenity, explosive powerhouse belter with massive, relaxed mellow baritone'},
    'VM0756': {'cat':'D','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'brilliant nightingale female trot vocalist celebrated as the golden voice of the 50s-60s, melodic hook-master male vocalist who defined G-Funk, smooth romantic male rapper'},
    'VM0757': {'cat':'D','tag':'파워 보컬','w':[50, 40, 10],'prompt':'powerful heartfelt classic pop male vocal with piano accompaniment, paradigm-shifting male rapper-producer and hip-hop genre, legendary nasal-melody female trot'},
    'VM0758': {'cat':'D','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'explosive power from small frame, timeless clear sorrowful Korean female vocal, anthem trance, heart-wrenching, bold theatrical baritone'},
    'VM0759': {'cat':'D','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'classically trained powerful high-note male trot vocalist who launched power-trot era, modern male rapper, velvety deep crooning'},
    'VM0760': {'cat':'D','tag':'감성 폭발','w':[50, 30, 20],'prompt':'silky smooth sensual male Motown soul vocal, warm intimate male folk pop vocal, explosive female disco'},
    'VM0761': {'cat':'D','tag':'천상의 목소리','w':[50, 40, 10],'prompt':'elegant 60s female trot vocalist layering sophisticated arrangements over traditional melody, soft gentle tenor with intimate breathy delivery, genius sensual male vocal'},
    'VM0762': {'cat':'D','tag':'폭발 에너지','w':[50, 40, 10],'prompt':'powerful UK national male rapper fusing grime with classic soul harmonics brilliantly, dramatic powerful male rock vocal with 4-octave theatrical, poetic conscious male rapper'},
    'VM0763': {'cat':'D','tag':'중저음 매력','w':[60, 20, 20],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, agile scatting tenor, cerebral eloquent male'},
    'VM0764': {'cat':'D','tag':'일렉트로닉','w':[60, 30, 10],'prompt':'solid high male vocal with retro and modern groove, multi-talented male rapper-producer with refined west-coast, clear bright female pop'},
    'VM0765': {'cat':'D','tag':'압도적 고음','w':[60, 30, 10],'prompt':'explosive power from small frame, timeless clear sorrowful Korean female vocal, precise rhythmic Swedish diva, Clean Bandit, airy ethereal male'},
    'VM0766': {'cat':'D','tag':'댄스 보컬','w':[60, 30, 10],'prompt':'legendary Three 6 Mafia female rapper ruling southern underground with fierce authority, positive upbeat female dance-trot vocalist who pioneered rhythmic, technically gifted male'},
    'VM0767': {'cat':'D','tag':'일렉트로닉','w':[60, 30, 10],'prompt':'young luminous classical crossover soprano with operatic innocence and grace, fierce female rapper from Ruff Ryders dominating 2000s, cute nasally charming'},
    'VM0768': {'cat':'D','tag':'크리스탈 톤','w':[50, 30, 20],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, pioneering male rapper mastering both hardcore hip-hop, world-class soprano with'},
    'VM0769': {'cat':'D','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, revolutionary female rapper, smooth R&B singing over'},
    'VM0770': {'cat':'D','tag':'중저음 매력','w':[60, 20, 20],'prompt':'charming deep baritone male vocal, the king of rock and roll, distinctive high-pitched male, suave romantic baritone'},
    'VM0771': {'cat':'D','tag':'파워 보컬','w':[60, 20, 20],'prompt':'classical crossover male trot vocalist harmonizing operatic power with grand orchestral scale, punk-rock crossover male, sophisticated folk soprano'},
    'VM0772': {'cat':'D','tag':'감성 보컬','w':[40, 40, 20],'prompt':'brilliant nightingale female trot vocalist celebrated as the, rough raspy alto with raw heartfelt, hard-hitting precise male'},
    'VM0773': {'cat':'D','tag':'감성 보컬','w':[50, 30, 20],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, pioneering male rapper who defined modern rhyme, gritty warm male'},
    'VM0774': {'cat':'D','tag':'거친 소울','w':[70, 20, 10],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, rhythmic male pop, rustic mid-low bending-note'},
    'VM0775': {'cat':'D','tag':'그루브 보컬','w':[50, 30, 20],'prompt':'legendary Three 6 Mafia female rapper ruling southern underground with fierce authority, heavy dubstep-trap hybrid, dark aggressive bass,, explosive operatic metal'},
    'VM0776': {'cat':'D','tag':'거친 소울','w':[50, 30, 20],'prompt':'elegant 60s female trot vocalist layering sophisticated arrangements over traditional melody, melodic hook-master male vocalist who defined G-Funk, refined female trot vocalist'},
    'VM0777': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'retro 8-bit electro-trap pioneer, funky chiptune rebel, original genre bender, heavy 808 trap EDM,, deep powerful female'},
    'VM0778': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'explosive power from small frame, timeless clear sorrowful Korean female vocal, foundational male rapper who architected modern trap, ethereal theatrical falsetto'},
    'VM0779': {'cat':'D','tag':'시원한 고음','w':[60, 20, 20],'prompt':'pristine classical crossover soprano with angelic symphonic serenity, powerfully raspy male, delicate lyrical tenor'},
    'VM0780': {'cat':'D','tag':'감성 폭발','w':[60, 20, 20],'prompt':'legendary harmony female trot vocalist showcasing textbook traditional duet vocal mastery, warm intimate male, smooth R&B singing over'},
    'VM0781': {'cat':'D','tag':'펑키 그루브','w':[60, 30, 10],'prompt':'legendary harmony female trot vocalist showcasing textbook traditional duet vocal mastery, revolutionary male rapper who weaponized his voice as, rapid-fire versatile female'},
    'VM0782': {'cat':'D','tag':'파워 보컬','w':[60, 20, 20],'prompt':'majestic operatic tenor with soaring classical purity and grand resonance, commanding dramatic diva, haunting atmospheric'},
    'VM0783': {'cat':'D','tag':'감성 폭발','w':[60, 30, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery that makes the world cry, accessible narrative male rapper layering popular, lyrical light tenor'},
    'VM0784': {'cat':'D','tag':'감성 폭발','w':[70, 20, 10],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, rhythmic male pop, The Voice, perfect female'},
    'VM0785': {'cat':'D','tag':'파워 보컬','w':[70, 20, 10],'prompt':'elegant refined tenor with classically elevated harmonic vocal phrasing, inventive creative female rapper, feathery high tenor'},
    'VM0786': {'cat':'D','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'wise philosophical male rapper layering classic boom-bap soul with modern narratives, wordplay-brilliant UK duo male rappers with, passionate revolutionary male'},
    'VM0787': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, feathery soft tender male vocal with, NYC underground queen'},
    'VM0788': {'cat':'D','tag':'깊은 울림','w':[70, 20, 10],'prompt':'charming deep baritone male vocal, the king of rock and roll, calm low mid-range, laid-back mellow baritone'},
    'VM0789': {'cat':'D','tag':'거친 소울','w':[60, 30, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery that makes the world cry, desperately emotional high-pitched male Korean vocal, creative fusion soprano'},
    'VM0790': {'cat':'D','tag':'감성 보컬','w':[60, 30, 10],'prompt':'refined French chanteuse with classic cinematic vocal elegance, raw desperate soprano with unfiltered emotional intensity, solid high male'},
    'VM0791': {'cat':'D','tag':'투명한 음색','w':[50, 30, 20],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, warm versatile mezzo with theatrical, revolutionary male rapper who'},
    'VM0792': {'cat':'D','tag':'거친 소울','w':[60, 30, 10],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, barefoot diva, deeply appealing Korean female vocal drawn from, polished hybrid male trot'},
    'VM0793': {'cat':'D','tag':'크리스탈 톤','w':[70, 20, 10],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, humorous witty male trot, nervous yet beautiful'},
    'VM0794': {'cat':'D','tag':'묵직한 저음','w':[70, 20, 10],'prompt':'charming deep baritone male vocal, the king of rock and roll, raw angular mezzo, doll-faced female trot'},
    'VM0795': {'cat':'D','tag':'감성 보컬','w':[40, 40, 20],'prompt':'deep classic husky female vocal with, powerful rough soulful female blues vocal, crystal clear yet'},
    'VM0796': {'cat':'D','tag':'깊은 울림','w':[60, 20, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, polished velvety, mellow melodic male rapper'},
    'VM0797': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'classical crossover male trot vocalist harmonizing operatic power with grand orchestral scale, folk-rock gentle female trot, husky theatrical baritone'},
    'VM0798': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'elegant silken high tenor with refined classical pop gentle sophistication, razor-sharp punchline male rapper with dazzling lyrical, soft dreamy'},
    'VM0799': {'cat':'D','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'solid high male vocal with retro and modern groove, energetic performer, fierce sharp-tongued New York female rapper, legendary harmony female'},
    'VM0800': {'cat':'D','tag':'일렉트로닉','w':[60, 30, 10],'prompt':'legendary high-tone female rapper with rhythmic agility and iconic vocal presence, rhythmic groove master male trot vocalist who electrified, agile scatting tenor'},
    'VM0801': {'cat':'D','tag':'투명한 음색','w':[60, 30, 10],'prompt':'young luminous classical crossover soprano with operatic innocence and grace, sharp nervous ultra-high screaming male rock, wise philosophical male'},
    'VM0802': {'cat':'D','tag':'천상의 목소리','w':[60, 30, 10],'prompt':'elegant silken high tenor with refined classical pop gentle sophistication, warm low-register male trot vocalist evoking hometown, clear pristine soprano'},
    'VM0803': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'powerful heartfelt classic pop male vocal with piano accompaniment, soft dreamy, gravelly soulful'},
    'VM0804': {'cat':'D','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'elegant silken high tenor with refined classical pop gentle sophistication, The Voice, perfect female vocal with flawless, sharp organic indie-trap'},
    'VM0805': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'charismatic retro tenor with vintage seventies pop-rock warmth and flair, underground technical female, whisper-soft literary female rapper'},
    'VM0806': {'cat':'D','tag':'굵은 바리톤','w':[60, 20, 20],'prompt':'rustic mid-low bending-note female trot vocalist anchoring legendary harmony foundations, authentic R&B alto, barefoot diva, deeply appealing'},
    'VM0807': {'cat':'D','tag':'파워 보컬','w':[40, 40, 20],'prompt':'classically trained powerful high-note male trot vocalist, commanding female rapper who elevated hip-hop with social, blended operatic tenor'},
    'VM0808': {'cat':'D','tag':'펑키 그루브','w':[50, 30, 20],'prompt':'legendary harmony female trot vocalist showcasing textbook traditional duet vocal mastery, explosive Canadian male rapper-singer with 80s-90s, lightning-fast male rapper'},
    'VM0809': {'cat':'D','tag':'압도적 고음','w':[70, 20, 10],'prompt':'elegant refined tenor with classically elevated harmonic vocal phrasing, iconic pop female vocal, sharp sorrowful tenor'},
    'VM0810': {'cat':'D','tag':'맑은 감성','w':[40, 40, 20],'prompt':'pure clean soprano capturing quiet depth, explosive belting soprano with piercing volume, transparent dewdrop-clear soprano'},
    'VM0811': {'cat':'D','tag':'투명한 음색','w':[60, 20, 20],'prompt':'crystalline nightingale female trot vocalist who dominated early classic Korean trot, bouncy yet heartfelt, breathy dreamy falsetto'},
    'VM0812': {'cat':'D','tag':'감성 보컬','w':[60, 20, 20],'prompt':'soulful male vocal with deep pain and emotion, the godfather of soul, theatrical sweeping male, world-class 5-octave powerful Korean'},
    'VM0813': {'cat':'D','tag':'투명한 음색','w':[50, 40, 10],'prompt':'pristine classical crossover soprano with angelic symphonic serenity, highway queen female trot vocalist with tender yet, trend-setting UK drill'},
    'VM0814': {'cat':'D','tag':'거친 소울','w':[70, 20, 10],'prompt':'bright cheerful male pop vocal with catchy melodic 60s piano flair, prodigious genius female, sorrowful French chanson female'},
    'VM0815': {'cat':'D','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'legendary 80s female rapper who spearheaded hip-hop mainstream with infectious energy, enka-trot queen female vocalist who conquered both Japan, revolutionary UK grime male'},
    'VM0816': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, plaintive gentle male trot vocal soothing homesick hearts, rustic mid-low bending-note'},
    'VM0817': {'cat':'D','tag':'파워 보컬','w':[70, 20, 10],'prompt':'powerful UK national male rapper fusing grime with classic soul harmonics brilliantly, funky freewheeling male, unwavering cool'},
    'VM0818': {'cat':'D','tag':'굵은 바리톤','w':[50, 40, 10],'prompt':'rustic mid-low bending-note female trot vocalist anchoring legendary harmony foundations, warm folk acoustic Korean male vocal, dreamy Latin-pop female'},
    'VM0819': {'cat':'D','tag':'허스키 매력','w':[60, 20, 20],'prompt':'rustic mid-low bending-note female trot vocalist anchoring legendary harmony foundations, cinematic male trot vocalist, legendary harmony female'},
    'VM0820': {'cat':'D','tag':'거친 소울','w':[70, 20, 10],'prompt':'textbook traditional female trot vocalist with the most flavorful classic bending delivery, youthful clear tenor, gentle wistful male'},
    'VM0821': {'cat':'D','tag':'펑키 그루브','w':[50, 30, 20],'prompt':'solid high male vocal with retro and modern groove, transparent earnest soprano with heartfelt, gritty soulful'},
    'VM0822': {'cat':'D','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'classical soprano female vocal pioneering symphonic metal genre, crystal clear yet steel-strong female belting, atmospheric Canadian male'},
    'VM0823': {'cat':'D','tag':'깊은 베이스','w':[60, 20, 20],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, dark European drill female, smoky low alto'},
    'VM0824': {'cat':'D','tag':'압도적 고음','w':[60, 20, 20],'prompt':'explosive power from small frame, timeless clear sorrowful Korean female vocal, multi-genre soprano with, sweet lyrical soprano'},
    'VM0825': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'legendary storytelling male rapper with unique accent and theatrical narrative flow, androgynous cold urban, explosive power from'},
    'VM0826': {'cat':'D','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'classical crossover male trot vocalist harmonizing operatic power, intense 90s New York hardcore female rapper, dignified low-tone male trot'},
    'VM0827': {'cat':'D','tag':'리듬 보컬','w':[70, 20, 10],'prompt':'legendary Three 6 Mafia female rapper ruling southern underground with fierce authority, ethereal breathtaking soprano, global Billboard-hitting female'},
    'VM0828': {'cat':'D','tag':'허스키 감성','w':[50, 40, 10],'prompt':'legendary nasal-melody female trot vocalist who comforted a colonized nation with sorrow, clear smooth pure falsetto, heavy dubstep-trap hybrid,'},
    'VM0829': {'cat':'D','tag':'크리스탈 톤','w':[70, 20, 10],'prompt':'crystalline pure-toned male trot tenor revered as the emperor of classic Korean enka, inventive creative female rapper, deep emotional dramatic'},
    'VM0830': {'cat':'D','tag':'압도적 고음','w':[50, 40, 10],'prompt':'explosive power from small frame, timeless clear sorrowful Korean female vocal, powerful rich baritone-tenor with sweeping orchestral, crystalline nightingale female'},
    'VM0831': {'cat':'D','tag':'감성 폭발','w':[50, 40, 10],'prompt':'brilliant nightingale female trot vocalist celebrated as the golden voice of the 50s-60s, raw aching male piano vocal that erupts from, dreamy sophisticated'},
    'VM0832': {'cat':'D','tag':'천상의 목소리','w':[50, 40, 10],'prompt':'legendary pure high male tenor with effortless sustained arena rock notes, feathery soft tender male vocal with, flawless crystal clear'},
    'VM0833': {'cat':'D','tag':'거친 소울','w':[40, 40, 20],'prompt':'legendary harmony female trot vocalist showcasing textbook, underground legend male rapper who built southern, thunderous deep-cave male rapper'},
    'VM0834': {'cat':'D','tag':'폭발 에너지','w':[50, 40, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, hook-driven addictive male trot vocalist dominating with catchy, legendary harmony female'},
    'VM0835': {'cat':'D','tag':'시원한 고음','w':[40, 40, 20],'prompt':'powerful UK national male rapper fusing grime with, overwhelming falsetto high male Korean ballad, majestic operatic tenor'},
    'VM0836': {'cat':'D','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, passionate tender tenor with soulful Latin, anthem trance, heart-wrenching'},
    'VM0837': {'cat':'D','tag':'파워 보컬','w':[70, 20, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, rapid-fire versatile female, flawless technique male'},
    'VM0838': {'cat':'D','tag':'감성 폭발','w':[60, 30, 10],'prompt':'legendary harmony female trot vocalist showcasing textbook traditional duet vocal mastery, foundational male rapper who architected modern trap, soft dreamy'},
    'VM0839': {'cat':'D','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'most destructive female rock vocal in, plaintive gentle male trot vocal soothing homesick hearts, transcendent tenor with'},
    'VM0840': {'cat':'D','tag':'강렬한 샤우팅','w':[40, 40, 20],'prompt':'powerful heartfelt classic pop male vocal, wildly innovative southern male rapper and hip-hop genre, refreshing clear soprano'},
    'VM0841': {'cat':'D','tag':'중저음 매력','w':[50, 30, 20],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, sweet romantic tenor male trot vocalist with handsome, rhythmic male pop'},
    'VM0842': {'cat':'D','tag':'일렉트로닉','w':[40, 40, 20],'prompt':'classical soprano female vocal pioneering, intellectually refined male rapper blending conscious, Miami hardcore female rapper'},
    'VM0843': {'cat':'D','tag':'깊은 베이스','w':[60, 20, 20],'prompt':'velvety deep crooning baritone with effortless vintage big band warmth, next-generation hardcore female, Miss Trot champion female'},
    'VM0844': {'cat':'D','tag':'감성 보컬','w':[50, 30, 20],'prompt':'flawless classic female vocal mastering Broadway and pop with zero error, husky powerful soprano with overwhelming, underground technical female'},
    'VM0845': {'cat':'D','tag':'일렉트로닉','w':[50, 40, 10],'prompt':'legendary 80s female rapper who spearheaded hip-hop mainstream with infectious energy, deep-voiced southern female rapper with heavy 808 impact, London-born healing female'},
    'VM0846': {'cat':'D','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'legendary nasal-melody female trot vocalist who comforted a colonized nation with sorrow, Afrobeat-drill fusion male rapper who pioneered Afroswing, explosive female disco'},
    'VM0847': {'cat':'D','tag':'폭발 에너지','w':[40, 40, 20],'prompt':'soaring rock soprano with legendary, intimate whispery baritone with atmospheric, powerful rough soulful'},
    'VM0848': {'cat':'D','tag':'폭발 에너지','w':[50, 30, 20],'prompt':'queen of soul, gospel-based explosive powerful female vocal full of holy spirit, bright energetic male trot vocalist radiating vitality with, rapid-fire technical male'},
    'VM0849': {'cat':'D','tag':'압도적 고음','w':[60, 30, 10],'prompt':'legendary pure high male tenor with effortless sustained arena rock notes, powerful pansori-toned female trot vocalist who made the, deep resonant female trot'},
    'VM0850': {'cat':'D','tag':'펑키 그루브','w':[60, 30, 10],'prompt':'solid high male vocal with retro and modern groove, pure crystal-clear folk soprano with gentle, refined French'},
    'VM0851': {'cat':'D','tag':'감성 보컬','w':[60, 30, 10],'prompt':'lush romantic male vocal blending classical piano grandeur with pop yearning, evolved male vocal reaching divine territory from folk, Australian-born female rapper'},
    'VM0852': {'cat':'D','tag':'맑은 천상','w':[40, 40, 20],'prompt':'young luminous classical crossover soprano with, bright narrative acoustic pop vocal with, rich deep mid-low'},
    'VM0853': {'cat':'D','tag':'댄스 보컬','w':[70, 20, 10],'prompt':'legendary 80s female rapper who spearheaded hip-hop mainstream with infectious energy, technically versatile Korean female, explosive raspy female rapper'},
    'VM0854': {'cat':'D','tag':'리듬 보컬','w':[50, 30, 20],'prompt':'retro 8-bit electro-trap pioneer, funky chiptune rebel, original genre bender, refined velvety tenor with elegant soaring, nervous yet beautiful'},
    'VM0855': {'cat':'D','tag':'투명한 음색','w':[70, 20, 10],'prompt':'young luminous classical crossover soprano with operatic innocence and grace, uniquely flavored male, girl-group trained female trot'},
    'VM0856': {'cat':'D','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'solid high male vocal with retro and modern groove, global Billboard-hitting female rapper with Thai-international swagger, cute nasally charming'},
    'VM0857': {'cat':'D','tag':'크리스탈 톤','w':[40, 40, 20],'prompt':'pristine classical crossover soprano with, underground legend male rapper who built southern, refined slightly nasal'},
    'VM0858': {'cat':'D','tag':'천상의 목소리','w':[60, 30, 10],'prompt':'most destructive female rock vocal in history, blood-vessel-popping screaming wail, warm intimate male folk pop vocal, modern healing tenor'},
    'VM0859': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, clear bright female pop vocal with ultra-high technique, laid-back mellow baritone'},
    'VM0860': {'cat':'D','tag':'압도적 고음','w':[60, 20, 20],'prompt':'legendary pure high male tenor with effortless sustained arena rock notes, delicate symphonic metal, polished velvety'},
    'VM0861': {'cat':'D','tag':'맑은 천상','w':[60, 20, 20],'prompt':'bright cheerful male pop vocal with catchy melodic 60s piano flair, soft dreamy, transcendent tenor with'},
    'VM0862': {'cat':'D','tag':'투명한 음색','w':[60, 20, 20],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, pansori-based male trot, genius sensual male vocal'},
    'VM0863': {'cat':'D','tag':'깊은 베이스','w':[60, 20, 20],'prompt':'smooth classic baritone male crooner jazz pop vocal, global Billboard-hitting female, raspy warm male'},
    'VM0864': {'cat':'D','tag':'리듬 보컬','w':[40, 40, 20],'prompt':'solid high male vocal with retro, raw rough soul-shaking male rock vocal, pure crystal-clear folk'},
    'VM0865': {'cat':'D','tag':'감성 폭발','w':[40, 40, 20],'prompt':'elegant 60s female trot vocalist layering sophisticated, trendy stylish male rapper layering fashion-forward aesthetics, precise rhythmic Swedish'},
    'VM0866': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'earnest warm tenor with pure heartfelt delivery and classic ballad phrasing, precisely polished UK drill, modern male rapper'},
    'VM0867': {'cat':'D','tag':'폭발 에너지','w':[40, 40, 20],'prompt':'majestic operatic tenor with soaring classical, raw explosive belting vocal tearing through, globally acclaimed UK'},
    'VM0868': {'cat':'D','tag':'압도적 고음','w':[70, 20, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, the living goddess of, heavy dubstep-trap hybrid,'},
    'VM0869': {'cat':'D','tag':'감성 보컬','w':[50, 40, 10],'prompt':'brilliant nightingale female trot vocalist celebrated as the golden voice of the 50s-60s, trembling soulful male vocal with aching falsetto, smooth classic'},
    'VM0870': {'cat':'D','tag':'거친 소울','w':[60, 20, 20],'prompt':'soulful male vocal with deep pain and emotion, the godfather of soul, genius lyricist UK, percussion-performing male trot'},
    'VM0871': {'cat':'D','tag':'천상의 목소리','w':[60, 20, 20],'prompt':'elegant 60s female trot vocalist layering sophisticated arrangements over traditional melody, minimal clear soprano, refreshing bright rock'},
    'VM0872': {'cat':'D','tag':'깊은 베이스','w':[40, 40, 20],'prompt':'rebellious melancholic raw retro soul jazz female vocal,, immortal deep baritone male trot vocalist who elevated, deep heavy contralto'},
    'VM0873': {'cat':'D','tag':'굵은 바리톤','w':[40, 40, 20],'prompt':'velvety deep crooning baritone with effortless, technically gifted male rapper with extraordinary rhyme, natural conversational mid-range'},
    'VM0874': {'cat':'D','tag':'묵직한 저음','w':[50, 30, 20],'prompt':'charming deep baritone male vocal, the king of rock and roll, distinctive husky low-tone male rapper with gritty, bold theatrical baritone'},
    'VM0875': {'cat':'D','tag':'맑은 감성','w':[50, 40, 10],'prompt':'crystalline pure-toned male trot tenor revered as the emperor of classic Korean enka, romantic aged baritone with weathered folk, explosive energy rough husky'},
    'VM0876': {'cat':'D','tag':'깊은 울림','w':[60, 20, 20],'prompt':'deep grand baritone with rich harmonic resonance and classical vocal control, powerfully raspy male, wailing blues-rock male'},
    'VM0877': {'cat':'D','tag':'압도적 고음','w':[40, 40, 20],'prompt':'powerful heartfelt classic pop male vocal, operatic tenor male trot vocalist completing orchestral-scale, inventive jazzy soprano'},
    'VM0878': {'cat':'D','tag':'일렉트로닉','w':[60, 20, 20],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, trailblazing New York female, precision-engineered modern male'},
    'VM0879': {'cat':'D','tag':'감성 보컬','w':[50, 30, 20],'prompt':'silky smooth sensual male Motown soul vocal, smooth R&B singing over 808 glide bass trap, legendary harmony female'},
    'VM0880': {'cat':'D','tag':'천상의 목소리','w':[60, 20, 20],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, smooth urban tenor, genius singer-songwriter male'},
    'VM0881': {'cat':'D','tag':'파워 보컬','w':[50, 40, 10],'prompt':'soaring rock soprano with legendary high-range stadium power, rich earthy male trot baritone comforting working-class, smooth romantic male rapper'},
    'VM0882': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 30, 20],'prompt':'charismatic retro tenor with vintage seventies pop-rock warmth and flair, warm versatile mezzo with theatrical, prodigy male trot vocalist'},
    'VM0883': {'cat':'D','tag':'파워 보컬','w':[50, 30, 20],'prompt':'elegant silken high tenor with refined classical pop gentle sophistication, theatrical sweeping male piano ballad vocal, whisper-soft literary female rapper'},
    'VM0884': {'cat':'D','tag':'시원한 고음','w':[50, 30, 20],'prompt':'most destructive female rock vocal in history, blood-vessel-popping screaming wail, multi-genre soprano with piercing high notes, classical crossover male trot'},
    'VM0885': {'cat':'D','tag':'파워 보컬','w':[50, 30, 20],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, revolutionary male rapper who weaponized his voice as, Korean R&B fairy female'},
    'VM0886': {'cat':'D','tag':'맑은 천상','w':[70, 20, 10],'prompt':'pristine classical crossover soprano with angelic symphonic serenity, pure crystalline tenor, trendy urban mezzo'},
    'VM0887': {'cat':'D','tag':'압도적 고음','w':[70, 20, 10],'prompt':'majestic operatic tenor with soaring classical purity and grand resonance, dance anthem, creative fusion soprano'},
    'VM0888': {'cat':'D','tag':'리듬 보컬','w':[60, 30, 10],'prompt':'legendary Three 6 Mafia female rapper ruling southern underground with fierce authority, airy ethereal male falsetto vocal building, Brooklyn deep-voiced female'},
    'VM0889': {'cat':'D','tag':'감성 보컬','w':[50, 40, 10],'prompt':'legendary nasal-melody female trot vocalist who comforted a colonized nation with sorrow, smooth R&B singing over 808 glide bass trap, textbook traditional female trot'},
    'VM0890': {'cat':'D','tag':'허스키 감성','w':[50, 40, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery that makes the world cry, bold thick-toned Korean female rapper anchoring songs, quirky playful'},
    'VM0891': {'cat':'D','tag':'펑키 그루브','w':[40, 40, 20],'prompt':'husky intelligent female R&B vocal with, explosive Canadian male rapper-singer with 80s-90s, humorous witty male trot'},
    'VM0892': {'cat':'D','tag':'천상의 목소리','w':[50, 30, 20],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, most sophisticated calm sensual mid-low female, breathy dreamy falsetto'},
    'VM0893': {'cat':'D','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'polished swinging male crooner vocal with warm big-band jazz tone, precise surgical male rapper with clean triplet, mellow melodic male rapper'},
    'VM0894': {'cat':'D','tag':'굵은 바리톤','w':[40, 40, 20],'prompt':'rustic mid-low bending-note female trot vocalist, lethal off-beat female rapper-singer delivering devastating, punchy dynamic male'},
    'VM0895': {'cat':'D','tag':'거친 소울','w':[70, 20, 10],'prompt':'legendary nasal-melody female trot vocalist who comforted a colonized nation with sorrow, edgy youthful, ice-cold sad rebellious'},
    'VM0896': {'cat':'D','tag':'투명한 음색','w':[60, 20, 20],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, angsty alternative, unique bright indie'},
    'VM0897': {'cat':'D','tag':'그루브 보컬','w':[40, 40, 20],'prompt':'solid high male vocal with retro and, tireless iron-throated high male rock vocal, deep-voiced southern female rapper'},
    'VM0898': {'cat':'D','tag':'허스키 매력','w':[50, 30, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, dreamy breathy contralto with cinematic, textbook female trot vocalist'},
    'VM0899': {'cat':'D','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'wise philosophical male rapper layering classic boom-bap soul with modern narratives, warm intimate male folk pop vocal, husky powerful male'},
    'VM0900': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'earnest warm tenor with pure heartfelt delivery and classic ballad phrasing, UK club hyperpop female, rapid-fire technical male'},
    'VM0901': {'cat':'D','tag':'파워 보컬','w':[60, 30, 10],'prompt':'powerful UK national male rapper fusing grime with classic soul harmonics brilliantly, gangster-crew male rappers carrying west-coast and global, tireless iron-throated high'},
    'VM0902': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'classically trained powerful high-note male trot vocalist who launched power-trot era, breezy light acoustic vocal with sunny, autotune-wielding male rapper'},
    'VM0903': {'cat':'D','tag':'리듬 보컬','w':[50, 30, 20],'prompt':'rustic mid-low bending-note female trot vocalist anchoring legendary harmony foundations, nasal high-pitched male rapper who pioneered, Elvis-inspired charismatic male'},
    'VM0904': {'cat':'D','tag':'폭발 에너지','w':[60, 20, 20],'prompt':'powerful heartfelt classic pop male vocal with piano accompaniment, tireless iron-throated high, power pop-rock EDM'},
    'VM0905': {'cat':'D','tag':'감성 보컬','w':[40, 40, 20],'prompt':'lush romantic male vocal blending classical piano, screaming high tenor with razor-sharp power, fleet-footed male rapper with'},
    'VM0906': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'classical soprano female vocal pioneering symphonic metal genre, transparent fragile male vocal with crystalline sad, explosive crunk-pioneer male'},
    'VM0907': {'cat':'D','tag':'감성 폭발','w':[60, 30, 10],'prompt':'brilliant nightingale female trot vocalist celebrated as the golden voice of the 50s-60s, authoritative male rapper who defined boom-bap, delicate lyrical tenor'},
    'VM0908': {'cat':'D','tag':'리듬 보컬','w':[40, 40, 20],'prompt':'legendary 80s female rapper who spearheaded hip-hop, deep powerful female vocal consuming jazz rock and, refreshing bright rock'},
    'VM0909': {'cat':'D','tag':'맑은 감성','w':[50, 40, 10],'prompt':'pure clean soprano capturing quiet depth with timeless graceful phrasing, stable powerhouse female trot vocalist with the most, folk-rock gentle female trot'},
    'VM0910': {'cat':'D','tag':'크리스탈 톤','w':[40, 40, 20],'prompt':'polished warm soprano with elegant 60s, NYC underground queen female rapper embodying alternative, gritty charismatic male'},
    'VM0911': {'cat':'D','tag':'굵은 바리톤','w':[50, 30, 20],'prompt':'rustic mid-low bending-note female trot vocalist anchoring legendary harmony foundations, mournful mid-bass male trot vocalist commanding, smooth romantic male rapper'},
    'VM0912': {'cat':'D','tag':'감성 폭발','w':[50, 30, 20],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, relentless southern female rapper with heavyweight, silky warm'},
    'VM0913': {'cat':'D','tag':'거친 소울','w':[60, 20, 20],'prompt':'bright cheerful male pop vocal with catchy melodic 60s piano flair, distinctive husky baritone, gritty soulful'},
    'VM0914': {'cat':'D','tag':'일렉트로닉','w':[50, 30, 20],'prompt':'powerful heartfelt classic pop male vocal with piano accompaniment, perfect pitch hybrid vocal, Zedd Stay, 2NE1 hardcore K-pop female'},
    'VM0915': {'cat':'D','tag':'거친 소울','w':[70, 20, 10],'prompt':'husky intelligent female R&B vocal with classic piano accompaniment, pop-rock acoustic male, deep literary lyrical'},
    'VM0916': {'cat':'D','tag':'허스키 감성','w':[50, 40, 10],'prompt':'legendary harmony female trot vocalist showcasing textbook traditional duet vocal mastery, creative fusion soprano blending traditional Korean vocal, genius singer-songwriter male'},
    'VM0917': {'cat':'D','tag':'리듬 보컬','w':[60, 20, 20],'prompt':'solid high male vocal with retro and modern groove, energetic performer, raw powerful black-soul-based, warm intimate male'},
    'VM0918': {'cat':'D','tag':'폭발 에너지','w':[60, 20, 20],'prompt':'legendary high-tone female rapper with rhythmic agility and iconic vocal presence, sorrowful falsetto transitioning, The Voice, perfect female'},
    'VM0919': {'cat':'D','tag':'펑키 그루브','w':[70, 20, 10],'prompt':'husky intelligent female R&B vocal with classic piano accompaniment, next-generation hardcore female, legendary Three 6 Mafia'},
    'VM0920': {'cat':'D','tag':'감성 보컬','w':[70, 20, 10],'prompt':'husky intelligent female R&B vocal with classic piano accompaniment, pansori-master young female trot, warm classic tenor'},
    'VM0921': {'cat':'D','tag':'거친 소울','w':[60, 20, 20],'prompt':'the king of Korean pop, versatile male vocal covering rock ballad and folk, warm robust tenor, rough raspy alto'},
    'VM0922': {'cat':'D','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, versatile male vocal, monster-vocal soprano with'},
    'VM0923': {'cat':'D','tag':'감성 폭발','w':[40, 40, 20],'prompt':'soulful male vocal with deep pain and emotion,, feathery soft tender male vocal with, global Billboard-hitting female'},
    'VM0924': {'cat':'D','tag':'감성 보컬','w':[60, 30, 10],'prompt':'textbook traditional female trot vocalist with the most flavorful classic bending delivery, easygoing sunny tenor with playful organic, revolutionary UK grime male'},
    'VM0925': {'cat':'D','tag':'그루브 보컬','w':[60, 30, 10],'prompt':'Dutch boom-bap female rapper who captivated all of Europe with classic flow, airy ethereal male falsetto vocal building, sweeping dramatic tenor'},
    'VM0926': {'cat':'D','tag':'폭발 에너지','w':[50, 30, 20],'prompt':'soaring rock soprano with legendary high-range stadium power, husky powerful male belting vocal with, world-class speed-rap female'},
    'VM0927': {'cat':'D','tag':'허스키 매력','w':[50, 40, 10],'prompt':'elegant 60s female trot vocalist layering sophisticated arrangements over traditional melody, breezy light acoustic vocal with sunny, sharp sorrowful tenor'},
    'VM0928': {'cat':'D','tag':'묵직한 저음','w':[40, 40, 20],'prompt':'polished swinging male crooner vocal with, eccentric brilliant male rapper commanding neo-soul, mellow melodic male rapper'},
    'VM0929': {'cat':'D','tag':'허스키 감성','w':[70, 20, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, deep heavy charismatic, folk-rooted gentle male trot'},
    'VM0930': {'cat':'D','tag':'굵은 바리톤','w':[50, 30, 20],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, sophisticated mid-century male trot vocalist bridging modern, heavyweight commanding male'},
    'VM0931': {'cat':'D','tag':'묵직한 저음','w':[60, 30, 10],'prompt':'smooth classic baritone male crooner jazz pop vocal, gravelly uniquely husky deep male jazz vocal, one, Korean R&B fairy female'},
    'VM0932': {'cat':'D','tag':'댄스 보컬','w':[50, 30, 20],'prompt':'legendary high-tone female rapper with rhythmic agility and iconic vocal presence, UK house-DnB hit vocal, chart-dominating, France greatest-selling female'},
    'VM0933': {'cat':'D','tag':'폭발 에너지','w':[60, 20, 20],'prompt':'flawless classic female vocal mastering Broadway and pop with zero error, stable soaring high-note, sad sharp Irish'},
    'VM0934': {'cat':'D','tag':'허스키 매력','w':[60, 20, 20],'prompt':'saddest tone in jazz history, wounded soul female vocal with laid-back phrasing, feathery soft tender, lyrical light tenor'},
    'VM0935': {'cat':'D','tag':'크리스탈 톤','w':[40, 40, 20],'prompt':'bright cheerful male pop vocal with catchy, dreamy Latin-pop female rapper weaving ethereal harmonics, powerful husky alto'},
    'VM0936': {'cat':'D','tag':'투명한 음색','w':[50, 30, 20],'prompt':'pure clean soprano capturing quiet depth with timeless graceful phrasing, sorrowful French chanson female vocal pouring raw life, 5-octave female vocal'},
    'VM0937': {'cat':'D','tag':'허스키 감성','w':[40, 40, 20],'prompt':'legendary nasal-melody female trot vocalist who comforted a, deep heavy charismatic low male vocal, literary poetic mezzo-soprano'},
    'VM0938': {'cat':'D','tag':'댄스 보컬','w':[60, 20, 20],'prompt':'legendary storytelling male rapper with unique accent and theatrical narrative flow, authoritative deep male, electronic-trap crossover male'},
    'VM0939': {'cat':'D','tag':'허스키 매력','w':[60, 30, 10],'prompt':'husky intelligent female R&B vocal with classic piano accompaniment, smooth silky male R&B piano vocal, overwhelming falsetto high'},
    'VM0940': {'cat':'D','tag':'일렉트로닉','w':[50, 30, 20],'prompt':'legendary Three 6 Mafia female rapper ruling southern underground with fierce authority, laid-back mellow baritone with serene breezy, Eurovision cinematic electro'},
    'VM0941': {'cat':'D','tag':'일렉트로닉','w':[50, 40, 10],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, eccentric brilliant male rapper commanding neo-soul, explosive hardcore female rapper'},
    'VM0942': {'cat':'D','tag':'중저음 매력','w':[40, 40, 20],'prompt':'velvety deep crooning baritone with effortless, deep emotional dramatic Korean male trot, inventive genius male'},
    'VM0943': {'cat':'D','tag':'그루브 보컬','w':[70, 20, 10],'prompt':'solid high male vocal with retro and modern groove, rough textured male, versatile raw male rapper'},
    'VM0944': {'cat':'D','tag':'맑은 천상','w':[60, 20, 20],'prompt':'pure clean soprano capturing quiet depth with timeless graceful phrasing, rhythmic powerhouse, thunderous military-grade male'},
    'VM0945': {'cat':'D','tag':'폭발 에너지','w':[60, 20, 20],'prompt':'legendary Three 6 Mafia female rapper ruling southern underground with fierce authority, powerful open-throated male, hard-hitting slide-drill male'},
    'VM0946': {'cat':'D','tag':'허스키 매력','w':[50, 30, 20],'prompt':'legendary harmony female trot vocalist showcasing textbook traditional duet vocal mastery, quirky playful soprano with witty, deep emotional dramatic'},
    'VM0947': {'cat':'D','tag':'폭발 에너지','w':[60, 30, 10],'prompt':'the king of Korean pop, versatile male vocal covering rock ballad and folk, mysterious witch-like rough vibrato female rock, clear earnest tenor'},
    'VM0948': {'cat':'D','tag':'일렉트로닉','w':[50, 40, 10],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, Eurovision cinematic electro queen, grand synthpop, new-wave rock female rapper'},
    'VM0949': {'cat':'D','tag':'파워 보컬','w':[50, 40, 10],'prompt':'most devastating powerful 4-octave male rock vocal in history, shattering intensity, new-wave rock female rapper who shattered boundaries between, elegant silken high'},
    'VM0950': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 30, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, energetic flashy male rapper with wild hybrid flow, refined female trot vocalist'},
    'VM0951': {'cat':'D','tag':'감성 보컬','w':[70, 20, 10],'prompt':'legendary nasal-melody female trot vocalist who comforted a colonized nation with sorrow, agile scatting tenor, new-wave rock female rapper'},
    'VM0952': {'cat':'D','tag':'펑키 그루브','w':[60, 30, 10],'prompt':'husky intelligent female R&B vocal with classic piano accompaniment, Australian-born female rapper who topped Billboard hip-hop, intense 90s New'},
    'VM0953': {'cat':'D','tag':'감성 폭발','w':[60, 30, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery that makes the world cry, raw desperate soprano with unfiltered emotional intensity, unique bright indie'},
    'VM0954': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'queen of soul, gospel-based explosive powerful female vocal full of holy spirit, pharmacist-turned female trot, textbook female trot vocalist'},
    'VM0955': {'cat':'D','tag':'맑은 감성','w':[50, 40, 10],'prompt':'crystalline nightingale female trot vocalist who dominated early classic Korean trot, honest deep tenor with raw sincerity and, funky freewheeling male'},
    'VM0956': {'cat':'D','tag':'투명한 음색','w':[60, 20, 20],'prompt':'crystalline pure-toned male trot tenor revered as the emperor of classic Korean enka, smooth silky male, battle-rap legend female'},
    'VM0957': {'cat':'D','tag':'감성 보컬','w':[40, 40, 20],'prompt':'saddest tone in jazz history, wounded soul female, global trendy tenor with cinematic pop polish, revolutionary female rapper'},
    'VM0958': {'cat':'D','tag':'거친 소울','w':[50, 40, 10],'prompt':'soulful male vocal with deep pain and emotion, the godfather of soul, electronic-trap crossover male rapper blending synths with, luxurious deep soulful'},
    'VM0959': {'cat':'D','tag':'묵직한 저음','w':[60, 20, 20],'prompt':'rustic mid-low bending-note female trot vocalist anchoring legendary harmony foundations, most sophisticated calm, raw rough soul-shaking'},
    'VM0960': {'cat':'D','tag':'감성 폭발','w':[70, 20, 10],'prompt':'deep classic husky female vocal with overwhelming emotional delivery, pristine clean high-note, punk-rage female rapper pioneering'},
    'VM0961': {'cat':'D','tag':'허스키 매력','w':[70, 20, 10],'prompt':'legendary nasal-melody female trot vocalist who comforted a colonized nation with sorrow, intense dramatic, sad sharp Irish'},
    'VM0962': {'cat':'D','tag':'깊은 베이스','w':[40, 40, 20],'prompt':'refined French chanteuse with classic, romantic aged baritone with weathered folk, heavyweight commanding male'},
    'VM0963': {'cat':'D','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'solid high male vocal with retro and modern groove, energetic performer, minimalist acoustic male trot vocalist with deep resonance, sophisticated folk soprano'},
    'VM0964': {'cat':'D','tag':'리듬 보컬','w':[50, 30, 20],'prompt':'charming deep baritone male vocal, the king of rock and roll, wordplay-brilliant UK duo male rappers with, soulful jazzy male rapper'},
    'VM0965': {'cat':'D','tag':'펑키 그루브','w':[50, 40, 10],'prompt':'flawless classic female vocal mastering Broadway and pop with zero error, explosive raspy female rapper with southern trap energy, legendary Three 6 Mafia'},
    'VM0966': {'cat':'D','tag':'강렬한 샤우팅','w':[70, 20, 10],'prompt':'elegant refined tenor with classically elevated harmonic vocal phrasing, warm intimate male, Latin reggaeton-drill crossover'},
    'VM0967': {'cat':'D','tag':'감성 폭발','w':[60, 20, 20],'prompt':'refined French chanteuse with classic cinematic vocal elegance, punchy dynamic male, world-class speed-rap female'},
    'VM0968': {'cat':'D','tag':'깊은 울림','w':[70, 20, 10],'prompt':'charming deep baritone male vocal, the king of rock and roll, dreamy sophisticated falsetto, perfect powerful female'},
    'VM0969': {'cat':'D','tag':'거친 소울','w':[50, 40, 10],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, rough torn raspy male vocal pouring soul, pansori-infused cinematic male trot'},
    'VM0970': {'cat':'D','tag':'깊은 울림','w':[50, 40, 10],'prompt':'smooth classic baritone male crooner jazz pop vocal, soft breathy warm female pop vocal, solid expressive female trot'},
    'VM0971': {'cat':'D','tag':'댄스 보컬','w':[50, 40, 10],'prompt':'retro 8-bit electro-trap pioneer, funky chiptune rebel, original genre bender, fleet-footed male rapper with dazzling speed and showmanship, addictive melodic Korean'},
    'VM0972': {'cat':'D','tag':'리듬 보컬','w':[40, 40, 20],'prompt':'timeless elegant male jazz crooner vocal, dreamy Latin-pop female rapper weaving ethereal harmonics, legendary storytelling male'},
    'VM0973': {'cat':'D','tag':'그루브 보컬','w':[70, 20, 10],'prompt':'legendary 80s female rapper who spearheaded hip-hop mainstream with infectious energy, most sophisticated calm, sophisticated folk soprano'},
    'VM0974': {'cat':'D','tag':'크리스탈 톤','w':[50, 40, 10],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, dreamy atmospheric house, lush vocal pads,, emotional healing trance,'},
    'VM0975': {'cat':'D','tag':'일렉트로닉','w':[60, 30, 10],'prompt':'powerful UK national male rapper fusing grime with classic soul harmonics brilliantly, rapid-fire technical male rapper balancing speed, Bronx drill female'},
    'VM0976': {'cat':'D','tag':'폭발 에너지','w':[40, 40, 20],'prompt':'rebellious melancholic raw retro soul jazz female vocal,, smooth urban tenor with silky tone and, high-pitched screaming male'},
    'VM0977': {'cat':'D','tag':'맑은 감성','w':[50, 40, 10],'prompt':'pristine classical crossover soprano with angelic symphonic serenity, heavyweight commanding male rapper with flawless flow, polished hybrid male trot'},
    'VM0978': {'cat':'D','tag':'시원한 고음','w':[50, 30, 20],'prompt':'young luminous classical crossover soprano with operatic innocence and grace, husky powerful male belting vocal with, overwhelming falsetto high'},
    'VM0979': {'cat':'D','tag':'중저음 매력','w':[60, 30, 10],'prompt':'wise philosophical male rapper layering classic boom-bap soul with modern narratives, resonant deep baritone with dramatic anthemic, mellow melodic male rapper'},
    'VM0980': {'cat':'D','tag':'압도적 고음','w':[60, 30, 10],'prompt':'majestic operatic tenor with soaring classical purity and grand resonance, cool understated spoken-word folk vocal with, bold theatrical baritone'},
    'VM0981': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'legendary Three 6 Mafia female rapper ruling southern underground with fierce authority, fierce sharp-tongued New, polished warm soprano'},
    'VM0982': {'cat':'D','tag':'그루브 보컬','w':[60, 30, 10],'prompt':'legendary 80s female rapper who spearheaded hip-hop mainstream with infectious energy, rich commanding contralto with majestic, funky freewheeling male'},
    'VM0983': {'cat':'D','tag':'감성 폭발','w':[60, 20, 20],'prompt':'legendary nasal sorrowful uniquely toned Korean trot folk female vocal soaking the soul, viral hook-machine female, pinnacle of Korean'},
    'VM0984': {'cat':'D','tag':'그루브 보컬','w':[70, 20, 10],'prompt':'legendary high-tone female rapper with rhythmic agility and iconic vocal presence, global EDM hit, lush romantic male'},
    'VM0985': {'cat':'D','tag':'일렉트로닉','w':[50, 30, 20],'prompt':'pure clean soprano capturing quiet depth with timeless graceful phrasing, perfect pitch hybrid vocal, Zedd Stay, 2NE1 hardcore K-pop female'},
    'VM0986': {'cat':'D','tag':'펑키 그루브','w':[60, 20, 20],'prompt':'wise philosophical male rapper layering classic boom-bap soul with modern narratives, tireless iron-throated high, Elvis-inspired charismatic male'},
    'VM0987': {'cat':'D','tag':'그루브 보컬','w':[50, 40, 10],'prompt':'solid high male vocal with retro and modern groove, energetic performer, prodigious genius female cinematic trot vocalist, battle-rap legend female'},
    'VM0988': {'cat':'D','tag':'천상의 목소리','w':[60, 20, 20],'prompt':'powerful UK national male rapper fusing grime with classic soul harmonics brilliantly, gentle refined male, tender longing falsetto'},
    'VM0989': {'cat':'D','tag':'거친 소울','w':[60, 30, 10],'prompt':'brilliant nightingale female trot vocalist celebrated as the golden voice of the 50s-60s, explosive powerhouse belter with massive, pristine smooth male'},
    'VM0990': {'cat':'D','tag':'일렉트로닉','w':[70, 20, 10],'prompt':'warm classic tenor with smooth legato phrasing and nostalgic vibrato tone, nasal high-pitched male, 2NE1 hardcore K-pop female'},
    'VM0991': {'cat':'D','tag':'맑은 감성','w':[70, 20, 10],'prompt':'comforting warm baritone with steady timeless phrasing and soothing resonance, global trendy tenor, rich deep mid-low'},
    'VM0992': {'cat':'D','tag':'일렉트로닉','w':[50, 30, 20],'prompt':'polished warm soprano with elegant 60s sophisticated pop soul tone, pioneering male rapper who defined modern rhyme, Bronx drill female'},
    'VM0993': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'classical soprano female vocal pioneering symphonic metal genre, fierce Miami trap duo, young prodigy female trot'},
    'VM0994': {'cat':'D','tag':'감성 보컬','w':[60, 20, 20],'prompt':'legendary harmony female trot vocalist showcasing textbook traditional duet vocal mastery, percussion-performing male trot, pioneering fierce female'},
    'VM0995': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'classical crossover male trot vocalist harmonizing operatic power with grand orchestral scale, aggressive hard-hitting male, pristine clean high-note'},
    'VM0996': {'cat':'D','tag':'묵직한 저음','w':[50, 40, 10],'prompt':'timeless elegant male jazz crooner vocal with effortless classic phrasing, rich velvety contralto with breathtaking sustained, crystalline soaring'},
    'VM0997': {'cat':'D','tag':'강렬한 샤우팅','w':[60, 20, 20],'prompt':'queen of soul, gospel-based explosive powerful female vocal full of holy spirit, massive cinematic soprano, trendy stylish male'},
    'VM0998': {'cat':'D','tag':'폭발 에너지','w':[50, 40, 10],'prompt':'classical soprano female vocal pioneering symphonic metal genre, massive operatic soprano with stadium-shaking, lethal off-beat female'},
    'VM0999': {'cat':'D','tag':'감성 폭발','w':[40, 40, 20],'prompt':'the king of Korean pop, versatile male vocal, hard-hitting gangster male rapper delivering textbook, precise pitch-perfect male pop'},
    'VM1000': {'cat':'D','tag':'강렬한 샤우팅','w':[50, 40, 10],'prompt':'classical crossover male trot vocalist harmonizing operatic power with grand orchestral scale, warm low-register male trot vocalist evoking hometown, pop-ballad crossover female'},
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

    # ── cgo-390: VOICE_MIX 프리셋 사용 경로 ──
    if req.voice_mix_id and req.voice_mix_id in VOICE_MIX:
        mix = VOICE_MIX[req.voice_mix_id]
        converted_prompt = mix["prompt"]
        vocal_tag = ""
        if req.vocal == "male":
            vocal_tag = "male vocals only, all male singers, "
        elif req.vocal == "female":
            vocal_tag = "female vocals only, all female singers, "
        elif req.vocal == "duet":
            vocal_tag = "male and female duet, "
        suno_prompt = f"{vocal_tag}{converted_prompt}, {req.style}, {req.bpm} BPM, key of {req.key}"
        if req.chords:
            suno_prompt += f", chord progression: {req.chords}"
    else:
        # ── cgo-382: 최소 1명 이상 보컬 검증 (독창 지원 — 기존 2명→1명) ──
        matched_artists = []
        seen_descriptions: set = set()
        for artist, description in VOICE_MAP.items():
            if artist in prompt and description not in seen_descriptions:
                matched_artists.append(artist)
                seen_descriptions.add(description)

        if len(seen_descriptions) < 1:
            return JSONResponse(status_code=400, content={
                "ok": False,
                "error": "보컬리스트를 1명 이상 선택해 주세요.",
                "hint": "보컬 선택에서 장르·성별을 고른 뒤 추첨하거나 카드를 직접 선택하세요",
                "matched": len(seen_descriptions)
            })

        # ── 가수 이름 → 보컬 설명 변환 (Suno 정책 준수) ──
        converted_prompt = prompt
        for artist, description in VOICE_MAP.items():
            if artist in converted_prompt:
                converted_prompt = converted_prompt.replace(artist, description)

        # ── Suno 프롬프트 조합 (cgo-382: 성별 + 독창/믹스 태그) ──
        vocal_tag = ""
        if req.vocal == "male":
            vocal_tag = "male vocals only, all male singers, "
        elif req.vocal == "female":
            vocal_tag = "female vocals only, all female singers, "
        elif req.vocal == "duet":
            vocal_tag = "male and female duet, "
        if len(seen_descriptions) == 1:
            vocal_tag += "solo vocal, single singer, "
        suno_prompt = f"{vocal_tag}{converted_prompt}, {req.style}, {req.bpm} BPM, key of {req.key}"
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
        minimum_frequency=librosa.note_to_hz('C4'),   # cgo-375: C3→C4 보컬 하한 (베이스/기타 악기음 제거)
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

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
