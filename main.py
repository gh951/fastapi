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


class VvipReq(BaseModel):
    prompt: str = ""
    lyrics: str = ""
    style: str = "pop"
    bpm: int = 120
    key: str = "C"
    chords: str = ""
    vocal: str = ""  # cgo-382: male/female/duet/bgm/child/choir


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
