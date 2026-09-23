"""
CGO 음악 렌더 서버 (Railway) v2 — 멜로디 중심 믹스·음량 보강 (cgo-297용), 이 파일 하나로 서버 전체가 동작합니다.
  GET  /             : 서버 깨우기·상태 확인
  POST /render       : (기존) 멜로디 한 줄 렌더  {bpm, notes:[{n,d}], instrument}
  POST /render_full  : (신규) 멜로디·베이스·코드·드럼 한 번에 렌더 → WAV 1개
  GET  /render_full  : 배포 확인용
"""
import glob
import io
import os
import shutil
import subprocess
import tempfile
import threading
import wave

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from typing import List, Optional

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
    return {"ok": True, "service": "cgo-render", "sf2": _find_sf2()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
