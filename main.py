# CGO Music Server v1.2
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import os, subprocess, tempfile, glob

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

def find_sf2():
    for pattern in ["/nix/store/**/FluidR3_GM2-2.sf2", "/nix/store/**/FluidR3_GM.sf2", "/nix/store/**/*.sf2"]:
        files = glob.glob(pattern, recursive=True)
        if files:
            return files[0]
    return None

@app.get("/")
def root():
    return {"status": "CGO Music Server OK", "version": "1.2"}

@app.get("/health")
def health():
    try:
        r = subprocess.run(["fluidsynth", "--version"], capture_output=True, text=True)
        return {"fluidsynth": r.stdout.strip(), "ok": True, "sf2": find_sf2()}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.post("/render")
def render(data: dict):
    from midiutil import MIDIFile
    sf2 = find_sf2()
    if not sf2:
        return {"error": "Soundfont not found"}
    bpm = data.get("bpm", 120)
    raw_notes = data.get("notes", [60, 62, 64, 65, 67, 69, 71, 72])
    instrument = data.get("instrument", 0)
    notes = []
    for item in raw_notes:
        if isinstance(item, dict):
            notes.append((int(item.get("n", 60)), float(item.get("d", 1.0))))
        else:
            notes.append((int(item), 1.0))
    midi = MIDIFile(1)
    midi.addTempo(0, 0, bpm)
    midi.addProgramChange(0, 0, 0, instrument)
    beat = 0.0
    for note_num, dur in notes:
        note_num = max(0, min(127, note_num))
        dur = max(0.1, float(dur))
        midi.addNote(0, 0, note_num, beat, dur * 0.9, 100)
        beat += dur
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        midi.writeFile(f)
        mid_path = f.name
    wav_path = mid_path.replace(".mid", ".wav")
    try:
        subprocess.run(["fluidsynth", "-ni", sf2, mid_path, "-F", wav_path, "-r", "44100"], check=True, capture_output=True)
        with open(wav_path, "rb") as f:
            audio = f.read()
    finally:
        if os.path.exists(mid_path):
            os.unlink(mid_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)
    return Response(content=audio, media_type="audio/wav")
