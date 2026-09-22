from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "CGO Music Server OK", "version": "1.0"}

@app.get("/health")
def health():
    import subprocess
    try:
        r = subprocess.run(["fluidsynth","--version"],
                          capture_output=True, text=True)
        return {"fluidsynth": r.stdout.strip(), "ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}
