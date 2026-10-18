from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Header
from fastapi.middleware.cors import CORSMiddleware
import gc
import logging
import os
import secrets
import threading
import time
import uuid

import uvicorn

from typing import List, Optional

from analyzer import analyze_audio, analyze_child, RecordingUnusable, MODEL_PATH, MODEL_METRICS, MAX_SPOTS
from audio_pipeline import AudioDecodeError, PIPELINE_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engine")

# ⚙️ CONFIG (environment variables)
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
API_KEY = os.environ.get("ENGINE_API_KEY", "")  # when set, requests must send X-API-Key
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024)
JOB_TTL_SECONDS = int(os.environ.get("JOB_TTL_SECONDS", "3600"))
MAX_JOBS = int(os.environ.get("MAX_JOBS", "500"))
ALLOWED_EXTENSIONS = {".webm", ".wav", ".mp3", ".m4a", ".ogg", ".oga", ".flac", ".aac", ".mp4", ".3gp"}

app = FastAPI(title="SautiYaAfya Async Engine")

# 🛡️ CORS — set ALLOWED_ORIGINS to the frontend URL(s) in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🧠 IN-MEMORY JOB STORE (single worker only — see gunicorn_config.py)
JOBS = {}
JOBS_LOCK = threading.Lock()
# One analysis at a time keeps peak RAM within the free tier
ANALYSIS_LOCK = threading.Lock()


def _purge_jobs():
    now = time.time()
    with JOBS_LOCK:
        for job_id in [j for j, v in JOBS.items() if now - v["created"] > JOB_TTL_SECONDS]:
            del JOBS[job_id]


def _set_job(job_id, **fields):
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def _check_api_key(x_api_key):
    if API_KEY and not secrets.compare_digest(x_api_key or "", API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _run_job(job_id: str, file_paths: list, analyse):
    """The heavy lifter that runs in the background. `analyse` gets no arguments and returns the result."""
    log.info("Job %s started (%d file(s))", job_id, len(file_paths))  # symptoms are patient data: never logged
    try:
        with ANALYSIS_LOCK:
            result = analyse()
        _set_job(job_id, status="completed", result=result)
        log.info("Job %s completed: %s", job_id, result["biomarkers"]["ai_diagnosis"])
    except (RecordingUnusable, AudioDecodeError) as e:
        _set_job(job_id, status="failed", error=f"Recording unusable: {e}", retake=True)
        log.info("Job %s rejected: %s", job_id, e)
    except Exception:
        _set_job(job_id, status="failed", error="Analysis failed. Please try again.")
        log.exception("Job %s crashed", job_id)
    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)
        gc.collect()


def background_worker(job_id: str, file_path: str, threshold: float, symptoms: str):
    _run_job(job_id, [file_path],
             lambda: analyze_audio(file_path, symptoms=symptoms, sensitivity_threshold=threshold))


async def _save_upload(file: UploadFile, stem: str) -> str:
    """Stream an upload to disk under a server-chosen name, enforcing type and size limits."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{ext}'")
    temp_path = os.path.join(UPLOAD_DIR, stem + ext)  # never use the client's filename in the path
    size = 0
    try:
        with open(temp_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail=f"File larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB")
                buffer.write(chunk)
    except BaseException:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise
    if size == 0:
        os.remove(temp_path)
        raise HTTPException(status_code=400, detail="Empty file")
    return temp_path


def _new_job_id():
    _purge_jobs()
    with JOBS_LOCK:
        if len(JOBS) >= MAX_JOBS:
            raise HTTPException(status_code=503, detail="Engine busy, try again shortly")
    return str(uuid.uuid4())


def _queue(job_id, background_tasks, fn, *args):
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "processing", "created": time.time()}
    background_tasks.add_task(fn, *args)
    log.info("Ticket issued: %s", job_id)
    return {"job_id": job_id, "status": "queued"}


def _child_metrics():
    test = MODEL_METRICS.get("test", {})
    return test.get("child") or test.get("all") or test


@app.get("/")
def read_root():
    return {
        "status": "AI Engine Online",
        "mode": "Async Ticket System",
        "model": os.path.basename(MODEL_PATH),
        "pipeline_version": PIPELINE_VERSION,
        "test_sensitivity_children": _child_metrics().get("recall_sensitivity"),
    }


@app.post("/analyze")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = Form(0.75),
    symptoms: str = Form("", max_length=2000),
    x_api_key: str = Header(default=""),
):
    """
    1. Receives File & Symptoms
    2. Returns 'Ticket' (Job ID) immediately
    3. Starts processing in background
    """
    _check_api_key(x_api_key)
    job_id = _new_job_id()
    temp_path = await _save_upload(file, job_id)
    return _queue(job_id, background_tasks, background_worker, job_id, temp_path, threshold, symptoms)


@app.post("/analyze_child")
async def submit_child_job(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    symptoms: str = Form("", max_length=2000),
    age_months: Optional[float] = Form(None, ge=0, le=216),
    breaths_per_minute: Optional[float] = Form(None, ge=5, le=150),
    x_api_key: str = Header(default=""),
):
    """One child: 1-4 chest-spot recordings + breathing rate + symptoms -> one IMCI-style result."""
    _check_api_key(x_api_key)
    if not 1 <= len(files) <= MAX_SPOTS:
        raise HTTPException(status_code=400, detail=f"Send 1 to {MAX_SPOTS} recordings")
    job_id = _new_job_id()
    paths = []
    try:
        for i, f in enumerate(files):
            paths.append(await _save_upload(f, f"{job_id}_{i}"))
    except BaseException:
        for path in paths:
            os.remove(path)
        raise
    return _queue(job_id, background_tasks, _run_job, job_id, paths,
                  lambda: analyze_child(paths, symptoms=symptoms, age_months=age_months,
                                        breaths_per_minute=breaths_per_minute))


@app.get("/status/{job_id}")
def check_status(job_id: str, x_api_key: str = Header(default="")):
    """Frontend calls this repeatedly to check if the job is done."""
    _check_api_key(x_api_key)
    _purge_jobs()
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found")
        return {k: v for k, v in job.items() if k != "created"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # 🛑 Keep a single worker: the job store is in memory
    uvicorn.run(app, host="0.0.0.0", port=port)
