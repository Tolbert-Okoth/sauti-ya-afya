from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import uvicorn
import shutil
import uuid
import time
import gc
import logging
from pathlib import Path
from analyzer import analyze_audio
from typing import Dict, Any

# ────────────────────────────────────────────────
# Logging setup
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("SautiYaAfya")

app = FastAPI(
    title="SautiYaAfya Lung Sound Analysis API",
    description="Async audio analysis engine for respiratory sound classification",
    version="2.1.0"
)

# ────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────
UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_FILE_SIZE_MB = int(os.getenv("MAX_UPLOAD_MB", 10))
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}

# In-memory job store with expiration (seconds)
JOB_TTL_SECONDS = 3600 * 2  # 2 hours
JOBS: Dict[str, Dict[str, Any]] = {}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # ← tighten in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def cleanup_old_jobs():
    """Remove expired jobs to prevent memory leak"""
    now = time.time()
    to_remove = []
    for job_id, job in JOBS.items():
        if job.get("created_at", 0) + JOB_TTL_SECONDS < now:
            to_remove.append(job_id)
    for jid in to_remove:
        del JOBS[jid]
        logger.info(f"Expired job removed: {jid}")


def background_worker(job_id: str, file_path: str, threshold: float, symptoms: str):
    logger.info(f"Worker started | Job: {job_id} | Symptoms: '{symptoms[:80]}...'")
    start_time = time.time()

    try:
        result = analyze_audio(
            file_path,
            symptoms=symptoms.strip(),
            sensitivity_threshold=threshold
        )

        JOBS[job_id].update({
            "status": "completed",
            "result": result,
            "finished_at": time.time(),
            "duration_seconds": round(time.time() - start_time, 2)
        })
        logger.info(f"Job completed successfully | {job_id} | {result.get('biomarkers', {}).get('ai_diagnosis')}")

    except Exception as e:
        logger.exception(f"Job failed | {job_id}")
        JOBS[job_id].update({
            "status": "failed",
            "error": str(e),
            "finished_at": time.time()
        })

    finally:
        # Always clean up file
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete temp file {file_path}: {e}")
        gc.collect()


@app.on_event("startup")
async def startup_event():
    logger.info("SautiYaAfya API starting...")


@app.get("/", summary="API Health Check")
async def root():
    cleanup_old_jobs()
    return {
        "status": "online",
        "version": app.version,
        "active_jobs": len(JOBS),
        "upload_dir_exists": UPLOAD_DIR.exists(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S EAT")
    }


@app.post("/analyze", status_code=status.HTTP_202_ACCEPTED)
async def submit_analysis_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    threshold: float = Form(0.75, ge=0.0, le=1.0),
    symptoms: str = Form("", max_length=300)
):
    """
    Submit audio file + optional symptoms → returns job ticket immediately.
    Processing happens asynchronously.
    """
    cleanup_old_jobs()

    # ── Basic validation ───────────────────────────────────────
    if not file.filename:
        raise HTTPException(400, "No file name provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    # ── Size check (approximate) ──────────────────────────────
    file.file.seek(0, 2)  # go to end
    file_size = file.file.tell()
    file.file.seek(0)     # reset
    if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File too large (> {MAX_FILE_SIZE_MB} MB)")

    # ── Create safe temp path ─────────────────────────────────
    job_id = str(uuid.uuid4())
    safe_filename = f"{job_id}{ext}"
    temp_path = UPLOAD_DIR / safe_filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        JOBS[job_id] = {
            "status": "queued",
            "created_at": time.time(),
            "filename": file.filename,
            "threshold": threshold,
            "symptoms_provided": bool(symptoms.strip())
        }

        background_tasks.add_task(
            background_worker,
            job_id,
            str(temp_path),
            threshold,
            symptoms
        )

        logger.info(f"Job queued | ID: {job_id} | File: {file.filename} | Size: {file_size//1024} KB")
        return {"job_id": job_id, "status": "queued"}

    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, "Failed to process upload")


@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Job not found or expired")

    response = {
        "job_id": job_id,
        "status": job["status"],
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job["created_at"])),
    }

    if job["status"] == "completed":
        response.update({
            "result": job["result"],
            "duration_seconds": job.get("duration_seconds"),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(job["finished_at"]))
        })
    elif job["status"] == "failed":
        response["error"] = job.get("error", "Unknown error")

    return response


if __name__ == "__main__":
    port = int(os.getenv("PORT", "10000"))
    workers = int(os.getenv("UVICORN_WORKERS", "1"))  # usually 1 on low-memory hosts

    logger.info(f"Starting server on port {port} with {workers} worker(s)")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info",
        reload=False  # disable in production
    )