from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
import shutil
import uuid
import gc
from analyzer import analyze_audio

app = FastAPI(title="SautiYaAfya Async Engine")

# 🛡️ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🧠 IN-MEMORY JOB STORE (The "Ticket" Ledger)
JOBS = {}

def background_worker(job_id: str, file_path: str, threshold: float, symptoms: str):
    """
    The heavy lifter that runs in the background.
    Now supports HYBRID DIAGNOSIS by passing symptoms to the analyzer.
    """
    print(f"👷 WORKER: Starting Job {job_id} (Symptoms: '{symptoms}')")
    try:
        # Run the heavy analysis with symptoms context
        result = analyze_audio(file_path, symptoms=symptoms, sensitivity_threshold=threshold)
        
        JOBS[job_id] = {
            "status": "completed",
            "result": result
        }
        print(f"✅ WORKER: Job {job_id} Finished Success")
        
    except Exception as e:
        print(f"❌ WORKER: Job {job_id} Failed: {str(e)}")
        JOBS[job_id] = {
            "status": "failed",
            "error": str(e)
        }
    finally:
        # Cleanup file immediately to save space
        if os.path.exists(file_path):
            os.remove(file_path)
        gc.collect()

@app.get("/")
def read_root():
    return {"status": "AI Engine Online", "mode": "Async Ticket System + Hybrid Diagnosis"}

@app.post("/analyze")
async def submit_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...), 
    threshold: float = Form(0.75),
    symptoms: str = Form("")  # 🏥 NEW: Receive symptoms from Frontend
):
    """
    1. Receives File & Symptoms
    2. Returns 'Ticket' (Job ID) immediately (200 OK)
    3. Starts processing in background
    """
    try:
        # Generate Ticket ID
        job_id = str(uuid.uuid4())
        temp_path = f"{UPLOAD_DIR}/{job_id}_{file.filename}"
        
        # Save file to disk
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Initialize Job Status
        JOBS[job_id] = {"status": "processing"}
        
        # Hand off to background worker (Pass symptoms!)
        background_tasks.add_task(background_worker, job_id, temp_path, threshold, symptoms)
        
        print(f"🎫 TICKET ISSUED: {job_id}")
        return {"job_id": job_id, "status": "queued"}

    except Exception as e:
        print(f"❌ SUBMIT ERROR: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status/{job_id}")
def check_status(job_id: str):
    """
    Frontend calls this repeatedly to check if the job is done.
    """
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    # 🛑 Keep workers=1 to prevent RAM crashes on Free Tier
    uvicorn.run(app, host="0.0.0.0", port=port)