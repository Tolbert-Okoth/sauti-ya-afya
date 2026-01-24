from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import os
import uvicorn
import shutil
import gc 
from analyzer import analyze_audio

app = FastAPI(title="SautiYaAfya AI Bridge [SAFETY MODE]")

# 🛡️ CORS: ALLOW ALL (Temporary Debugging)
# This prevents "False Positive" CORS errors when the server errors out.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 ALLOW EVERYONE
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"status": "AI Engine Online", "mode": "Safety Mode"}

@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...), 
    threshold: float = Form(0.75) 
):
    temp_path = f"{UPLOAD_DIR}/{file.filename}"
    
    try:
        # 🧹 Pre-emptive Cleanup
        gc.collect()
        
        # 1. SAVE FILE
        with open(temp_path, "wb") as buffer:
            size = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk: break
                size += len(chunk)
                buffer.write(chunk)
        
        print(f"🔍 DIAGNOSTIC: File saved ({size} bytes). Starting Analysis...")

        # 2. RUN ANALYZER
        # run_in_threadpool prevents blocking the main server heartbeat
        signal_result = await run_in_threadpool(
            analyze_audio, 
            temp_path, 
            sensitivity_threshold=threshold
        )
        
        # 3. CLEANUP
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        if signal_result.get("status") == "error":
            print(f"❌ ANALYZER ERROR: {signal_result.get('message')}")
            raise HTTPException(status_code=500, detail=signal_result["message"])
            
        print("✅ SUCCESS: Sending response.")
        return signal_result

    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        print(f"❌ SERVER CRASH: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)