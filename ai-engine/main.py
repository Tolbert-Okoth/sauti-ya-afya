from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool # Preserved: Async Threading
import os
from analyzer import analyze_audio
from llm_bridge import get_medical_explanation

app = FastAPI(title="SautiYaAfya AI Bridge [SECURE]")

# --- 🛡️ SECURITY LAYER 1: STRICT CORS ---
# Only allow known origins. "Wildcard *" is dangerous in production.
origins = [
    "http://localhost:3000", # React Client
    "http://localhost:5000", # Node.js Orchestrator
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Locked down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 🛡️ SECURITY LAYER 2: CONSTANTS ---
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB Limit (Prevents Memory Overflow)
ACCEPTED_MIME_TYPES = ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/webm"]

# --- 🛡️ SECURITY LAYER 3: MAGIC BYTES ---
def validate_magic_bytes(file_header: bytes) -> bool:
    """
    Reads the first few bytes to ensure the file is actually audio
    and not a disguised virus (e.g., .exe renamed to .wav).
    """
    # WAV (RIFF....WAVE)
    if file_header.startswith(b'RIFF') and file_header[8:12] == b'WAVE':
        return True
    # MP3 (ID3 or Sync Frame 0xFF 0xFB/0xF3)
    if file_header.startswith(b'ID3') or file_header.startswith(b'\xff\xfb') or file_header.startswith(b'\xff\xf3'):
        return True
    # WEBM (1A 45 DF A3) - often sent by browser recorders
    if file_header.startswith(b'\x1a\x45\xdf\xa3'):
        return True
    
    return False

@app.get("/")
def read_root():
    return {"status": "SautiYaAfya AI Engine Online & Secure"}

@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...), 
    age: str = Form("Unknown"),
    symptoms: str = Form("None reported"),
    threshold: float = Form(0.75) 
):
    """
    Receives audio + metadata + ADMIN THRESHOLD.
    Runs 'The Sniper' analysis in a background thread.
    Protected by Magic Byte Validation & Size Limits.
    """
    
    temp_path = f"{UPLOAD_DIR}/{file.filename}"
    
    try:
        # --- 🛡️ SECURITY CHECK: MIME TYPE ---
        # Note: Headers can be spoofed, so we rely more on Magic Bytes below.
        if file.content_type not in ACCEPTED_MIME_TYPES:
             # Log warning but don't crash, let Magic Bytes decide the truth.
             print(f"[Security Warning] Uncommon MIME type: {file.content_type}")

        # --- 🛡️ SECURITY CHECK: MAGIC BYTES & SIZE ---
        
        # 1. Read first 12 bytes for Magic Byte check
        header = await file.read(12)
        if not validate_magic_bytes(header):
            raise HTTPException(status_code=400, detail="Security Alert: File signature rejected. Not a valid audio file.")
        
        # 2. Reset cursor to start
        await file.seek(0)
        
        # 3. Secure Write with Size Limit
        with open(temp_path, "wb") as buffer:
            size = 0
            while True:
                chunk = await file.read(1024 * 1024) # Read in 1MB chunks
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
                buffer.write(chunk)
            
        # --- END SECURITY CHECKS ---

        # 4. Run Analysis (NON-BLOCKING)
        signal_result = await run_in_threadpool(
            analyze_audio, 
            temp_path, 
            sensitivity_threshold=threshold
        )
        
        # 5. Cleanup Audio immediately after analysis
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if signal_result.get("status") == "error":
            raise HTTPException(status_code=500, detail=signal_result["message"])
            
        # 6. Run LLM (NON-BLOCKING)
        ai_explanation = await run_in_threadpool(
            get_medical_explanation,
            biomarkers=signal_result["biomarkers"], 
            age=age, 
            symptoms=symptoms,
            threshold=threshold
        )
        
        # 7. Return Result
        return {
            **signal_result,             
            "ai_explanation": ai_explanation 
        }

    except HTTPException as he:
        # Re-raise HTTP exceptions (like 413 or 400) so frontend sees them
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise he

    except Exception as e:
        # Ensure cleanup happens even if something crashes
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)