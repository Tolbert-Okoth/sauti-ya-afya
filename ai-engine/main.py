from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import os
import uvicorn
import shutil # 👈 ADDED for diagnostics
from analyzer import analyze_audio

# --- OPTIONAL: LLM BRIDGE ---
try:
    from llm_bridge import get_medical_explanation
    HAS_LLM = True
except ImportError:
    HAS_LLM = False
    print("⚠️ Warning: llm_bridge.py not found. AI Explanations disabled.")

app = FastAPI(title="SautiYaAfya AI Bridge [SECURE]")

# --- 🛡️ SECURITY LAYER 1: PRODUCTION CORS ---
origins = [
    "http://localhost:3000",                  # Local React
    "http://localhost:5000",                  # Local Node
    "https://sauti-ya-afya.vercel.app",       # ✅ FIX: Added hyphens to match your Vercel URL
    "https://sauti-backend.onrender.com"      # Production Backend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # ✅ FIX: Use specific list, NOT ["*"]
    allow_credentials=True,      # This requires specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- 🛡️ SECURITY LAYER 2: CONSTANTS ---
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB Limit
ACCEPTED_MIME_TYPES = ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/webm"]

# --- 🛡️ SECURITY LAYER 3: MAGIC BYTES ---
def validate_magic_bytes(file_header: bytes) -> bool:
    """Check file signature to prevent renamed .exe files."""
    if file_header.startswith(b'RIFF') and file_header[8:12] == b'WAVE': return True
    if file_header.startswith(b'ID3') or file_header.startswith(b'\xff\xfb') or file_header.startswith(b'\xff\xf3'): return True
    if file_header.startswith(b'\x1a\x45\xdf\xa3'): return True # WebM
    return False

# ✅ HEALTH CHECK ROUTE (For UptimeRobot)
@app.get("/")
def read_root():
    return {"status": "SautiYaAfya AI Engine Online & Secure", "llm_active": HAS_LLM}

@app.post("/analyze")
async def analyze_endpoint(
    file: UploadFile = File(...), 
    age: str = Form("Unknown"),
    symptoms: str = Form("None reported"),
    threshold: float = Form(0.75) 
):
    temp_path = f"{UPLOAD_DIR}/{file.filename}"
    
    try:
        # 1. MIME Check
        if file.content_type not in ACCEPTED_MIME_TYPES:
             print(f"[Security Warning] Uncommon MIME type: {file.content_type}")

        # 2. Magic Bytes & Size Check
        header = await file.read(12)
        if not validate_magic_bytes(header):
            raise HTTPException(status_code=400, detail="Security Alert: Invalid file signature.")
        
        await file.seek(0)
        
        # 📂 WRITE FILE & DIAGNOSTIC CHECK
        with open(temp_path, "wb") as buffer:
            size = 0
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk: break
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="File too large (Max 10MB).")
                buffer.write(chunk)
        
        # 🔍 ROOT CAUSE DIAGNOSTIC 1: FILE SIZE
        print(f"🔍 DIAGNOSTIC: Saved {file.filename} - Size: {size} bytes")
        if size < 100:
             print("❌ CRITICAL: File is too small (<100 bytes). Likely empty.")
             raise HTTPException(status_code=400, detail="Uploaded file is empty or corrupted.")

        # 🔍 ROOT CAUSE DIAGNOSTIC 2: FFmpeg CHECK
        ffmpeg_path = shutil.which("ffmpeg")
        print(f"🔍 DIAGNOSTIC: FFmpeg found at: {ffmpeg_path}")
        if not ffmpeg_path:
             print("❌ CRITICAL: FFmpeg NOT FOUND. Audio load will hang.")
            
        # 3. Audio Analysis (Threaded)
        print("🔍 DIAGNOSTIC: Handing off to Analyzer...")
        
        signal_result = await run_in_threadpool(
            analyze_audio, 
            temp_path, 
            sensitivity_threshold=threshold
        )
        
        # Cleanup immediately
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if signal_result.get("status") == "error":
            raise HTTPException(status_code=500, detail=signal_result["message"])
            
        # 4. LLM Explanation (Threaded & Conditional)
        ai_explanation = "LLM Module Not Active"
        if HAS_LLM:
            try:
                ai_explanation = await run_in_threadpool(
                    get_medical_explanation,
                    biomarkers=signal_result["biomarkers"], 
                    age=age, 
                    symptoms=symptoms,
                    threshold=threshold
                )
            except Exception as e:
                print(f"LLM Error: {e}")
                ai_explanation = "Analysis complete, but AI explanation unavailable."
        
        return {
            **signal_result,             
            "ai_explanation": ai_explanation 
        }

    except HTTPException as he:
        if os.path.exists(temp_path): os.remove(temp_path)
        raise he
    except Exception as e:
        if os.path.exists(temp_path): os.remove(temp_path)
        print(f"Server Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Render provides PORT env var, otherwise default to 10000 (or 8000 local)
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)