# ai-engine/gunicorn_config.py
# Start with: gunicorn -c gunicorn_config.py main:app
import os

worker_class = "uvicorn.workers.UvicornWorker"  # FastAPI is ASGI; the default sync worker cannot serve it
workers = 1             # Must stay 1: the job store lives in memory (and free tier RAM is limited)
timeout = 120           # Give the AI 2 minutes to think before timing out
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"  # Render sets PORT
