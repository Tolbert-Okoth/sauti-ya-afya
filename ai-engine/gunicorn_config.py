# ai-engine/gunicorn_config.py
workers = 1             # Free tier has limited RAM, so we stick to 1 worker
threads = 2             # Allow concurrent requests
timeout = 120           # Give the AI 2 minutes to think before timing out
bind = "0.0.0.0:10000"  # Render expects port 10000 by default