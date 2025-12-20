#!/bin/bash
# Start FastAPI backend
uvicorn back_end.api:app --reload &

# Start simple HTTP server for front-end
cd front_end
python3 -m http.server 5500