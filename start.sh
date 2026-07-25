#!/bin/bash
# QuantFlow Local Launcher — no Docker required
# Usage: bash start.sh
set -e

PROJECT=/mnt/d/Dev/QuantFlow
PYTHON=/root/miniforge3/envs/quantflow/bin/python
export PYTHONPATH=$PROJECT

cd $PROJECT
echo "╔══════════════════════════════════════════╗"
echo "║      QuantFlow Local Platform            ║"
echo "╚══════════════════════════════════════════╝"

# Kill any existing services
fuser -k 3000/tcp 8000/tcp 8080/tcp 2>/dev/null || true
sleep 1

# ML Engine (port 8000)
echo -n "Starting ML Engine..."
$PYTHON -c "
import uvicorn, sys, os
sys.path.insert(0, '$PROJECT')
os.environ['MODEL_PATH'] = '$PROJECT/checkpoints/cascade_anp.pt'
os.environ['DATA_PATH'] = '$PROJECT/data/alpaca_5m.parquet'
os.environ['CACHE_DIR'] = '$PROJECT/data/cache'
uvicorn.run('docker.ml.server:app', host='0.0.0.0', port=8000, log_level='error')
" &>/tmp/qf_ml.log &
ML_PID=$!
echo " PID=$ML_PID"

# Backend (port 3000)
echo -n "Starting Backend..."
$PYTHON -c "
import uvicorn, sys, os
sys.path.insert(0, '$PROJECT')
os.environ['ML_ENGINE_URL'] = 'http://localhost:8000'
os.environ['PAPER_TRADING'] = 'true'
os.environ['JWT_SECRET'] = 'dev-secret'
uvicorn.run('docker.backend.server:app', host='0.0.0.0', port=3000, log_level='error')
" &>/tmp/qf_backend.log &
BE_PID=$!
echo " PID=$BE_PID"

# Frontend (port 8080)
echo -n "Starting Frontend..."
$PYTHON -m http.server 8080 --directory "$PROJECT/frontend-expo/dist" &>/tmp/qf_frontend.log &
FE_PID=$!
echo " PID=$FE_PID"

# Wait for services
echo ""
echo "Waiting for services..."
for i in {1..30}; do
  ML_OK=$(curl -s http://localhost:8000/health 2>/dev/null | grep -c "ok" || true)
  BE_OK=$(curl -s http://localhost:3000/health 2>/dev/null | grep -c "ok" || true)
  if [ "$ML_OK" -gt 0 ] && [ "$BE_OK" -gt 0 ]; then
    echo ""
    echo "✓ All services ready!"
    echo ""
    echo "  Frontend:  http://localhost:8080"
    echo "  API:       http://localhost:3000/docs"
    echo "  ML Engine: http://localhost:8000/health"
    echo ""
    echo "  Press Ctrl+C to stop all services."
    echo ""
    echo "  PIDs: ML=$ML_PID  Backend=$BE_PID  Frontend=$FE_PID"
    wait
    exit 0
  fi
  echo -n "."
  sleep 2
done

echo ""
echo "⚠ Some services failed to start. Check logs:"
echo "  ML:       tail /tmp/qf_ml.log"
echo "  Backend:  tail /tmp/qf_backend.log"
