#!/bin/bash

clear
echo "======================================"
echo "🦺 Construction Safety Demo (Remote)"
echo "======================================"
echo ""

# Kill any existing processes
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8080 | xargs kill -9 2>/dev/null

# Start backend
echo "Starting backend server..."
cd backend
source ../venv/bin/activate
python3 main.py &
BACKEND_PID=$!
cd ..

# Start frontend server
echo "Starting frontend server..."
cd frontend
python3 -m http.server 8080 &
FRONTEND_PID=$!
cd ..

# Wait for backend
echo "Waiting for services to start..."
sleep 5

echo ""
echo "======================================"
echo "✅ Demo is running!"
echo "======================================"
echo ""
echo "Access from your Windows laptop:"
echo "👉 http://192.168.10.117:8080"
echo ""
echo "Backend API: http://192.168.10.117:8000"
echo ""
echo "Press Ctrl+C to stop the demo"
echo "======================================"

# Cleanup function
cleanup() {
    echo "Shutting down..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup INT
wait
