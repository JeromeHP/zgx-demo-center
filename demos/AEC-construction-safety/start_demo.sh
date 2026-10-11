#!/bin/bash

clear
echo "======================================"
echo "🦺 Construction Safety Demo"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./install.sh first"
    exit 1
fi

# Kill any existing processes on port 8000
echo "Checking for existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null

# Start backend
echo "Starting backend server..."
cd backend
source ../venv/bin/activate
python3 main.py &
BACKEND_PID=$!
cd ..

# Wait for backend to initialize
echo ""
echo "Waiting for model to load..."
for i in {1..10}; do
    if curl -s http://localhost:8000 > /dev/null; then
        echo "✅ Backend ready!"
        break
    fi
    sleep 2
    echo "Loading... ($i/10)"
done

# Open frontend in browser
echo ""
echo "Opening frontend in browser..."
sleep 2

# Try different browsers
if command -v firefox &> /dev/null; then
    firefox frontend/index.html 2>/dev/null &
elif command -v chromium-browser &> /dev/null; then
    chromium-browser frontend/index.html 2>/dev/null &
elif command -v google-chrome &> /dev/null; then
    google-chrome frontend/index.html 2>/dev/null &
else
    xdg-open frontend/index.html 2>/dev/null &
fi

echo ""
echo "======================================"
echo "✅ Demo is running!"
echo "======================================"
echo ""
echo "Backend API: http://localhost:8000"
echo "Frontend: Opened in your browser"
echo ""
echo "Press Ctrl+C to stop the demo"
echo "======================================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down demo..."
    kill $BACKEND_PID 2>/dev/null
    exit 0
}

# Set trap to cleanup on Ctrl+C
trap cleanup INT

# Keep script running
wait $BACKEND_PID