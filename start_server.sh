# Production-ready startup script
#!/bin/bash

# start_server.sh - Robust server startup with proper signal handling

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MODEL_BACKEND="auto"
export UNSURE_THRESHOLD="0.60"

# Check dependencies
echo "Checking dependencies..."
if ! python -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "Installing missing dependencies..."
    pip install -r requirements.txt
fi

# Check model files
echo "Checking model files..."
if [ ! -f "models/lr_v1/model.joblib" ]; then
    echo "Warning: Model files not found. Run training scripts first."
fi

# Create data directory
mkdir -p data

# Function to handle cleanup
cleanup() {
    echo "Shutting down server..."
    kill $SERVER_PID 2>/dev/null
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start server with proper error handling
echo "Starting AI vs Human server..."
echo "Server will be available at: http://localhost:8000"
echo "Press Ctrl+C to stop the server"

# Run in background and capture PID
uvicorn app:app --host 0.0.0.0 --port 8000 --reload=false --access-log &
SERVER_PID=$!

# Wait for server process
wait $SERVER_PID