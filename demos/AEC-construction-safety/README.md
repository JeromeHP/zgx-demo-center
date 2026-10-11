# Construction Safety Assistant

A web-based construction safety assistant application that provides instant answers to safety-related questions using AI and offline fallback responses.

## Overview

This application consists of a FastAPI backend with an AI model for safety Q&A, and a simple HTML/JavaScript frontend with voice recognition capabilities. It can operate in both online (AI model) and offline (fallback responses) modes.

## Features

- AI-powered safety question answering using a fine-tuned language model
- Offline fallback mode with pre-configured safety responses
- Voice recognition for hands-free operation
- Text-to-speech for audio responses
- Quick action buttons for common safety queries
- Real-time connection status monitoring
- Emergency procedures quick access

## System Requirements

- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended for AI model)
- 3GB disk space for model files
- Modern web browser with speech recognition support (Chrome/Edge recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/curtburk/AEC-construction-safety.git
cd Construction-safety-demo
```

2. Run the installation script:
```bash
chmod +x install.sh
./install.sh
```

The installer will:
- Create a Python virtual environment
- Install required dependencies
- Download the AI model (first-time only, requires internet)
- Configure offline fallback responses

## Running the Application

### Local Development
```bash
./start_demo.sh
```

Access the application at: http://localhost:8080

### Remote Access
For accessing from another device on the network:
```bash
./start_demo_remote.sh
```

Access from other devices using the server's IP address (e.g., http://192.168.10.117:8080)

## Project Structure

```
Construction-safety-demo/
├── backend/
│   ├── main.py                    # FastAPI backend server
│   ├── requirements.txt           # Python dependencies
│   ├── offline_responses.json    # Fallback safety responses
│   └── model/                    # AI model storage (created on first run)
├── frontend/
│   └── index.html                # Web application interface
├── venv/                         # Python virtual environment
├── install.sh                    # Installation script
├── start_demo.sh                 # Local startup script
└── start_demo_remote.sh          # Remote access startup script
```

## API Endpoints

### GET /
Returns server status and model availability

### POST /ask
Submit a safety question
```json
{
  "question": "What PPE do I need?",
  "context": "General Construction Site"
}
```

Response:
```json
{
  "answer": "Required PPE includes...",
  "timestamp": "2024-10-14T10:30:00",
  "context": "General Construction Site",
  "mode": "ai"
}
```

## Configuration

### Backend Port
Default: 8000
Modify in `backend/main.py`:
```python
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Frontend Port
Default: 8080
Modify in startup scripts or run manually:
```bash
python3 -m http.server 8080
```

### Model Configuration
The application uses the ConstructionSafetyQA-1.2B-V1 model. Model path can be configured in `backend/main.py`:
```python
model_path = "./model/ConstructionSafetyQA-1.2B-V1"
```

## Offline Mode

If the AI model cannot be loaded or is unavailable, the application automatically falls back to pre-configured responses stored in `offline_responses.json`. This ensures the application remains functional even without model access.

## Browser Compatibility

- Chrome/Edge: Full support including voice recognition
- Firefox: Limited voice recognition support
- Safari: Limited voice recognition support

For best experience, use Chrome or Edge browsers.

## Safety Topics Covered

- Personal Protective Equipment (PPE)
- Ladder Safety
- Fall Protection
- Emergency Procedures
- Scaffold Safety
- Excavation and Trenching
- Electrical Safety and Lockout/Tagout
- Crane and Rigging Operations
- Hot Work and Welding
- Confined Space Entry
- Chemical Handling and Hazmat

## Troubleshooting

### Model fails to load
The application will automatically switch to offline mode. Check:
- Available disk space (3GB required)
- Internet connectivity (first-time download)
- Python dependencies installed correctly

### Port already in use
Kill existing processes:
```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:8080 | xargs kill -9
```

### Voice recognition not working
- Ensure using Chrome or Edge browser
- Grant microphone permissions when prompted
- Check browser console for errors

## Development

To modify offline responses, edit `backend/offline_responses.json` with your safety guidelines.

To customize the frontend appearance, modify the CSS in `frontend/index.html`.

To add new API endpoints, extend `backend/main.py` with additional FastAPI routes.

## DISCLAIMER

This project is provided as-is for construction safety demonstration purposes.

