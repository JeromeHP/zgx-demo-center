# Document Translation Demo

A demonstration application showcasing multilingual translation capabilities using Meta's NLLB-200 (No Language Left Behind) model. This demo is designed for HP ZGX Nano sales and marketing teams to showcase AI-powered translation at events.

## Overview

This demo provides a web-based interface for translating text between 27 languages using the NLLB-200-distilled-600M model. The application features a FastAPI backend that handles model loading and translation requests, paired with an HTML frontend for easy interaction.

### Key Features

- Support for 54 language pair combinations (27 languages to/from English)
- Real-time translation through an intuitive web interface
- On-demand model loading to optimize resource usage
- Sample text buttons for quick demonstrations
- Visual model status indicator

### Supported Languages

The demo supports translation between English and the following languages: Arabic, Chinese, Croatian, Czech, Danish, Dutch, Finnish, French, German, Hungarian, Indonesian, Italian, Japanese, Korean, Malay, Norwegian, Norwegian Bokmal, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Ukrainian, and Vietnamese.

## System Requirements

- HP ZGX Nano AI Station (or compatible Linux system with NVIDIA GPU)
- Python 3.x
- CUDA-compatible GPU with 8GB+ VRAM recommended
- Active network connection for initial model download

## Directory Structure

```
document-translate/
├── backend/
│   ├── main.py              # FastAPI backend server
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Web interface
│   └── hp_logo.png          # HP branding asset
├── document-translate-env/  # Virtual environment (must exist before install)
├── install.sh               # Installation script
├── start_demo_remote.sh     # Demo startup script
└── download_models.sh       # Model download from S3 (optional)
```

## Installation

### Step 1: Create the Virtual Environment

Before running the installation script, you must create the virtual environment:

```bash
python3 -m venv document-translate-env
```

### Step 2: Run the Installation Script

```bash
chmod +x install.sh
./install.sh
```

The installation script will:
- Verify Python 3 is installed
- Activate the existing virtual environment
- Upgrade pip
- Install all required Python dependencies

### Step 3: Start the Demo

```bash
chmod +x start_demo_remote.sh
./start_demo_remote.sh
```

The startup script will:
- Detect and display your server IP address
- Clean up any processes on ports 8000 and 8080
- Start the FastAPI backend on port 8000
- Start the frontend web server on port 8080
- Automatically update the frontend to use your server IP

## Configuration for Sales Teams

**CRITICAL:** The following items require modification when deploying to different environments.

### Server IP Address

The `start_demo_remote.sh` script automatically detects your server IP. However, a fallback IP is hard-coded:

**File:** `start_demo_remote.sh`
```bash
SERVER_IP=$(hostname -I | awk '{print $1}')
if [ -z "$SERVER_IP" ]; then
    SERVER_IP="192.168.10.117"  # <-- UPDATE THIS FALLBACK IP
fi
```

**File:** `frontend/index.html`
```javascript
const API_URL = 'http://192.168.10.117:8000';  // <-- This is updated automatically by start script
```

The startup script modifies `index.html` automatically, but if running manually, update the `API_URL` to match your server.

### S3 Model Download (Optional)

If using the S3 model download script, update the bucket path:

**File:** `download_models.sh`
```bash
aws s3 cp s3://finetuning-demo-models/nllb-200-distilled-600M/ ~/.cache/huggingface/hub/models--facebook--nllb-200-distilled-600M/ --recursive
```

Update `finetuning-demo-models` to your organization's S3 bucket if applicable.

## Running the Demo

### Accessing the Interface

After starting the demo, access it from any device on the same network:

```
http://YOUR_SERVER_IP:8080
```

The startup script displays the exact URL to use.

### Demo Workflow

1. Open the web interface in your browser
2. Click the "Load Models" button (first-time load takes 1-2 minutes)
3. Wait for the status indicator to turn green
4. Select a language pair from the dropdown menu
5. Enter text to translate or click a sample text button
6. Click "Translate Text" to see the translation

### Stopping the Demo

Press `Ctrl+C` in the terminal running the startup script. This cleanly shuts down both the backend and frontend servers.

## Model Information

This demo uses:
- **Model:** facebook/nllb-200-distilled-600M
- **Size:** ~600M parameters (~1.2GB on disk)
- **Precision:** float16 (for reduced memory usage)
- **Source:** Hugging Face Hub (downloaded on first load) or optional S3 pre-download

The model is loaded on-demand when you click "Load Models" in the interface. Initial loading downloads from Hugging Face if not cached locally.

## Troubleshooting

### "Models not loaded" Error

Click the "Load Models" button in the web interface before attempting to translate. The status indicator should turn green when models are ready.

### Cannot Connect to Backend

1. Verify the backend is running:
   ```bash
   curl http://localhost:8000/
   ```
2. Check that port 8000 is not blocked by firewall
3. Ensure the `API_URL` in `index.html` matches your server IP

### Port Already in Use

The startup script automatically kills processes on ports 8000 and 8080. If issues persist:
```bash
lsof -ti:8000 | xargs kill -9
lsof -ti:8080 | xargs kill -9
```

### Translation Errors

- Ensure the selected language pair is supported
- Check that the input text is not empty
- Verify model loaded successfully (green status indicator)
- Monitor terminal for error messages

### Virtual Environment Not Found

If you see "Virtual environment 'document-translate-env' not found":
```bash
python3 -m venv document-translate-env
```
Then re-run the installation script.

## Demo Tips for Sales Teams

- Load models before the audience arrives (takes 1-2 minutes)
- Use the sample text buttons for quick, consistent demonstrations
- Start with English to French or Spanish translations for familiar results
- Show bidirectional translation by switching language pairs
- Demonstrate multiple languages to showcase the model's breadth
- Keep translations concise for faster response times

## Files Reference

| File | Purpose |
|------|---------|
| `main.py` | FastAPI backend handling model loading and translation requests |
| `index.html` | Web interface with language selection and translation display |
| `requirements.txt` | Python package dependencies |
| `install.sh` | One-time installation script |
| `start_demo_remote.sh` | Demo startup script with auto-IP detection |
| `download_models.sh` | Optional S3 model pre-download script |

## Support

If you have questions about this demo contact Curtis Burkhalter at curtisburkhalter@gmail.com
