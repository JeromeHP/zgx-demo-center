---
title: Consent Agent Reachy
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# Medical Triage Voice Agent for Reachy Mini

An always-listening voice agent that runs entirely on local hardware (HP ZGX Nano) — no cloud APIs needed. Designed for medical office triage assistance.

## What it does

1. Listens continuously via the robot's built-in microphone
2. Detects speech using Silero VAD (voice activity detection)
3. Transcribes speech with faster-whisper (running locally on CPU)
4. Generates a medical triage response with Llama-3.1-8B-UltraMedical via vLLM (running locally on GPU)
5. Synthesizes speech with Piper TTS (running locally)
6. Plays the response through the robot's built-in speaker

## Requirements

- HP ZGX Nano (or any NVIDIA GPU with 24GB+ VRAM) running the ZGX AI API container
- Reachy Mini (Wireless) with SDK installed

## Architecture

```
Reachy Mini (mic/speaker)
    │
    │  HTTP (network)
    │
ZGX Nano Docker Container
    ├── faster-whisper STT (with Silero VAD)
    ├── vLLM + Llama-3.1-8B-UltraMedical
    └── Piper TTS
```