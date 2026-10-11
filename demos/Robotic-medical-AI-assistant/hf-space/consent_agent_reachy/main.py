"""
Consent Agent for Reachy Mini — Optimized with Expressions
------------------------------------------------------------
Medical triage voice agent with expressive robot movements.

States and their visual cues:
  READY:     Antennas up, head neutral — "I'm listening"
  LISTENING: Antennas wiggle, slight head tilt — "I hear you"
  THINKING:  Antennas down, head tilted up — "Processing..."
  SPEAKING:  Antennas bounce gently — "Here's my response"
"""

from reachy_mini.apps.app import ReachyMiniApp
from reachy_mini.reachy_mini import ReachyMini
import threading
import time
import logging
import os
import io
import wave
import subprocess
import requests
import numpy as np

try:
    from reachy_mini.utils import create_head_pose
except ImportError:
    create_head_pose = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = os.getenv("ZGX_API_URL", "http://192.168.10.123:8090")

# ALSA device for the Reachy Mini mic
ALSA_DEVICE = os.getenv("ALSA_DEVICE", "reachymini_audio_src")
ALSA_CHANNELS = 2
SAMPLE_RATE = 16000
SAMPLE_FORMAT = "S16_LE"

# VAD-based recording parameters
CHUNK_SECONDS = 1
SILENCE_THRESHOLD = 400
SILENCE_CHUNKS = 3          # Wait 3 silent chunks before stopping (more patient with pauses)
MIN_SPEECH_CHUNKS = 1       # Allow short responses
MAX_CHUNKS = 15

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("consent-agent")


# ---------------------------------------------------------------------------
# Robot expressions (non-blocking, fire-and-forget)
# ---------------------------------------------------------------------------

def expr_ready(reachy_mini):
    """Antennas up, head neutral — ready to listen."""
    try:
        head = create_head_pose(yaw=0, pitch=0, roll=0, degrees=True)
        reachy_mini.goto_target(head=head, antennas=np.array([0.3, -0.3]), duration=0.5)
    except Exception as e:
        logger.error(f"expr_ready error: {e}")


def expr_listening(reachy_mini):
    """Slight head tilt, antennas forward — I'm paying attention."""
    try:
        head = create_head_pose(yaw=0, pitch=-5, roll=8, degrees=True)
        reachy_mini.goto_target(head=head, antennas=np.array([0.5, -0.5]), duration=0.3)
    except Exception as e:
        logger.error(f"expr_listening error: {e}")


def expr_thinking(reachy_mini):
    """Head tilted up slightly, antennas down — processing."""
    try:
        head = create_head_pose(yaw=0, pitch=8, roll=0, degrees=True)
        reachy_mini.goto_target(head=head, antennas=np.array([-0.2, 0.2]), duration=0.4)
    except Exception as e:
        logger.error(f"expr_thinking error: {e}")


def expr_speaking(reachy_mini):
    """Gentle antenna bounce, head slightly forward — talking to you."""
    try:
        head = create_head_pose(yaw=0, pitch=-8, roll=0, degrees=True)
        reachy_mini.goto_target(head=head, antennas=np.array([0.4, -0.4]), duration=0.3)
    except Exception as e:
        logger.error(f"expr_speaking error: {e}")


def expr_antenna_wiggle(reachy_mini):
    """Quick antenna wiggle to show the robot is alive/happy."""
    try:
        reachy_mini.goto_target(antennas=np.array([0.6, -0.6]), duration=0.2)
        time.sleep(0.25)
        reachy_mini.goto_target(antennas=np.array([-0.3, 0.3]), duration=0.2)
        time.sleep(0.25)
        reachy_mini.goto_target(antennas=np.array([0.3, -0.3]), duration=0.2)
    except Exception as e:
        logger.error(f"expr_wiggle error: {e}")


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def record_chunk_alsa(duration: float = 1.0) -> bytes:
    """Record a short audio chunk using arecord, return raw PCM bytes."""
    cmd = [
        "arecord",
        "-D", ALSA_DEVICE,
        "-f", SAMPLE_FORMAT,
        "-r", str(SAMPLE_RATE),
        "-c", str(ALSA_CHANNELS),
        "-d", str(int(duration)),
        "-t", "raw",
        "-q",
        "-",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=duration + 3)
        if result.returncode != 0:
            return b""
        return result.stdout
    except Exception:
        return b""


def compute_rms_pcm(pcm_bytes: bytes) -> float:
    """Compute RMS energy from raw 16-bit PCM bytes."""
    if len(pcm_bytes) < 2:
        return 0.0
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def stereo_pcm_to_mono_wav(pcm_chunks: list, sample_rate: int) -> bytes:
    """Convert list of stereo raw PCM chunks to a mono WAV file."""
    raw = b"".join(pcm_chunks)
    samples = np.frombuffer(raw, dtype=np.int16)
    samples = samples.reshape(-1, 2)
    mono = samples.mean(axis=1).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(mono.tobytes())
    return buf.getvalue()


def wav_bytes_to_float32(wav_bytes: bytes) -> tuple:
    """Read WAV bytes and return (sample_rate, float32 numpy array)."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()

    if sample_width == 2:
        float_data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        float_data = np.frombuffer(raw, dtype=np.float32)

    if n_channels > 1:
        float_data = float_data.reshape(-1, n_channels)[:, 0]

    return sample_rate, float_data


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def wait_for_api(timeout: int = 120) -> bool:
    """Block until the ZGX AI API is reachable."""
    logger.info(f"Waiting for API at {API_URL} ...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            if r.status_code == 200:
                logger.info(f"API is up: {r.json()}")
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    logger.error(f"API not reachable after {timeout}s")
    return False


def send_audio_to_api(wav_bytes: bytes) -> tuple:
    """Send WAV audio to /process endpoint."""
    files = {"audio": ("recording.wav", wav_bytes, "audio/wav")}
    response = requests.post(f"{API_URL}/process", files=files, timeout=180)
    response.raise_for_status()
    transcript = response.headers.get("X-Transcript", "")
    ai_response = response.headers.get("X-Response", "")
    return response.content, transcript, ai_response


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

class ConsentAgentReachy(ReachyMiniApp):
    """
    Medical triage voice agent with expressive movements.

    Visual feedback:
      - READY: Antennas up, wiggle on startup
      - LISTENING: Head tilted, antennas forward
      - THINKING: Head up, antennas down
      - SPEAKING: Head forward, antennas animated
    """

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event):
        logger.info("=" * 60)
        logger.info("CONSENT AGENT — MEDICAL TRIAGE")
        logger.info("=" * 60)

        if not wait_for_api():
            logger.error("Cannot reach API — exiting.")
            return

        # Test ALSA
        logger.info(f"Testing ALSA device '{ALSA_DEVICE}'...")
        test = record_chunk_alsa(1)
        if not test:
            logger.error("ALSA recording failed — exiting.")
            return
        logger.info(f"ALSA OK ({len(test)} bytes)")

        # Start playback
        output_sr = reachy_mini.media.get_output_audio_samplerate()
        reachy_mini.media.start_playing()
        time.sleep(0.5)

        # Wake up the robot — motors take time to become available after app start
        logger.info("Waiting for motor control...")
        INIT_HEAD_POSE = create_head_pose(yaw=0, pitch=0, roll=0, degrees=True)
        awake = False
        for attempt in range(30):  # Try for up to 60 seconds
            try:
                reachy_mini.goto_target(INIT_HEAD_POSE, antennas=np.array([0.0, 0.0]), duration=1.0)
                time.sleep(1.5)
                # Check if we can wiggle antennas (sign that motors are live)
                reachy_mini.goto_target(antennas=np.array([0.3, -0.3]), duration=0.3)
                time.sleep(0.5)
                reachy_mini.goto_target(antennas=np.array([0.0, 0.0]), duration=0.3)
                awake = True
                logger.info(f"Robot motors active (attempt {attempt + 1}).")
                break
            except Exception as e:
                logger.info(f"Motors not ready yet (attempt {attempt + 1})...")
                time.sleep(2)

        if awake:
            # Play wake-up sound and wiggle
            try:
                reachy_mini.media.play_sound("wake_up.wav")
                time.sleep(0.5)
            except Exception:
                pass
            expr_antenna_wiggle(reachy_mini)
            time.sleep(0.5)
        else:
            logger.warning("Motors did not respond after 60s — continuing without expressions.")

        # Speak greeting
        logger.info("Ready! Greeting patient...")
        expr_speaking(reachy_mini)

        # Generate and play greeting via TTS
        try:
            greeting = "Hi there! I'm your medical triage assistant. How can I help you today?"
            r = requests.post(
                f"{API_URL}/speak",
                json={"text": greeting},
                timeout=30,
            )
            if r.status_code == 200:
                resp_sr, resp_samples = wav_bytes_to_float32(r.content)
                if resp_sr != output_sr:
                    from scipy.signal import resample
                    n_out = int(len(resp_samples) * output_sr / resp_sr)
                    resp_samples = resample(resp_samples, n_out).astype(np.float32)

                chunk_size = output_sr // 10
                for i in range(0, len(resp_samples), chunk_size):
                    c = resp_samples[i:i + chunk_size]
                    reachy_mini.media.push_audio_sample(c)
                    time.sleep(len(c) / output_sr * 0.8)

                logger.info("Greeting played.")
                time.sleep(0.3)
        except Exception as e:
            logger.warning(f"Could not play greeting: {e}")

        expr_ready(reachy_mini)

        logger.info("Listening...")

        try:
            while not stop_event.is_set():
                try:
                    self._listen_and_respond(reachy_mini, stop_event, output_sr)
                except Exception as e:
                    logger.error(f"Error: {e}", exc_info=True)
                    if not stop_event.wait(2):
                        continue
        finally:
            # Return to neutral on exit
            try:
                expr_ready(reachy_mini)
                reachy_mini.media.stop_playing()
            except Exception:
                pass

        logger.info("Shutting down.")

    def _listen_and_respond(self, reachy_mini, stop_event, output_sr):
        """One cycle: detect speech, record until silence, process, play."""

        # === READY STATE — antennas up, waiting ===
        expr_ready(reachy_mini)

        # ------------------------------------------------------------------
        # Phase 1: Wait for speech
        # ------------------------------------------------------------------
        while not stop_event.is_set():
            chunk = record_chunk_alsa(CHUNK_SECONDS)
            if not chunk:
                continue

            rms = compute_rms_pcm(chunk)
            if rms > SILENCE_THRESHOLD:
                logger.info(f"Speech detected (RMS={rms:.0f})")
                # === LISTENING STATE — tilt head, antennas forward ===
                expr_listening(reachy_mini)
                break
        else:
            return

        # ------------------------------------------------------------------
        # Phase 2: Keep recording until silence
        # ------------------------------------------------------------------
        chunks = [chunk]
        speech_chunks = 1
        silent_consecutive = 0

        for _ in range(MAX_CHUNKS - 1):
            if stop_event.is_set():
                return

            chunk = record_chunk_alsa(CHUNK_SECONDS)
            if not chunk:
                continue

            chunks.append(chunk)
            rms = compute_rms_pcm(chunk)

            if rms < SILENCE_THRESHOLD:
                silent_consecutive += 1
                if silent_consecutive >= SILENCE_CHUNKS:
                    logger.info("Silence — done recording.")
                    break
            else:
                silent_consecutive = 0
                speech_chunks += 1

        if speech_chunks < MIN_SPEECH_CHUNKS:
            logger.info("Too little speech — skipping.")
            return

        duration = len(chunks) * CHUNK_SECONDS
        logger.info(f"Captured ~{duration}s ({speech_chunks} speech chunks). Sending...")

        # === THINKING STATE — head up, antennas down ===
        expr_thinking(reachy_mini)

        # ------------------------------------------------------------------
        # Phase 3: Convert and send to API
        # ------------------------------------------------------------------
        mono_wav = stereo_pcm_to_mono_wav(chunks, SAMPLE_RATE)

        try:
            response_audio, transcript, ai_response = send_audio_to_api(mono_wav)

            logger.info(f"User said: {transcript}")
            logger.info(f"AI: {ai_response[:120]}...")

            if not transcript or not transcript.strip():
                logger.info("Empty transcript — skipping playback.")
                return

            # === SPEAKING STATE — head forward, antennas animated ===
            expr_speaking(reachy_mini)

            # ----------------------------------------------------------
            # Phase 4: Play response
            # ----------------------------------------------------------
            logger.info("Playing response...")
            resp_sr, resp_samples = wav_bytes_to_float32(response_audio)

            if resp_sr != output_sr:
                from scipy.signal import resample
                n_out = int(len(resp_samples) * output_sr / resp_sr)
                resp_samples = resample(resp_samples, n_out).astype(np.float32)

            chunk_size = output_sr // 10
            for i in range(0, len(resp_samples), chunk_size):
                if stop_event.is_set():
                    break
                c = resp_samples[i:i + chunk_size]
                reachy_mini.media.push_audio_sample(c)
                time.sleep(len(c) / output_sr * 0.8)

            logger.info("Playback complete.")
            time.sleep(0.3)

        except requests.HTTPError as e:
            if e.response and e.response.status_code == 400:
                logger.info("No speech detected by API.")
            else:
                logger.error(f"API error: {e}")
        except requests.ConnectionError:
            logger.error("Lost API connection.")
            time.sleep(5)
        except Exception as e:
            logger.error(f"Failed: {e}", exc_info=True)


if __name__ == "__main__":
    app = ConsentAgentReachy()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
