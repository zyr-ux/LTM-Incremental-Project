import hashlib
import wave
from pathlib import Path
from typing import Any, Dict, Optional, cast
import numpy as np

try:
    from transformers import pipeline as hf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    hf_pipeline = None
    TRANSFORMERS_AVAILABLE = False

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_VOICE_DIR = ROOT_DIR / "data" / "processed" / "voice_replies"
DEFAULT_TTS_MODEL = "facebook/mms-tts-eng"


def save_wav_file(audio_array: np.ndarray, sample_rate: int, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audio_flat = np.asarray(audio_array, dtype=np.float32).flatten()
    max_val = float(np.max(np.abs(audio_flat))) if audio_flat.size > 0 else 1.0
    if max_val > 1.0:
        audio_flat = audio_flat / max_val
    pcm_data = (np.clip(audio_flat, -1.0, 1.0) * 32767).astype(np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data.tobytes())
    return output_path


def _generate_fallback_audio(text: str, sample_rate: int = 16000) -> np.ndarray:
    duration = min(max(len(text) * 0.04, 0.5), 4.0)
    num_samples = int(sample_rate * duration)
    t = np.linspace(0.0, duration, num_samples, endpoint=False)
    digest = hashlib.md5(text.encode("utf-8")).digest()
    base_freq = 220.0 + (digest[0] % 120)
    waveform = 0.25 * np.sin(2.0 * np.pi * base_freq * t)
    return waveform.astype(np.float32)


class VoiceSynthesizer:
    def __init__(self, model_id: str = DEFAULT_TTS_MODEL, use_mock: bool = False):
        self.model_id = model_id
        self.use_mock = use_mock
        self.tts_pipeline: Any = None

    def _load_pipeline(self) -> None:
        if self.tts_pipeline is not None or self.use_mock or not TRANSFORMERS_AVAILABLE:
            return
        if hf_pipeline is None:
            return
        try:
            tts_fn = cast(Any, hf_pipeline)
            self.tts_pipeline = tts_fn("text-to-speech", model=self.model_id)
        except Exception:
            self.tts_pipeline = None

    def speak(self, text: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("Text for speech synthesis cannot be empty.")

        if output_path is None:
            text_hash = hashlib.md5(clean_text.encode("utf-8")).hexdigest()[:10]
            target_path = DEFAULT_VOICE_DIR / f"reply_{text_hash}.wav"
        else:
            target_path = Path(output_path)

        self._load_pipeline()
        if self.tts_pipeline is not None and not self.use_mock:
            result = self.tts_pipeline(clean_text)
            audio_data = np.asarray(result["audio"], dtype=np.float32)
            sample_rate = int(result["sampling_rate"])
            mode = "huggingface_tts"
        else:
            sample_rate = 16000
            audio_data = _generate_fallback_audio(clean_text, sample_rate=sample_rate)
            mode = "fallback_tone"

        saved_path = save_wav_file(audio_data, sample_rate, target_path)
        duration_seconds = round(float(len(audio_data)) / float(sample_rate), 2)

        return {
            "status": "Generated",
            "text": clean_text,
            "audio_path": str(saved_path),
            "sample_rate": sample_rate,
            "duration_seconds": duration_seconds,
            "mode": mode,
        }
