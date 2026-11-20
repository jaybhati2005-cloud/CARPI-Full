"""Speech-delivery measurements for CareerPilot interview recordings.

These are approximate coaching signals, not assessments of personality or ability.
"""
from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

import librosa
import numpy as np

try:
    import av
except ImportError:  # Optional; librosa/FFmpeg may still decode supported formats.
    av = None

FILLERS = re.compile(r"\b(um+|uh+|erm+|ah+|like|you know|basically|actually|sort of|kind of)\b", re.I)


def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w']+\b", text or ""))


def filler_word_count(text: str) -> int:
    return len(FILLERS.findall(text or ""))


def _load_audio(file_path: str):
    """Load common browser formats, using pydub/FFmpeg as a fallback."""
    try:
        audio, sample_rate = librosa.load(file_path, sr=16000, mono=True)
        if len(audio):
            return audio, sample_rate
    except Exception:
        pass

    try:
        from pydub import AudioSegment
        segment = AudioSegment.from_file(file_path).set_channels(1)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
        try:
            segment.export(wav_path, format="wav")
            return librosa.load(wav_path, sr=16000, mono=True)
        finally:
            Path(wav_path).unlink(missing_ok=True)
    except Exception as exc:
        raise ValueError(
            "Could not decode this recording. Install FFmpeg and restart CareerPilot, "
            "or record again using a supported browser."
        ) from exc


def analyze_audio(file_path: str, transcript: str = "") -> dict:
    """Calculate duration, speaking rate, filler counts, pitch and energy."""
    audio, sample_rate = _load_audio(file_path)
    if audio is None or not len(audio):
        raise ValueError("The recording appears to be empty.")

    duration = float(librosa.get_duration(y=audio, sr=sample_rate))
    if duration <= 0:
        raise ValueError("The recording appears to be empty.")

    words = word_count(transcript)
    try:
        rms = librosa.feature.rms(y=audio)[0]
        average_energy = float(np.mean(rms)) if len(rms) else 0.0
    except Exception:
        average_energy = 0.0

    pitch_values = np.array([], dtype=float)
    try:
        f0 = librosa.yin(audio, fmin=65, fmax=400, sr=sample_rate, frame_length=1024, hop_length=512)
        pitch_values = f0[np.isfinite(f0) & (f0 >= 65) & (f0 <= 400)]
    except Exception:
        pass

    return {
        "duration_seconds": round(duration, 2),
        "word_count": words,
        "speaking_rate_wpm": round(words / duration * 60, 1) if words else None,
        "filler_word_count": filler_word_count(transcript),
        "average_energy": round(average_energy, 5),
        "average_pitch_hz": round(float(np.mean(pitch_values)), 1) if len(pitch_values) else None,
        "pitch_variation_hz": round(float(np.std(pitch_values)), 1) if len(pitch_values) else None,
        "note": "Speech measurements are approximate and affected by microphone quality, noise, and transcription accuracy.",
    }
