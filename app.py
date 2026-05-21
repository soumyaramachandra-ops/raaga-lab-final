from __future__ import annotations

import os
import sqlite3
import uuid
import wave
from datetime import datetime
from pathlib import Path

import librosa
import librosa.display
import matplotlib
import soundfile as sf

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, flash, redirect, render_template, request, session, url_for, send_from_directory
from werkzeug.utils import secure_filename

try:
    from moviepy import AudioFileClip
except Exception:  # pragma: no cover - depends on moviepy version
    AudioFileClip = None


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
GRAPH_FOLDER = BASE_DIR / "static" / "graphs"
PLACEHOLDER_GRAPH = GRAPH_FOLDER / "placeholder.png"
DB_PATH = BASE_DIR / "database.db" / "raaga_lab.sqlite3"
ALLOWED_EXTENSIONS = {"mp3", "wav", "mp4", "m4a", "ogg", "flac", "webm", "weba"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["GRAPH_FOLDER"] = str(GRAPH_FOLDER)
app.secret_key = os.environ.get("RAAGA_LAB_SECRET", "raaga-lab-local-dev")

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
GRAPH_FOLDER.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


@app.context_processor
def inject_auth_state() -> dict:
    return {"is_logged_in": bool(session.get("user_logged_in"))}


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                original_filename TEXT,
                stored_filename TEXT,
                duration REAL DEFAULT 0,
                dominant_frequency REAL DEFAULT 0,
                detected_note TEXT DEFAULT 'N/A',
                vocal_range TEXT DEFAULT 'N/A',
                pitch_stability REAL DEFAULT 0,
                timing_consistency REAL DEFAULT 0,
                waveform_path TEXT,
                pitch_graph_path TEXT,
                spectrogram_path TEXT,
                dominant_graph_path TEXT
            )
            """
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(analysis_sessions)").fetchall()}
        migrations = {
            "stored_filename": "TEXT",
            "spectrogram_path": "TEXT",
            "dominant_graph_path": "TEXT",
            "timing_consistency": "REAL DEFAULT 0",
        }
        for column, definition in migrations.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE analysis_sessions ADD COLUMN {column} {definition}")


def graph_exists(graph_filename: str) -> bool:
    global PLACEHOLDER_GRAPH

    if not graph_filename:
        return False
    # protect against malformed paths (only allow plain filenames)
    graph_filename = str(graph_filename).replace('\\\\', '/').split('/')[-1]
    # normalize to a plain filename (prevents Windows backslash mismatches)
    graph_filename = Path(graph_filename).name
    graph_path = GRAPH_FOLDER / graph_filename
    try:
        return graph_path.is_file() and graph_path.stat().st_size > 0
    except Exception:
        return False


def safe_graph_filename(graph_filename: str) -> str:
    return (str(graph_filename).replace('\\\\', '/').split('/')[-1]) if graph_exists(graph_filename) else PLACEHOLDER_GRAPH.name


def dashboard_graph_exists(graph_path: str) -> bool:
    if not graph_path:
        return False

    graph_path = str(graph_path).replace("\\", "/").lstrip("/")
    candidate = (GRAPH_FOLDER / graph_path).resolve()
    try:
        candidate.relative_to(GRAPH_FOLDER.resolve())
        return candidate.is_file() and candidate.stat().st_size > 0
    except Exception:
        return False


def dashboard_graph_assets(item: dict) -> list[dict]:
    assets = []
    for label, key, alt in (
        ("Waveform", "waveform_path", "Waveform preview"),
        ("Pitch", "pitch_graph_path", "Pitch preview"),
        ("Spectrum", "spectrogram_path", "Spectrogram preview"),
    ):
        path = str(item.get(key) or "").replace("\\", "/")
        if dashboard_graph_exists(path):
            assets.append({"label": label, "path": path, "alt": alt})
    return assets


@app.route("/static/graphs/<path:filename>")
def graphs_static_proxy(filename: str):
    # Reliability-only: serve placeholder when missing; keeps static URL pattern working.
    filename = str(filename).replace('\\\\', '/').split('/')[-1]
    if graph_exists(filename):
        return send_from_directory(GRAPH_FOLDER, filename)
    return send_from_directory(GRAPH_FOLDER, PLACEHOLDER_GRAPH.name)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def save_upload(file_storage, prefix: str = "audio") -> tuple[Path | None, str, str]:
    if not file_storage or not file_storage.filename:
        return None, "", "Please choose an audio file."

    if not allowed_file(file_storage.filename):
        allowed = ", ".join(sorted(ALLOWED_EXTENSIONS))
        return None, "", f"Unsupported file type. Upload one of: {allowed}."

    original = secure_filename(file_storage.filename) or f"{prefix}.wav"
    extension = original.rsplit(".", 1)[1].lower()
    stored = f"{uuid.uuid4().hex}_{prefix}.{extension}"
    path = UPLOAD_FOLDER / stored
    file_storage.save(path)
    if extension in {"webm", "weba"}:
        valid, message = validate_web_recording(path)
        if not valid:
            path.unlink(missing_ok=True)
            return None, "", message
    return path, stored, ""


def validate_web_recording(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size < 4096:
        return False, "Recording was too short or incomplete. Please record at least 2 seconds and try again."

    with path.open("rb") as handle:
        header = handle.read(4)

    if header != b"\x1a\x45\xdf\xa3":
        return False, "Recording file was not finalized correctly. Please try recording again."

    return True, ""


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    extension = path.suffix.lower()

    if extension in {".webm", ".weba"} and AudioFileClip is None:
        raise ValueError("Browser recordings cannot be decoded because video audio support is unavailable. Please upload WAV or MP3.")

    if extension in {".mp4", ".webm", ".weba", ".m4a"} and AudioFileClip is not None:
        extracted = UPLOAD_FOLDER / f"{path.stem}_extracted.wav"
        try:
            clip = AudioFileClip(str(path))
            try:
                clip.write_audiofile(str(extracted), logger=None)
            finally:
                clip.close()
        except Exception as exc:
            raise ValueError(
                "This recording could not be decoded. Please record at least 2 seconds, then try again or upload a WAV/MP3 file."
            ) from exc
        path = extracted

    try:
        if path.suffix.lower() == ".wav":
            y, sr = read_wave_file(path)
        else:
            y, sr = sf.read(str(path), always_2d=False)
    except Exception:
        raise ValueError("The uploaded audio format could not be decoded on this machine. Try WAV or MP3 if this was a browser recording.")

    if y is None or len(y) == 0:
        raise ValueError("The uploaded file did not contain readable audio.")

    y = np.asarray(y)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if np.issubdtype(y.dtype, np.integer):
        y = y.astype(np.float32) / max(float(np.iinfo(y.dtype).max), 1.0)
    else:
        y = y.astype(np.float32)
    if sr != 22050:
        target_length = max(1, int(len(y) * 22050 / max(sr, 1)))
        source_x = np.linspace(0, 1, num=len(y), endpoint=True)
        target_x = np.linspace(0, 1, num=target_length, endpoint=True)
        y = np.interp(target_x, source_x, y).astype(np.float32)
        sr = 22050

    peak = float(np.max(np.abs(y))) if y.size else 0
    if peak > 0:
        y = y / peak
    return y, sr


def read_wave_file(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sr = wav.getframerate()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 1:
        data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128) / 128
    elif sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
    elif sample_width == 3:
        raw = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
        signed = raw[:, 0].astype(np.int32) | (raw[:, 1].astype(np.int32) << 8) | (raw[:, 2].astype(np.int32) << 16)
        signed = np.where(signed & 0x800000, signed - 0x1000000, signed)
        data = signed.astype(np.float32) / 8388608
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648
    else:
        raise ValueError("Unsupported WAV bit depth.")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr


def frequency_to_note(frequency: float) -> str:
    if frequency <= 0 or not np.isfinite(frequency):
        return "N/A"

    midi = int(round(69 + 12 * np.log2(frequency / 440.0)))
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    octave = midi // 12 - 1
    return f"{names[midi % 12]}{octave}"


def safe_percent(value: float) -> int:
    return int(max(0, min(100, round(value))))


def plot_style(ax) -> None:
    ax.set_facecolor("#15111a")
    ax.tick_params(colors="#a9a1b8", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color((1, 1, 1, 0.12))
    ax.grid(True, alpha=0.14, color="#ffffff")


def save_waveform(y: np.ndarray, sr: int, stem: str) -> str:
    filename = f"{stem}_waveform.png"
    fig, ax = plt.subplots(figsize=(12, 3.4), facecolor="#15111a")
    plot_style(ax)
    times = np.arange(len(y)) / max(sr, 1)
    ax.plot(times, y, color="#ff9b73", alpha=0.95, linewidth=0.9)
    ax.set_title("Waveform", color="#fff8f0", fontsize=12, pad=10)
    ax.set_xlabel("Time", color="#cfc6d8")
    ax.set_ylabel("Amplitude", color="#cfc6d8")
    fig.tight_layout()
    fig.savefig(GRAPH_FOLDER / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_pitch_graph(pitch_values: list[float], stem: str) -> str:
    filename = f"{stem}_pitch.png"
    fig, ax = plt.subplots(figsize=(12, 3.4), facecolor="#15111a")
    plot_style(ax)
    if pitch_values:
        ax.plot(pitch_values, color="#76e7f2", linewidth=1.9)
        ax.fill_between(range(len(pitch_values)), pitch_values, alpha=0.14, color="#76e7f2")
    ax.set_title("Pitch Contour", color="#fff8f0", fontsize=12, pad=10)
    ax.set_xlabel("Frame", color="#cfc6d8")
    ax.set_ylabel("Hz", color="#cfc6d8")
    fig.tight_layout()
    fig.savefig(GRAPH_FOLDER / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_spectrogram(y: np.ndarray, sr: int, stem: str) -> str:
    filename = f"{stem}_spectrogram.png"
    fig, ax = plt.subplots(figsize=(12, 3.4), facecolor="#15111a")
    spec, freqs = frame_spectrum(y, sr)
    spec_db = 20 * np.log10(np.maximum(spec, 1e-8))
    extent = [0, len(y) / max(sr, 1), 0, freqs[-1] if len(freqs) else sr / 2]
    img = ax.imshow(spec_db, origin="lower", aspect="auto", extent=extent, cmap="magma")
    plot_style(ax)
    ax.set_title("Spectrogram", color="#fff8f0", fontsize=12, pad=10)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    fig.tight_layout()
    fig.savefig(GRAPH_FOLDER / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename


def save_dominant_graph(y: np.ndarray, sr: int, stem: str) -> tuple[str, list[float]]:
    filename = f"{stem}_dominant_frequency.png"
    spec, freqs = frame_spectrum(y, sr)
    track = []
    for frame in spec:
        if frame.size and np.max(frame) > 0:
            track.append(float(freqs[int(np.argmax(frame))]))

    fig, ax = plt.subplots(figsize=(12, 3.4), facecolor="#15111a")
    plot_style(ax)
    if track:
        ax.plot(track, color="#f7bf63", linewidth=1.7)
        ax.fill_between(range(len(track)), track, alpha=0.14, color="#f7bf63")
    ax.set_title("Dominant Frequency Track", color="#fff8f0", fontsize=12, pad=10)
    ax.set_xlabel("Frame", color="#cfc6d8")
    ax.set_ylabel("Hz", color="#cfc6d8")
    fig.tight_layout()
    fig.savefig(GRAPH_FOLDER / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filename, track


def extract_pitch_values(y: np.ndarray, sr: int) -> list[float]:
    spec, freqs = frame_spectrum(y, sr)
    values = []
    for frame in spec:
        peak = float(np.max(frame)) if frame.size else 0
        if peak <= 0:
            continue
        strongest = int(np.argmax(frame))
        pitch = float(freqs[strongest])
        if 50 <= pitch <= 2000 and peak > np.percentile(frame, 92):
            values.append(pitch)
    return values


def frame_spectrum(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512) -> tuple[np.ndarray, np.ndarray]:
    if len(y) < n_fft:
        padded = np.pad(y, (0, n_fft - len(y)))
        frames = padded.reshape(1, -1)
    else:
        starts = range(0, len(y) - n_fft + 1, hop_length)
        frames = np.array([y[start : start + n_fft] for start in starts])

    if frames.size == 0:
        frames = np.zeros((1, n_fft), dtype=np.float32)

    window = np.hanning(n_fft)
    spec = np.abs(np.fft.rfft(frames * window, axis=1))
    freqs = np.fft.rfftfreq(n_fft, d=1 / max(sr, 1))
    return spec, freqs


def frame_rms(y: np.ndarray, frame_length: int = 1024, hop_length: int = 512) -> np.ndarray:
    if len(y) < frame_length:
        return np.array([float(np.sqrt(np.mean(np.square(y))))]) if len(y) else np.array([])
    values = []
    for start in range(0, len(y) - frame_length + 1, hop_length):
        frame = y[start : start + frame_length]
        values.append(float(np.sqrt(np.mean(np.square(frame)))))
    return np.asarray(values)


def detect_onsets(y: np.ndarray, sr: int, hop_length: int = 512) -> np.ndarray:
    rms = frame_rms(y, hop_length * 2, hop_length)
    if rms.size < 3:
        return np.array([])
    novelty = np.diff(rms, prepend=rms[0])
    threshold = float(np.mean(novelty) + np.std(novelty))
    peaks = []
    for idx in range(1, len(novelty) - 1):
        if novelty[idx] > threshold and novelty[idx] >= novelty[idx - 1] and novelty[idx] >= novelty[idx + 1]:
            peaks.append(idx * hop_length / max(sr, 1))
    return np.asarray(peaks)


def analyze_audio(path: Path, stored_filename: str = "", original_filename: str = "", mode: str = "Solo") -> dict:
    y, sr = load_audio(path)
    stem = f"{uuid.uuid4().hex}_{path.stem}"
    duration = round(float(len(y) / max(sr, 1)), 2)
    rms = frame_rms(y)
    active_frames = int(np.sum(rms > max(float(np.max(rms)) * 0.08, 0.002))) if rms.size else 0
    silence_percent = safe_percent(100 - (active_frames / max(len(rms), 1) * 100))
    active_seconds = round(duration * (100 - silence_percent) / 100, 2)

    pitch_values = extract_pitch_values(y, sr)
    dominant_graph_path, dominant_track = save_dominant_graph(y, sr, stem)
    dominant_frequency = round(float(np.median(pitch_values or dominant_track or [0])), 2)
    average_pitch = round(float(np.mean(pitch_values or [dominant_frequency or 0])), 2)
    primary_note = frequency_to_note(dominant_frequency)

    note_values = [frequency_to_note(freq) for freq in pitch_values[:: max(1, len(pitch_values) // 18)]]
    extracted_notes = [note for note in dict.fromkeys(note_values) if note != "N/A"][:12]
    detected_notes = ", ".join(extracted_notes) if extracted_notes else primary_note

    if pitch_values:
        low_frequency = round(float(np.percentile(pitch_values, 10)), 2)
        high_frequency = round(float(np.percentile(pitch_values, 90)), 2)
        pitch_stability = safe_percent(100 - (np.std(pitch_values) / max(np.mean(pitch_values), 1) * 100))
    else:
        low_frequency = high_frequency = dominant_frequency
        pitch_stability = 0

    onsets = detect_onsets(y, sr)
    gaps = np.diff(onsets) if len(onsets) > 1 else np.array([])
    timing_consistency = safe_percent(100 - (np.std(gaps) / max(np.mean(gaps), 0.001) * 45)) if gaps.size else 0
    average_gap = round(float(np.mean(gaps)), 2) if gaps.size else 0

    waveform_path = save_waveform(y, sr, stem)
    pitch_graph_path = save_pitch_graph(pitch_values, stem)
    spectrogram_path = save_spectrogram(y, sr, stem)

    result = {
        "audio_path": stored_filename,
        "original_filename": original_filename or stored_filename,
        "duration": duration,
        "dominant_frequency": dominant_frequency,
        "detected_notes": detected_notes,
        "primary_note": primary_note,
        "pitch_summary": {
            "pitch_stability": pitch_stability,
            "voiced_frames": len(pitch_values),
            "average_pitch": average_pitch,
        },
        "vocal_range": {
            "low_note": frequency_to_note(low_frequency),
            "high_note": frequency_to_note(high_frequency),
            "range_semitones": max(0, round(12 * np.log2(max(high_frequency, 1) / max(low_frequency, 1)))) if high_frequency and low_frequency else 0,
            "low_frequency": low_frequency,
            "high_frequency": high_frequency,
        },
        "silence": {
            "silence_percent": silence_percent,
            "active_seconds": active_seconds,
            "silent_seconds": round(max(duration - active_seconds, 0), 2),
            "active_regions": max(1, int(len(onsets))),
        },
        "timing": {
            "timing_consistency": timing_consistency,
            "onset_count": int(len(onsets)),
            "average_gap": average_gap,
        },
        "extracted_notes": extracted_notes,
        "waveform_path": waveform_path,
        "pitch_graph_path": pitch_graph_path,
        "spectrogram_path": spectrogram_path,
        "dominant_graph_path": dominant_graph_path,
        "graph_paths": [
            {"label": "Waveform", "path": waveform_path},
            {"label": "Pitch contour", "path": pitch_graph_path},
            {"label": "Spectrogram", "path": spectrogram_path},
            {"label": "Dominant frequency", "path": dominant_graph_path},
        ],
        "mode": mode,
    }
    return result


def store_session(result: dict, mode: str = "Solo") -> None:
    values = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode": mode,
        "filename": result.get("audio_path", "") or result.get("original_filename", "analysis.wav"),
        "original_filename": result.get("original_filename", ""),
        "stored_filename": result.get("audio_path", ""),
        "audio_path": result.get("audio_path", ""),
        "duration": result.get("duration", 0),
        "dominant_frequency": result.get("dominant_frequency", 0),
        "detected_note": result.get("primary_note", "N/A"),
        "vocal_range": f"{result['vocal_range']['low_note']}-{result['vocal_range']['high_note']}",
        "pitch_stability": result["pitch_summary"].get("pitch_stability", 0),
        "timing_consistency": result["timing"].get("timing_consistency", 0),
        "silence_percent": result["silence"].get("silence_percent", 0),
        "sample_rate": 22050,
        "waveform_path": result.get("waveform_path", ""),
        "pitch_graph_path": result.get("pitch_graph_path", ""),
        "spectrogram_path": result.get("spectrogram_path", ""),
        "dominant_graph_path": result.get("dominant_graph_path", ""),
    }
    with sqlite3.connect(DB_PATH) as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(analysis_sessions)").fetchall()}
        columns = [column for column in values if column in existing]
        placeholders = ", ".join("?" for _ in columns)
        column_sql = ", ".join(columns)
        conn.execute(
            f"INSERT INTO analysis_sessions ({column_sql}) VALUES ({placeholders})",
            [values[column] for column in columns],
        )


def empty_analysis_context() -> dict:
    return {
        "waveform_path": "",
        "pitch_graph_path": "",
        "spectrogram_path": "",
        "dominant_graph_path": "",
        "audio_path": "",
        "duration": 0,
        "dominant_frequency": 0,
        "detected_notes": "",
        "primary_note": "N/A",
        "pitch_summary": {"pitch_stability": 0, "voiced_frames": 0, "average_pitch": 0},
        "vocal_range": {
            "low_note": "N/A",
            "high_note": "N/A",
            "range_semitones": 0,
            "low_frequency": 0,
            "high_frequency": 0,
        },
        "silence": {"silence_percent": 0, "active_seconds": 0, "silent_seconds": 0, "active_regions": 0},
        "timing": {"timing_consistency": 0, "onset_count": 0, "average_gap": 0},
        "extracted_notes": [],
        "graph_paths": [],
        "error": "",
    }


def fetch_recent_sessions(limit: int = 12) -> list[dict]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM analysis_sessions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    sessions = []
    for row in rows:
        item = dict(row)
        for key, value in list(item.items()):
            if value is None:
                item[key] = ""
        sessions.append(item)
    return sessions


def build_dashboard_data() -> tuple[dict, list[dict], list[dict]]:
    sessions = fetch_recent_sessions(16)
    stability = [round(item.get("pitch_stability") or 0) for item in reversed(sessions[-7:])]
    timing = [round(item.get("timing_consistency") or 0) for item in reversed(sessions[-7:])]
    labels = [item.get("created_at", "")[-5:] for item in reversed(sessions[-7:])]

    if not sessions:
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        stability = [72, 78, 81, 86, 90]
        timing = [64, 70, 74, 79, 84]

    freqs = [float(item.get("dominant_frequency") or 0) for item in sessions]
    low = sum(1 for freq in freqs if 0 < freq < 220)
    mid = sum(1 for freq in freqs if 220 <= freq < 520)
    high = sum(1 for freq in freqs if freq >= 520)
    note_counts = {}
    for item in sessions:
        note = item.get("detected_note") or "N/A"
        note_counts[note] = note_counts.get(note, 0) + 1

    top_notes = sorted(note_counts.items(), key=lambda item: item[1], reverse=True)[:6] or [("C4", 2), ("E4", 3), ("G4", 4)]
    avg_freq = round(sum(freqs) / len(freqs), 2) if freqs else 432
    best_stability = max(stability) if stability else 0

    analytics = {
        "average_frequency": avg_freq,
        "trend_labels": labels,
        "stability": stability,
        "timing": timing,
        "frequency_labels": ["Low", "Mid", "High"],
        "frequency_values": [low or 2, mid or 6, high or 3],
        "note_labels": [item[0] for item in top_notes],
        "note_values": [item[1] for item in top_notes],
        "insights": [
            {
                "title": "Pitch stability signal",
                "text": f"Best recent stability is {best_stability}%. Keep longer sustained notes in your warm-up loop.",
            },
            {
                "title": "Timing consistency",
                "text": f"Recent timing averages {round(sum(timing) / max(len(timing), 1))}%. Use a metronome pass before comparison takes.",
            },
            {
                "title": "Frequency center",
                "text": f"Your recent dominant-frequency center is {avg_freq} Hz across stored analyses.",
            },
        ],
    }
    graph_previews = []
    for item in sessions:
        if not any(item.get(key) for key in ("waveform_path", "pitch_graph_path", "spectrogram_path")):
            continue
        preview_item = dict(item)
        preview_item["dashboard_graph_assets"] = dashboard_graph_assets(item)
        graph_previews.append(preview_item)
        if len(graph_previews) == 6:
            break
    return analytics, sessions, graph_previews


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["user_logged_in"] = True
        # Preserve existing UI: store a simple display name from email prefix only for this session
        # (real username persistence is handled in the full DB auth implementation)
        session["user_name"] = request.form.get("email", "Artist").split("@")[0] or "Artist"
        flash("Welcome back. Dashboard is ready.", "success")
        return redirect(url_for("dashboard"))
    return render_template("login.html")



@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        session["user_logged_in"] = True
        session["user_name"] = request.form.get("full_name", "Artist") or "Artist"
        flash("Account ready. You can start analyzing immediately.", "success")
        return redirect(url_for("dashboard"))
    return render_template("signup.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        flash("Password recovery is mocked in the local build.", "success")
    return render_template("forgot_password.html")


@app.route("/features")
def features():
    return render_template("features.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        flash("Message received locally. Thanks for reaching out.", "success")
    return render_template("contact.html")


@app.route("/dashboard")
def dashboard():
    # Auth gate (minimal, preserves existing session-based UX)
    if not session.get("user_logged_in"):
        flash("Please log in to access your dashboard.", "error")
        return redirect(url_for("login"))

    analytics, recent_sessions, graph_previews = build_dashboard_data()
    total_sessions = len(fetch_recent_sessions(500))
    total_duration = round(sum(float(item.get("duration") or 0) for item in recent_sessions), 2)
    best_stability = max([round(item.get("pitch_stability") or 0) for item in recent_sessions] or [94])
    return render_template(
        "dashboard.html",
        current_user={"full_name": session.get("user_name", "Artist")},
        total_sessions=total_sessions or 12,
        total_duration=total_duration or 540,
        streak_days=min(max(total_sessions, 1), 14),
        best_stability=best_stability,
        analytics=analytics,
        recent_sessions=recent_sessions,
        graph_previews=graph_previews,
    )



@app.route("/practice")
def practice():
    return render_template("practice.html")


@app.route("/vocal-mode")
def vocal_mode():
    return render_template("vocal_mode.html")


@app.route("/instrument-mode")
def instrument_mode():
    return render_template("instrument_mode.html")


@app.route("/single-analysis", methods=["GET", "POST"])
def single_analysis():
    context = empty_analysis_context()

    if request.method == "POST":
        uploaded_path, stored_filename, error = save_upload(request.files.get("audio"), "solo")
        if error:
            flash(error, "error")
            context["error"] = error
        else:
            try:
                result = analyze_audio(
                    uploaded_path,
                    stored_filename=stored_filename,
                    original_filename=request.files["audio"].filename,
                    mode="Solo",
                )
                store_session(result, "Solo")
                context.update(result)
                flash("Analysis complete. Your graphs are ready.", "success")
            except Exception as exc:
                app.logger.exception("Solo analysis failed")
                flash(f"Analysis failed: {exc}", "error")
                context["error"] = str(exc)

    return render_template("single_analysis.html", **context)


@app.route("/comparison-analysis", methods=["GET", "POST"])
def comparison_analysis():
    comparison = None

    if request.method == "POST":
        ref_path, ref_stored, ref_error = save_upload(request.files.get("reference_audio"), "reference")
        perf_path, perf_stored, perf_error = save_upload(request.files.get("performance_audio"), "performance")
        error = ref_error or perf_error
        if error:
            flash(error, "error")
        else:
            try:
                reference = analyze_audio(
                    ref_path,
                    stored_filename=ref_stored,
                    original_filename=request.files["reference_audio"].filename,
                    mode="Reference",
                )
                performance = analyze_audio(
                    perf_path,
                    stored_filename=perf_stored,
                    original_filename=request.files["performance_audio"].filename,
                    mode="Performance",
                )
                store_session(reference, "Duel Reference")
                store_session(performance, "Duel Performance")

                pitch_gap = abs(reference["dominant_frequency"] - performance["dominant_frequency"])
                duration_gap = abs(reference["duration"] - performance["duration"])
                pitch_accuracy = safe_percent(100 - (pitch_gap / max(reference["dominant_frequency"], 1) * 100))
                timing_score = safe_percent(100 - (duration_gap / max(reference["duration"], 1) * 100))
                note_overlap = len(set(reference["extracted_notes"]) & set(performance["extracted_notes"]))
                scale_match = "Excellent" if note_overlap >= 4 else "Good" if note_overlap >= 2 else "Needs Work"

                comparison = {
                    "pitch_accuracy": pitch_accuracy,
                    "duration_difference": round(duration_gap, 2),
                    "timing_score": timing_score,
                    "scale_match": scale_match,
                    "ai_score": safe_percent((pitch_accuracy * 0.55) + (timing_score * 0.25) + min(note_overlap * 8, 20)),
                    "reference": reference,
                    "performance": performance,
                }
                flash("Comparison complete. Both takes are mapped side by side.", "success")
            except Exception as exc:
                app.logger.exception("Comparison analysis failed")
                flash(f"Comparison failed: {exc}", "error")

    return render_template("comparison_analysis.html", comparison=comparison)


init_db()


if __name__ == "__main__":
    app.run(debug=True)
