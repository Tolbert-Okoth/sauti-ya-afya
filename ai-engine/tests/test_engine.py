"""Run from ai-engine/:  python -m pytest -q tests"""
import os
import sys
import wave

import numpy as np
import pytest

ENGINE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_DIR)

import audio_pipeline as ap  # noqa: E402
import analyzer  # noqa: E402

SR = ap.SAMPLE_RATE


def breath_like(seconds=10, seed=0):
    """Band-limited noise with a slow breathing envelope plus a faint tone."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    noise = np.convolve(rng.standard_normal(len(t)), np.ones(8) / 8, mode="same")
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * t)
    y = 0.1 * envelope * noise + 0.02 * np.sin(2 * np.pi * 400 * t)
    return (y / np.max(np.abs(y)) * 0.5).astype(np.float32)


def write_wav(path, y):
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes())
    return str(path)


# --- pipeline -------------------------------------------------------------

def test_spectrogram_is_not_binary():
    img = np.array(ap.spectrogram_image(breath_like(5)))
    assert len(np.unique(img)) > 50


def test_spectrogram_does_not_depend_on_volume():
    y = breath_like(5)
    loud = np.array(ap.spectrogram_image(y), dtype=float)
    quiet = np.array(ap.spectrogram_image(y * 0.05), dtype=float)
    assert np.mean(np.abs(loud - quiet)) < 2.0


def test_windows_pad_tail_and_drop_tiny_pieces():
    assert len(ap.split_windows(np.zeros(SR * 12, dtype=np.float32))) == 3   # 5 + 5 + 2(padded)
    assert len(ap.split_windows(np.zeros(SR * 11, dtype=np.float32))) == 2   # 1 s tail dropped
    assert all(len(w) == ap.WINDOW_SAMPLES for w in ap.split_windows(np.zeros(SR * 12, dtype=np.float32)))


# --- analysis -------------------------------------------------------------

def test_analysis_is_deterministic_and_probabilities_sum_to_one(tmp_path):
    path = write_wav(tmp_path / "a.wav", breath_like(10))
    r1 = analyzer.analyze_audio(path)
    r2 = analyzer.analyze_audio(path)
    assert r1 == r2
    b = r1["biomarkers"]
    assert abs(b["prob_pneumonia"] + b["prob_asthma"] + b["prob_normal"] - 1.0) < 0.01


def test_volume_does_not_change_the_verdict(tmp_path):
    y = breath_like(10, seed=3)
    loud = analyzer.analyze_audio(write_wav(tmp_path / "loud.wav", y))["biomarkers"]
    quiet = analyzer.analyze_audio(write_wav(tmp_path / "quiet.wav", y * 0.1))["biomarkers"]
    assert loud["ai_diagnosis"] == quiet["ai_diagnosis"]
    assert abs(loud["prob_pneumonia"] - quiet["prob_pneumonia"]) < 0.05


def test_silence_is_rejected_not_low_risk(tmp_path):
    path = write_wav(tmp_path / "silence.wav", np.zeros(SR * 10, dtype=np.float32))
    with pytest.raises(analyzer.RecordingUnusable):
        analyzer.analyze_audio(path)


def test_too_short_is_rejected(tmp_path):
    path = write_wav(tmp_path / "short.wav", breath_like(1))
    with pytest.raises(analyzer.RecordingUnusable):
        analyzer.analyze_audio(path)


def test_symptoms_do_not_change_probabilities_but_danger_signs_escalate(tmp_path):
    path = write_wav(tmp_path / "a.wav", breath_like(10))
    base = analyzer.analyze_audio(path)
    with_symptoms = analyzer.analyze_audio(path, symptoms="fever, cough, convulsions")
    for k in ("prob_pneumonia", "prob_asthma", "prob_normal"):
        assert base["biomarkers"][k] == with_symptoms["biomarkers"][k]
    assert with_symptoms["risk_level_output"] == "High"
    assert "convulsions" in with_symptoms["biomarkers"]["danger_signs"]


@pytest.mark.parametrize("probs,threshold,expected", [
    ({"Asthma": 0.1, "Normal": 0.8, "Pneumonia": 0.1}, 0.75, ("Normal", "Low")),
    ({"Asthma": 0.2, "Normal": 0.6, "Pneumonia": 0.2}, 0.75, ("Inconclusive", "Medium")),
    ({"Asthma": 0.1, "Normal": 0.1, "Pneumonia": 0.8}, 0.75, ("Pneumonia", "High")),
    ({"Asthma": 0.55, "Normal": 0.3, "Pneumonia": 0.15}, 0.75, ("Suspected Asthma", "Medium")),
])
def test_decide(probs, threshold, expected):
    assert analyzer.decide(probs, threshold) == expected


def test_threshold_is_clamped():
    assert analyzer.clamp_threshold(5) == analyzer.MAX_THRESHOLD
    assert analyzer.clamp_threshold(0) == analyzer.MIN_THRESHOLD
    assert analyzer.clamp_threshold("nan") == analyzer.DEFAULT_THRESHOLD


# --- symptoms -------------------------------------------------------------

@pytest.mark.parametrize("text,present,danger", [
    ("no fever, no cough, breathing fine", [], []),
    ("wheezing and coughing", ["cough", "wheeze"], []),
    ("fever but no cough", ["fever"], []),
    ("homa na kikohozi", ["fever", "cough"], []),
    ("hakuna homa", [], []),
    ("unable to drink, chest indrawing", [], ["unable_to_drink_or_breastfeed", "chest_indrawing"]),
    ("mtoto hawezi kunyonya", [], ["unable_to_drink_or_breastfeed"]),
    ("shortness of breath", ["difficulty_breathing"], []),
])
def test_parse_symptoms(text, present, danger):
    got_present, got_danger = analyzer.parse_symptoms(text)
    assert sorted(got_present) == sorted(present)
    assert sorted(got_danger) == sorted(danger)


# --- API ------------------------------------------------------------------

@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app), main


def test_api_happy_path(client, tmp_path):
    c, _ = client
    path = write_wav(tmp_path / "a.wav", breath_like(10))
    with open(path, "rb") as fh:
        r = c.post("/analyze", files={"file": ("rec.wav", fh, "audio/wav")}, data={"threshold": "0.75"})
    assert r.status_code == 200
    job = c.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "completed"
    assert job["result"]["risk_level_output"] in {"Low", "Medium", "High"}


def test_api_silence_fails_with_retake(client, tmp_path):
    c, _ = client
    path = write_wav(tmp_path / "s.wav", np.zeros(SR * 10, dtype=np.float32))
    with open(path, "rb") as fh:
        r = c.post("/analyze", files={"file": ("rec.wav", fh, "audio/wav")})
    job = c.get(f"/status/{r.json()['job_id']}").json()
    assert job["status"] == "failed" and job.get("retake") is True


def test_api_rejects_bad_extension_and_traversal_name(client):
    c, _ = client
    r = c.post("/analyze", files={"file": ("../../evil.m3u8", b"#EXTM3U", "audio/x-mpegurl")})
    assert r.status_code == 415


def test_api_rejects_oversized_upload(client, monkeypatch):
    c, main = client
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    r = c.post("/analyze", files={"file": ("rec.wav", b"0" * 5000, "audio/wav")})
    assert r.status_code == 413
    assert not os.listdir(main.UPLOAD_DIR) or all(not f.endswith(".wav") for f in os.listdir(main.UPLOAD_DIR))


def test_api_key_enforced_when_configured(client, monkeypatch):
    c, main = client
    monkeypatch.setattr(main, "API_KEY", "secret")
    assert c.get("/status/abc").status_code == 401
    assert c.get("/status/abc", headers={"X-API-Key": "secret"}).status_code == 404


def test_jobs_expire(client, monkeypatch):
    c, main = client
    with main.JOBS_LOCK:
        main.JOBS["old"] = {"status": "completed", "created": 0}
    assert c.get("/status/old").status_code == 404
