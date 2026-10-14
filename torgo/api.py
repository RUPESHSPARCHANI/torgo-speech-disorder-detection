"""
==============================================================================
api.py — Production REST API + Web UI for Dysarthric Speech Classification
==============================================================================
FastAPI web service with a rich interactive frontend for uploading audio files
and receiving instant speech pathology predictions with visual results.

Run server:
    python api.py
Then open: http://localhost:8000
==============================================================================
"""

import sys
import shutil
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from predict import predict_audio, get_production_bundle

sys.stdout.reconfigure(encoding="utf-8")


@asynccontextmanager
async def lifespan(app):
    """Warm up model bundle on server startup."""
    try:
        bundle = get_production_bundle()
        print(f"  API Startup: Successfully loaded model bundle v{bundle['version']}")
    except Exception as e:
        print(f"  Warning during model load: {e}")
    yield


app = FastAPI(
    title="TORGO Dysarthric Speech Pathology API",
    description="Production-ready REST API for automatic detection of dysarthria from voice audio recordings.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── HTML Web UI ─────────────────────────────────────────────────────────────
FRONTEND_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TORGO Speech Pathology Analyzer</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg-primary: #0a0e1a;
            --bg-card: #111827;
            --bg-card-hover: #1a2234;
            --accent-cyan: #00d4ff;
            --accent-pink: #ff4d8d;
            --accent-green: #00e68a;
            --accent-orange: #ffaa33;
            --accent-red: #ff4444;
            --text-primary: #f0f4ff;
            --text-secondary: #8b95b0;
            --border-subtle: rgba(255,255,255,0.06);
            --glow-cyan: 0 0 30px rgba(0,212,255,0.15);
        }

        body {
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            overflow-x: hidden;
        }

        /* Animated background */
        body::before {
            content: '';
            position: fixed; top: 0; left: 0; right: 0; bottom: 0;
            background:
                radial-gradient(ellipse 600px 400px at 20% 20%, rgba(0,212,255,0.04), transparent),
                radial-gradient(ellipse 500px 500px at 80% 80%, rgba(255,77,141,0.03), transparent);
            z-index: -1;
        }

        .container {
            max-width: 880px;
            margin: 0 auto;
            padding: 40px 24px;
        }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 48px;
        }
        .header-badge {
            display: inline-flex; align-items: center; gap: 8px;
            background: rgba(0,212,255,0.08);
            border: 1px solid rgba(0,212,255,0.2);
            border-radius: 100px;
            padding: 6px 18px;
            font-size: 12px; font-weight: 600;
            color: var(--accent-cyan);
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 20px;
        }
        .header-badge .dot {
            width: 7px; height: 7px;
            background: var(--accent-green);
            border-radius: 50%;
            animation: pulse-dot 2s infinite;
        }
        @keyframes pulse-dot {
            0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0,230,138,0.5); }
            50% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(0,230,138,0); }
        }
        .header h1 {
            font-size: 36px; font-weight: 800;
            background: linear-gradient(135deg, var(--accent-cyan) 0%, #a78bfa 50%, var(--accent-pink) 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.2;
            margin-bottom: 12px;
        }
        .header p {
            color: var(--text-secondary);
            font-size: 16px;
            max-width: 560px;
            margin: 0 auto;
            line-height: 1.6;
        }

        /* Upload Card */
        .upload-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 32px;
            transition: border-color 0.3s;
        }
        .upload-card:hover { border-color: rgba(0,212,255,0.15); }

        .drop-zone {
            border: 2px dashed rgba(0,212,255,0.25);
            border-radius: 16px;
            padding: 48px 32px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: rgba(0,212,255,0.02);
        }
        .drop-zone:hover, .drop-zone.drag-over {
            border-color: var(--accent-cyan);
            background: rgba(0,212,255,0.06);
            box-shadow: var(--glow-cyan);
        }
        .drop-zone .icon {
            font-size: 48px;
            margin-bottom: 16px;
        }
        .drop-zone h3 {
            font-size: 18px; font-weight: 600;
            margin-bottom: 8px;
        }
        .drop-zone p {
            color: var(--text-secondary);
            font-size: 14px;
        }
        .drop-zone .formats {
            margin-top: 12px;
            display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;
        }
        .drop-zone .formats span {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 6px;
            padding: 3px 10px;
            font-size: 12px; font-weight: 500;
            color: var(--text-secondary);
        }

        .file-input { display: none; }

        .file-selected {
            display: none;
            align-items: center;
            gap: 14px;
            background: rgba(0,212,255,0.06);
            border: 1px solid rgba(0,212,255,0.15);
            border-radius: 12px;
            padding: 16px 20px;
            margin-top: 20px;
        }
        .file-selected.show { display: flex; }
        .file-selected .file-icon { font-size: 28px; }
        .file-selected .file-info { flex: 1; }
        .file-selected .file-name { font-weight: 600; font-size: 14px; }
        .file-selected .file-size { color: var(--text-secondary); font-size: 12px; }
        .file-selected .remove-btn {
            background: none; border: none;
            color: var(--text-secondary);
            cursor: pointer; font-size: 20px;
            padding: 4px;
            transition: color 0.2s;
        }
        .file-selected .remove-btn:hover { color: var(--accent-red); }

        /* Analyze Button */
        .analyze-btn {
            width: 100%;
            padding: 16px;
            border: none;
            border-radius: 14px;
            font-size: 16px; font-weight: 700;
            cursor: pointer;
            margin-top: 24px;
            background: linear-gradient(135deg, var(--accent-cyan), #6366f1);
            color: white;
            letter-spacing: 0.5px;
            transition: all 0.3s;
            position: relative;
            overflow: hidden;
        }
        .analyze-btn:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(0,212,255,0.3);
        }
        .analyze-btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }
        .analyze-btn.loading {
            pointer-events: none;
        }

        /* Spinner */
        .spinner {
            display: none;
            width: 20px; height: 20px;
            border: 3px solid rgba(255,255,255,0.2);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin: 0 auto;
        }
        .loading .spinner { display: inline-block; }
        .loading .btn-text { display: none; }
        @keyframes spin { to { transform: rotate(360deg); } }

        /* Results Section */
        .results-section {
            display: none;
            animation: fadeUp 0.5s ease;
        }
        .results-section.show { display: block; }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* Diagnosis Banner */
        .diagnosis-banner {
            border-radius: 20px;
            padding: 36px;
            text-align: center;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }
        .diagnosis-banner::before {
            content: '';
            position: absolute; inset: 0;
            border-radius: 20px;
            padding: 1px;
            background: linear-gradient(135deg, transparent, rgba(255,255,255,0.1), transparent);
            mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            mask-composite: exclude;
            -webkit-mask-composite: xor;
        }
        .diagnosis-banner.control {
            background: linear-gradient(135deg, rgba(0,230,138,0.1), rgba(0,180,220,0.08));
            border: 1px solid rgba(0,230,138,0.2);
        }
        .diagnosis-banner.dysarthric {
            background: linear-gradient(135deg, rgba(255,68,68,0.1), rgba(255,170,51,0.08));
            border: 1px solid rgba(255,68,68,0.2);
        }
        .diagnosis-icon { font-size: 52px; margin-bottom: 12px; }
        .diagnosis-label {
            font-size: 14px; font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 6px;
        }
        .diagnosis-value {
            font-size: 32px; font-weight: 800;
            margin-bottom: 8px;
        }
        .diagnosis-banner.control .diagnosis-value { color: var(--accent-green); }
        .diagnosis-banner.dysarthric .diagnosis-value { color: var(--accent-red); }
        .diagnosis-confidence {
            font-size: 18px;
            font-weight: 600;
            color: var(--text-secondary);
        }
        .diagnosis-confidence strong {
            color: var(--text-primary);
        }

        /* Risk Badge */
        .risk-badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 16px;
            border-radius: 100px;
            font-size: 13px; font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 12px;
        }
        .risk-badge.low { background: rgba(0,230,138,0.12); color: var(--accent-green); border: 1px solid rgba(0,230,138,0.3); }
        .risk-badge.moderate { background: rgba(255,170,51,0.12); color: var(--accent-orange); border: 1px solid rgba(255,170,51,0.3); }
        .risk-badge.high { background: rgba(255,68,68,0.12); color: var(--accent-red); border: 1px solid rgba(255,68,68,0.3); }

        /* Details Grid */
        .details-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            margin-bottom: 24px;
        }
        @media (max-width: 600px) { .details-grid { grid-template-columns: 1fr; } }

        .detail-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 24px;
        }
        .detail-card h4 {
            font-size: 12px;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 16px;
        }
        .detail-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 0;
            border-bottom: 1px solid var(--border-subtle);
        }
        .detail-row:last-child { border-bottom: none; }
        .detail-row .label {
            font-size: 13px;
            color: var(--text-secondary);
        }
        .detail-row .value {
            font-size: 14px;
            font-weight: 600;
        }

        /* Probability Bar */
        .prob-bar-container {
            margin-top: 12px;
        }
        .prob-bar-label {
            display: flex; justify-content: space-between;
            font-size: 13px; margin-bottom: 6px;
        }
        .prob-bar-label .name { color: var(--text-secondary); }
        .prob-bar-label .pct { font-weight: 600; }
        .prob-bar {
            height: 10px;
            background: rgba(255,255,255,0.06);
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 14px;
        }
        .prob-bar-fill {
            height: 100%;
            border-radius: 10px;
            transition: width 0.8s ease;
        }
        .prob-bar-fill.control-bar { background: linear-gradient(90deg, var(--accent-cyan), var(--accent-green)); }
        .prob-bar-fill.dysarthric-bar { background: linear-gradient(90deg, var(--accent-orange), var(--accent-red)); }

        /* Recommendation Box */
        .recommendation {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 24px;
        }
        .recommendation h4 {
            font-size: 14px; font-weight: 700;
            margin-bottom: 10px;
        }
        .recommendation p {
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.7;
        }

        /* Footer */
        .footer {
            text-align: center;
            padding: 32px 0 16px;
            color: var(--text-secondary);
            font-size: 13px;
        }
        .footer a {
            color: var(--accent-cyan);
            text-decoration: none;
        }
        .footer a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-badge"><span class="dot"></span> Model Online</div>
            <h1>TORGO Speech Pathology Analyzer</h1>
            <p>Upload a voice recording to detect dysarthric speech patterns using AI-powered acoustic biomarker analysis.</p>
        </div>

        <!-- Upload Card -->
        <div class="upload-card">
            <div class="drop-zone" id="dropZone">
                <div class="icon">&#127908;</div>
                <h3>Drop your audio file here</h3>
                <p>or click to browse from your computer</p>
                <div class="formats">
                    <span>.wav</span><span>.mp3</span><span>.ogg</span><span>.flac</span><span>.m4a</span>
                </div>
            </div>
            <input type="file" class="file-input" id="fileInput" accept=".wav,.mp3,.ogg,.flac,.m4a,.aac">

            <div class="file-selected" id="fileSelected">
                <span class="file-icon">&#127925;</span>
                <div class="file-info">
                    <div class="file-name" id="fileName"></div>
                    <div class="file-size" id="fileSize"></div>
                </div>
                <button class="remove-btn" id="removeBtn" title="Remove file">&times;</button>
            </div>

            <button class="analyze-btn" id="analyzeBtn" disabled>
                <span class="btn-text">Analyze Speech</span>
                <span class="spinner"></span>
            </button>
        </div>

        <!-- Results -->
        <div class="results-section" id="results">
            <div class="diagnosis-banner" id="diagBanner">
                <div class="diagnosis-icon" id="diagIcon"></div>
                <div class="diagnosis-label">Diagnosis</div>
                <div class="diagnosis-value" id="diagValue"></div>
                <div class="diagnosis-confidence">Confidence: <strong id="diagConf"></strong></div>
                <div class="risk-badge" id="riskBadge"></div>
            </div>

            <div class="details-grid">
                <div class="detail-card">
                    <h4>Probabilities</h4>
                    <div class="prob-bar-container">
                        <div class="prob-bar-label">
                            <span class="name">Healthy Control</span>
                            <span class="pct" id="probControl"></span>
                        </div>
                        <div class="prob-bar"><div class="prob-bar-fill control-bar" id="barControl"></div></div>

                        <div class="prob-bar-label">
                            <span class="name">Dysarthric</span>
                            <span class="pct" id="probDys"></span>
                        </div>
                        <div class="prob-bar"><div class="prob-bar-fill dysarthric-bar" id="barDys"></div></div>
                    </div>
                </div>

                <div class="detail-card">
                    <h4>Acoustic Biomarkers</h4>
                    <div class="detail-row">
                        <span class="label">Energy Variability (RMS)</span>
                        <span class="value" id="bioRms"></span>
                    </div>
                    <div class="detail-row">
                        <span class="label">HF Spectral Contrast</span>
                        <span class="value" id="bioContrast"></span>
                    </div>
                    <div class="detail-row">
                        <span class="label">MFCC-1 Envelope</span>
                        <span class="value" id="bioMfcc"></span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Zero Crossing Rate</span>
                        <span class="value" id="bioZcr"></span>
                    </div>
                </div>
            </div>

            <div class="details-grid">
                <div class="detail-card">
                    <h4>Audio Metadata</h4>
                    <div class="detail-row">
                        <span class="label">Speech Duration</span>
                        <span class="value" id="metaDuration"></span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Sample Rate</span>
                        <span class="value" id="metaSr"></span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Silence Trimmed</span>
                        <span class="value" id="metaTrimmed"></span>
                    </div>
                </div>
                <div class="detail-card">
                    <h4>Inference</h4>
                    <div class="detail-row">
                        <span class="label">Model Used</span>
                        <span class="value" id="metaModel"></span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Latency</span>
                        <span class="value" id="metaLatency"></span>
                    </div>
                    <div class="detail-row">
                        <span class="label">Features Analyzed</span>
                        <span class="value">126</span>
                    </div>
                </div>
            </div>

            <div class="recommendation" id="recommendation">
                <h4>&#128203; Clinical Recommendation</h4>
                <p id="recText"></p>
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            TORGO Speech Pathology Analyzer &mdash; NIT Sikkim &mdash;
            <a href="/docs">API Documentation</a>
        </div>
    </div>

    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const fileSelected = document.getElementById('fileSelected');
        const fileName = document.getElementById('fileName');
        const fileSize = document.getElementById('fileSize');
        const removeBtn = document.getElementById('removeBtn');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const results = document.getElementById('results');

        let selectedFile = null;

        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        fileInput.addEventListener('change', () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });
        removeBtn.addEventListener('click', clearFile);

        function handleFile(file) {
            selectedFile = file;
            fileName.textContent = file.name;
            fileSize.textContent = (file.size / 1024).toFixed(1) + ' KB';
            fileSelected.classList.add('show');
            analyzeBtn.disabled = false;
            results.classList.remove('show');
        }

        function clearFile() {
            selectedFile = null;
            fileInput.value = '';
            fileSelected.classList.remove('show');
            analyzeBtn.disabled = true;
        }

        analyzeBtn.addEventListener('click', async () => {
            if (!selectedFile) return;
            analyzeBtn.classList.add('loading');
            analyzeBtn.disabled = true;
            results.classList.remove('show');

            const formData = new FormData();
            formData.append('file', selectedFile);

            try {
                const resp = await fetch('/predict?model=svm', { method: 'POST', body: formData });
                if (!resp.ok) {
                    const err = await resp.json();
                    throw new Error(err.detail || 'Server error');
                }
                const data = await resp.json();
                displayResults(data);
            } catch (e) {
                alert('Error: ' + e.message);
            } finally {
                analyzeBtn.classList.remove('loading');
                analyzeBtn.disabled = false;
            }
        });

        function displayResults(d) {
            const banner = document.getElementById('diagBanner');
            const isDys = d.is_dysarthric;
            banner.className = 'diagnosis-banner ' + (isDys ? 'dysarthric' : 'control');
            document.getElementById('diagIcon').textContent = isDys ? '\\u26A0\\uFE0F' : '\\u2705';
            document.getElementById('diagValue').textContent = d.diagnosis.toUpperCase();
            document.getElementById('diagConf').textContent = d.confidence_score + '%';

            const rb = document.getElementById('riskBadge');
            const riskClass = d.clinical_risk_level.includes('HIGH') ? 'high' : d.clinical_risk_level.includes('MODERATE') ? 'moderate' : 'low';
            rb.className = 'risk-badge ' + riskClass;
            rb.textContent = d.clinical_risk_level;

            document.getElementById('probControl').textContent = d.probabilities.control + '%';
            document.getElementById('probDys').textContent = d.probabilities.dysarthric + '%';
            document.getElementById('barControl').style.width = d.probabilities.control + '%';
            document.getElementById('barDys').style.width = d.probabilities.dysarthric + '%';

            const bio = d.key_acoustic_biomarkers;
            document.getElementById('bioRms').textContent = bio.energy_variability_rms_std;
            document.getElementById('bioContrast').textContent = bio.spectral_contrast_hf;
            document.getElementById('bioMfcc').textContent = bio.first_mfcc_envelope;
            document.getElementById('bioZcr').textContent = bio.zero_crossing_rate_mean;

            const meta = d.audio_metadata;
            document.getElementById('metaDuration').textContent = meta.speech_duration_seconds + 's';
            document.getElementById('metaSr').textContent = (meta.sample_rate_hz / 1000) + ' kHz';
            document.getElementById('metaTrimmed').textContent = meta.silence_trimmed ? 'Yes' : 'No';
            document.getElementById('metaModel').textContent = d.model_used;
            document.getElementById('metaLatency').textContent = d.inference_latency_ms + ' ms';

            document.getElementById('recText').textContent = d.clinical_recommendation;

            results.classList.add('show');
            results.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    </script>
</body>
</html>
"""


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, tags=["UI"])
def serve_frontend():
    """Serve the interactive web UI."""
    return HTMLResponse(content=FRONTEND_HTML)


@app.get("/health", tags=["Health"])
def health_check():
    """Check API status and model readiness."""
    try:
        bundle = get_production_bundle()
        return {
            "status": "healthy",
            "service": "TORGO Dysarthria Detection API",
            "model_version": bundle["version"],
            "features_tracked": bundle["num_features"],
            "supported_audio_formats": [".wav", ".mp3", ".ogg", ".flac", ".m4a"],
            "target_sample_rate": bundle["sample_rate"]
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


@app.post("/predict", tags=["Inference"])
async def predict_speech_file(
    file: UploadFile = File(..., description="Audio file of spoken speech"),
    model: str = Query("svm", pattern="^(svm|rf)$", description="Classifier: 'svm' or 'rf'")
):
    """Upload an audio file to analyze for dysarthria."""
    file_ext = Path(file.filename).suffix.lower()
    allowed_exts = [".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"]
    if file_ext not in allowed_exts:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{file_ext}'. Allowed: {allowed_exts}")

    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir) / file.filename

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        result = predict_audio(temp_path, model_choice=model)
        result["uploaded_filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/biomarkers", tags=["Documentation"])
def get_biomarker_info():
    """Information about key acoustic biomarkers."""
    return {
        "biomarkers": [
            {"name": "RMS Energy Variability (rms_std)", "significance": "Top indicator. Dysarthric speakers show high variability in vocal intensity."},
            {"name": "High-Frequency Spectral Contrast", "significance": "Measures harmonic clarity in upper frequencies."},
            {"name": "First MFCC Spectral Envelope", "significance": "Overall acoustic power distribution of the vocal tract."},
            {"name": "Zero Crossing Rate (ZCR)", "significance": "Reflects unvoiced phonetic transitions and breath pauses."},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 70)
    print("  TORGO Speech Pathology Analyzer")
    print("  Open in browser: http://localhost:8000")
    print("  API Documentation: http://localhost:8000/docs")
    print("=" * 70)
    uvicorn.run(app, host="127.0.0.1", port=8000)
