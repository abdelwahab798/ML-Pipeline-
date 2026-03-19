from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import sys
import os
from deployment.deployment import load_artifacts, predict

app = FastAPI()

BASE_DIR = os.path.dirname(__file__)
model, vectorizer = load_artifacts(
    os.path.join(BASE_DIR, "model.pkl"),
    os.path.join(BASE_DIR, "vectorizer.pkl")
)
@app.get("/")
def home():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Spam Detector</title>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

        :root {
            --bg: #0a0a0f;
            --surface: #13131a;
            --border: #2a2a3a;
            --accent: #00ff88;
            --danger: #ff3d5a;
            --text: #e8e8f0;
            --muted: #6b6b80;
        }

        body {
            background: var(--bg);
            color: var(--text);
            font-family: 'DM Sans', sans-serif;
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }

        body::before {
            content: '';
            position: fixed;
            inset: 0;
            background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(0,255,136,0.08) 0%, transparent 70%);
            pointer-events: none;
        }

        .container {
            width: 100%;
            max-width: 640px;
        }

        .header {
            margin-bottom: 2.5rem;
        }

        .badge {
            display: inline-block;
            font-family: 'Space Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 0.15em;
            text-transform: uppercase;
            color: var(--accent);
            border: 1px solid rgba(0,255,136,0.3);
            padding: 0.25rem 0.75rem;
            border-radius: 2px;
            margin-bottom: 1rem;
        }

        h1 {
            font-size: 2.2rem;
            font-weight: 600;
            letter-spacing: -0.03em;
            line-height: 1.1;
            color: var(--text);
        }

        h1 span {
            color: var(--accent);
        }

        .subtitle {
            margin-top: 0.6rem;
            color: var(--muted);
            font-size: 0.95rem;
            font-weight: 300;
        }

        .card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 2rem;
        }

        label {
            display: block;
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.75rem;
        }

        textarea {
            width: 100%;
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            color: var(--text);
            font-family: 'DM Sans', sans-serif;
            font-size: 0.95rem;
            padding: 1rem;
            resize: vertical;
            min-height: 140px;
            outline: none;
            transition: border-color 0.2s;
        }

        textarea:focus {
            border-color: rgba(0,255,136,0.4);
        }

        textarea::placeholder { color: var(--muted); }

        button {
            margin-top: 1.25rem;
            width: 100%;
            padding: 0.9rem;
            background: var(--accent);
            color: #000;
            border: none;
            border-radius: 8px;
            font-family: 'Space Mono', monospace;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.1s;
        }

        button:hover { opacity: 0.85; }
        button:active { transform: scale(0.99); }
        button:disabled { opacity: 0.4; cursor: not-allowed; }

        .result {
            margin-top: 1.5rem;
            padding: 1.25rem 1.5rem;
            border-radius: 8px;
            display: none;
            align-items: center;
            gap: 1rem;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        .result.spam {
            background: rgba(255,61,90,0.1);
            border: 1px solid rgba(255,61,90,0.3);
        }

        .result.not-spam {
            background: rgba(0,255,136,0.08);
            border: 1px solid rgba(0,255,136,0.25);
        }

        .result-icon {
            font-size: 1.8rem;
            flex-shrink: 0;
        }

        .result-label {
            font-family: 'Space Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
            margin-bottom: 0.2rem;
        }

        .result-value {
            font-size: 1.2rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }

        .result.spam .result-value   { color: var(--danger); }
        .result.not-spam .result-value { color: var(--accent); }

        .loading {
            margin-top: 1.5rem;
            text-align: center;
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.1em;
            color: var(--muted);
            display: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="badge">ML Pipeline — v1.0</div>
            <h1>Spam<span>.</span>Detector</h1>
            <p class="subtitle">Paste any email content and let the model decide.</p>
        </div>

        <div class="card">
            <label for="emailText">Email Content</label>
            <textarea id="emailText" placeholder="Paste your email text here..."></textarea>
            <button id="predictBtn" onclick="runPredict()">Analyze Email</button>

            <div class="loading" id="loading">Analyzing...</div>

            <div class="result" id="result">
                <div class="result-icon" id="resultIcon"></div>
                <div>
                    <div class="result-label">Classification Result</div>
                    <div class="result-value" id="resultValue"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function runPredict() {
            const text = document.getElementById('emailText').value.trim();
            if (!text) return;

            const btn = document.getElementById('predictBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');

            btn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';

            try {
                const formData = new FormData();
                formData.append('text', text);

                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();

                const isSpam = data.prediction === 'spam';
                result.className = 'result ' + (isSpam ? 'spam' : 'not-spam');
                document.getElementById('resultIcon').textContent = isSpam ? '🚨' : '✅';
                document.getElementById('resultValue').textContent = isSpam ? 'Spam Detected' : 'Not Spam';
                result.style.display = 'flex';
            } catch (err) {
                alert('Something went wrong.');
            } finally {
                btn.disabled = false;
                loading.style.display = 'none';
            }
        }
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content, status_code=200)


@app.post("/predict")
def predict_spam(text: str = Form(...)):
    try:
        label = predict(text=text, model=model, vectorizer=vectorizer)
        return {"prediction": label}
    except Exception as e:
        return {"error": str(e)}