# ARS-VG Analyzer

**AEM-REM Substitution and Vulnerability Graph Analyzer**

A forensic accounting research tool for detecting earnings manipulation through graph-based analysis, causal reasoning, and explainable AI.

> Yuan Ze University, College of Management | PhD Research

---

## What It Does

- Detects earnings manipulation patterns (AEM/REM substitution)
- Builds financial vulnerability graphs from SEC filings
- Retrieves similar historical fraud cases (JarFraud dataset)
- Generates explainable reports with interactive visualizations
- Benchmarks against Beneish M-Score and Dechow F-Score

---

## Quick Start (Local - No GPU Needed)

### Option A: One-Command Setup (Linux/Mac)

```bash
git clone https://github.com/ApoorvSaxena0109/cli-2.git
cd cli-2
bash setup.sh
source venv/bin/activate
python app.py
```

### Option B: Manual Setup (Any OS)

```bash
git clone https://github.com/ApoorvSaxena0109/cli-2.git
cd cli-2
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

pip install -r requirements.txt
python app.py
```

Then open **http://localhost:7860** in your browser.

### Option C: Share with Others Instantly

```bash
python app.py --share
```

This creates a **public Gradio URL** (valid for 72 hours) that anyone can access - no installation needed on their end. Send the URL to Professor Yanjie's CPA friends and they can test immediately.

---

## Making It Available for Others to Test

### Approach 1: Quick Share (Temporary - 72 hours)

Run locally and get an instant public URL:

```bash
python app.py --share
# Output: Running on public URL: https://abc123.gradio.live
```

Send that URL to anyone. No installation on their end. Re-run for a new link.

### Approach 2: Hugging Face Spaces (Permanent - Free)

**Best for long-term sharing.** A permanent URL that stays online 24/7.

1. Create free account at [huggingface.co](https://huggingface.co/join)
2. Go to [huggingface.co/new-space](https://huggingface.co/new-space)
3. Select **Gradio** SDK, **CPU Basic** (free), create the Space
4. Upload 3 files via the "Files" tab:
   - `app.py` (from this repo)
   - `requirements.txt` (from `deploy/huggingface-spaces/`)
   - `README.md` (from `deploy/huggingface-spaces/`)
5. Wait 2-3 minutes. Your permanent URL: `https://huggingface.co/spaces/YOUR_USERNAME/ars-vg-analyzer`

> See [DEPLOY.md](DEPLOY.md) for detailed step-by-step instructions, Docker deployment, and cloud VM options.

### Other Options

| Method | Duration | Cost | Best For |
|--------|----------|------|----------|
| `--share` | 72 hours | Free | Quick demos |
| HF Spaces | Permanent | Free | Sharing with testers |
| Docker (Render/Railway) | Permanent | Free tier | Production |
| Google Colab | Session-based | Free/Pro | Development |

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ApoorvSaxena0109/cli-2/blob/main/ARS_VG_Analyzer.ipynb)

---

## Command-Line Options

```
python app.py                    # Launch on localhost:7860
python app.py --share            # Generate public URL for sharing
python app.py --port 8080        # Use custom port
python app.py --no-ollama        # Disable LLM features (lightweight CPU mode)
```

---

## Optional: Enable LLM Reasoning

The analyzer works fully without LLM - graph analysis, substitution detection, and traditional scores all run on CPU. To add LLM-powered explanations:

```bash
# Install Ollama (free, runs locally)
curl -fsSL https://ollama.com/install.sh | sh

# Pull a small model (4GB, works on most laptops)
ollama pull deepseek-r1:7b

# Start the analyzer (it auto-detects Ollama)
python app.py
```

---

## Project Structure

```
cli-2/
├── app.py                            # Standalone application (run this)
├── ARS_VG_Analyzer.ipynb             # Original Colab notebook
├── requirements.txt                  # Python dependencies
├── setup.sh                          # One-command local setup script
├── Dockerfile                        # Docker deployment
├── DEPLOY.md                         # Detailed deployment guide
├── deploy/
│   └── huggingface-spaces/           # HF Spaces deployment files
│       ├── README.md                 # HF Spaces metadata
│       └── requirements.txt          # HF-specific dependencies
└── README.md                         # This file
```

---

## Architecture

```
Financial Data --> Ingestion --> Graph Model --> Substitution Detection
                                    |                    |
                                    v                    v
                            Case Retrieval      AEM/REM Scoring
                                    |                    |
                                    v                    v
                             LLM Synthesis    -->   Report + Visualization
```

**Modules:**
1. **Ingestion Service** - PDF/TXT parsing, SEC EDGAR data loading
2. **Reasoning Service** - LLM-powered analysis via Ollama (optional)
3. **Graph Service** - Financial vulnerability graph construction (NetworkX)
4. **Substitution Algorithm** - AEM/REM pattern detection
5. **Output Generation** - HTML reports, JSON exports, interactive graphs

---

## Research Foundation

- Graph-based detection captures relational patterns that ratio-based analysis misses
- Validated against JarFraud dataset (SEC AAER fraud labels)
- Ensemble method combines Beneish M-Score, Dechow F-Score, and graph-based scoring
- Statistical validation with cross-validated AUC-ROC, McNemar's test
