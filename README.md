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

## Deployment Options (Free, No Disconnections)

### 1. Hugging Face Spaces (Recommended for Sharing)

**Free. Always on. No disconnections. Just share a URL.**

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces) and create a new Space
2. Select **Gradio** as the SDK
3. Upload these files:
   - `app.py`
   - `requirements.txt`
4. The app will auto-deploy and give you a permanent URL

> This is the best option for letting others test it - they just click a link.

### 2. Google Colab (Original Method)

Open the notebook directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ApoorvSaxena0109/cli-2/blob/main/ARS_VG_Analyzer.ipynb)

> Note: Colab may disconnect during long sessions. Use `--share` or Hugging Face Spaces for persistent access.

### 3. Local Machine (Best for Development)

Works on any laptop/desktop. No GPU required - the graph-based analysis and traditional scores (Beneish, Dechow) run on CPU. Only the optional LLM reasoning feature benefits from GPU.

```bash
bash setup.sh
source venv/bin/activate
python app.py
```

### 4. Free Cloud VMs

These services offer free tiers that can run the analyzer 24/7:

| Service | Free Tier | Setup |
|---------|-----------|-------|
| [Render](https://render.com) | 750 hrs/month | Connect GitHub repo, set start command to `python app.py --port $PORT` |
| [Railway](https://railway.app) | $5 free credit | Connect GitHub repo, auto-detects Python |
| [Fly.io](https://fly.io) | 3 shared VMs | `fly launch` then `fly deploy` |

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
├── app.py                    # Standalone application (run this)
├── ARS_VG_Analyzer.ipynb     # Original Colab notebook
├── requirements.txt          # Python dependencies
├── setup.sh                  # One-command setup script
└── README.md                 # This file
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
