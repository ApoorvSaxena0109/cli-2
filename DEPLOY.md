# Deployment Guide - ARS-VG Analyzer

Two approaches to make the analyzer available for testing:

---

## Approach 1: Quick Share (Temporary - 72 hours)

Run locally and get an instant public URL. Good for quick demos.

```bash
# Setup (one time)
bash setup.sh
source venv/bin/activate

# Launch with public URL
python app.py --share
```

You'll see output like:
```
Running on public URL: https://abc123def456.gradio.live
```

Send that URL to anyone - they open it in a browser and can test immediately.
Valid for 72 hours. Re-run to get a new URL.

---

## Approach 2: Hugging Face Spaces (Permanent - Free)

A permanent public URL that stays online 24/7. No server to maintain.

### Step-by-Step:

1. **Create a Hugging Face account** (free)
   - Go to https://huggingface.co/join

2. **Create a new Space**
   - Go to https://huggingface.co/new-space
   - **Space name:** `ars-vg-analyzer` (or any name you want)
   - **License:** MIT
   - **SDK:** Select **Gradio**
   - **Hardware:** **CPU Basic** (free tier - this is enough!)
   - Click **Create Space**

3. **Upload 3 files** to the Space (use the "Files" tab → "Add file" → "Upload files"):

   | File | Source |
   |------|--------|
   | `app.py` | From this repo (the main app) |
   | `requirements.txt` | From `deploy/huggingface-spaces/requirements.txt` |
   | `README.md` | From `deploy/huggingface-spaces/README.md` |

4. **Wait 2-3 minutes** for the Space to build and deploy.

5. **Done!** Your permanent URL will be:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/ars-vg-analyzer
   ```

   Share this URL with anyone. It stays online permanently and is free.

### Alternative: Deploy via Git (for advanced users)

```bash
# Clone your HF Space repo
git clone https://huggingface.co/spaces/YOUR_USERNAME/ars-vg-analyzer
cd ars-vg-analyzer

# Copy files
cp /path/to/cli-2/app.py .
cp /path/to/cli-2/deploy/huggingface-spaces/requirements.txt .
cp /path/to/cli-2/deploy/huggingface-spaces/README.md .

# Push
git add -A
git commit -m "Deploy ARS-VG Analyzer"
git push
```

### Notes on HF Spaces:
- **Free CPU tier** is enough for all analysis (graph, scores, validation)
- LLM reasoning is auto-disabled (no Ollama server on HF)
- If the Space sleeps after inactivity, it wakes up when someone visits (~30s)
- To keep it always awake, upgrade to a paid tier ($0/month for persistent CPU)

---

## Approach 3: Docker (Any Cloud Provider)

Use this for Render, Railway, Fly.io, AWS, GCP, Azure, or any Docker host.

### Build and run locally:
```bash
docker build -t ars-vg-analyzer .
docker run -p 7860:7860 ars-vg-analyzer
```

### Deploy to Render (free tier):

1. Go to https://render.com → New → Web Service
2. Connect your GitHub repo (`ApoorvSaxena0109/cli-2`)
3. Settings:
   - **Environment:** Docker
   - **Instance Type:** Free
   - **Health Check Path:** `/`
4. Click Deploy

### Deploy to Railway (free credits):

1. Go to https://railway.app → New Project → Deploy from GitHub
2. Select this repo
3. It auto-detects the Dockerfile and deploys

### Deploy to Fly.io (free tier):

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch
fly deploy
```

---

## Comparison

| Feature | Quick Share | HF Spaces | Docker Cloud |
|---------|-----------|-----------|-------------|
| **Cost** | Free | Free | Free tier |
| **Duration** | 72 hours | Permanent | Permanent |
| **Setup time** | 1 minute | 5 minutes | 10 minutes |
| **Custom domain** | No | No (unless paid) | Yes |
| **GPU support** | Your machine | Paid upgrade | Paid upgrade |
| **LLM support** | Yes (local Ollama) | No | No (unless you add Ollama) |
| **Best for** | Quick demos | Sharing with testers | Production use |

---

## Recommended for Professor Yanjie's CPA Testers

**Use Hugging Face Spaces (Approach 2).** It's:
- Free and permanent
- No installation needed on their end
- Just share a URL
- Runs 24/7 without your computer being on
