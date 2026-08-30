# AI Chat & Problem Solver

A lightweight AI chatbot built from scratch with a custom-trained model. Runs on free tier services.

## Architecture

| Component | Tech | Free Tier |
|-----------|------|-----------|
| Model | SmolLM-135M (INT4 GGUF) | ~70MB |
| Training | LoRA/QLoRA on Kaggle | 30hrs/week GPU |
| Backend | FastAPI + llama.cpp | Render free (512MB) |
| Database | PostgreSQL | Neon free (0.5GB) |
| Frontend | HTML/CSS/JS | Render free |

## Quick Start (Deployment Only)

If you already have the quantized `.gguf` model file:

```bash
# 1. Install backend dependencies
pip install -r backend/requirements.txt

# 2. Set your Neon database URL
export DATABASE_URL="postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/dbname"

# 3. Run the server
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

## Full Training Pipeline

### Step 1: Download Datasets

```bash
pip install kaggle
# Set up Kaggle API token (kaggle.com/settings)
python training/download_datasets.py
```

### Step 2: Preprocess Data

```bash
python training/preprocess.py
```

### Step 3: Train on Kaggle

1. Go to kaggle.com/code → New Notebook
2. Settings → Accelerator → GPU T4 x2
3. Upload `training/train.py`
4. Run all cells (~3 hours)

### Step 4: Quantize to INT4

```bash
python training/quantize.py
```

### Step 5: Deploy

1. Push to GitHub
2. Connect Render to repo
3. Set `DATABASE_URL` env var (from Neon)
4. Deploy

## Database Setup (Neon)

1. Sign up at neon.tech (free)
2. Create a project
3. Run the SQL in `database.sql` in the SQL Console
4. Copy the connection string to Render env var `DATABASE_URL`

## Project Structure

```
├── backend/
│   ├── main.py          # FastAPI server
│   ├── model.py         # Model inference
│   ├── database.py      # PostgreSQL
│   ├── schemas.py       # Request/response models
│   └── requirements.txt
├── frontend/
│   ├── index.html       # Chat UI
│   ├── style.css        # Styling
│   └── script.js        # Frontend logic
├── training/
│   ├── download_datasets.py
│   ├── preprocess.py
│   ├── train.py
│   ├── quantize.py
│   └── requirements.txt
├── models/              # Saved weights (gitignored)
├── data/                # Datasets (gitignored)
├── database.sql         # Schema
├── render.yaml          # Deployment config
└── .gitignore
```

## Memory Usage

| Component | RAM |
|-----------|-----|
| Python runtime | ~80MB |
| FastAPI + uvicorn | ~30MB |
| Model (INT4 135M) | ~70MB |
| KV cache | ~10MB |
| **Total** | **~190MB** (fits in 512MB) |
