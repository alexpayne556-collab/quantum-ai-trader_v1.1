# Shadow PC (Windows) GPU Setup — One-Paste Guide

Goal: Install everything needed to run GPU-accelerated NLP (PyTorch + Transformers) on a Shadow Windows PC with an RTX 40-series GPU. Includes exact URLs and step-by-step instructions. Paste this into Perplexity Pro or follow directly.

---

## Prerequisites
- Shadow PC plan with NVIDIA RTX GPU (e.g., RTX 4080-class)
- Windows 10/11 (64-bit)
- Admin rights on Shadow

---

## Step 1 — Install NVIDIA GPU Driver
Best for AI/ML stability: Studio Driver (works fine for gaming too).
- NVIDIA Drivers main page: https://www.nvidia.com/Download/index.aspx
- Direct Studio Driver page: https://www.nvidia.com/en-us/geforce/drivers/

Manual search (use these selections):
- Product Type: GeForce
- Product Series: GeForce RTX 40 Series
- Product: GeForce RTX 4080 (or your exact card)
- Operating System: Windows 11 64-bit (or Windows 10 64-bit, if applicable)
- Download Type: Studio Driver
- Language: English (US)

Actions:
1) Click Search → Download.
2) Run installer → choose "Express Install".
3) Reboot when finished.

Optional verification (after reboot):
- NVIDIA Control Panel → System Information → Check driver version.
- If `nvidia-smi` is available, run it in Command Prompt to confirm GPU.

---

## Step 2 — Install Visual C++ Runtime (required by PyTorch on Windows)
- VC++ x64 Runtime (latest): https://aka.ms/vs/17/release/vc_redist.x64.exe
Actions: Download → Run → Next → Finish.

---

## Step 3 — Install Python (recommended: 3.11)
- Python for Windows: https://www.python.org/downloads/windows/
Actions:
1) Download Python 3.11.x (64-bit).
2) Run installer → Check "Add Python to PATH" → Install.

---

## Step 4 — Create Virtual Environment and Upgrade pip
Open PowerShell (Windows Terminal works too):
```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate
python -m pip install -U pip
```

---

## Step 5 — Install GPU-Enabled PyTorch (CUDA wheels)
Use the official selector to confirm the current index URL:
- PyTorch Get Started: https://pytorch.org/get-started/locally/

Example (CUDA 12.1 wheels):
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
If the site shows a newer CUDA build (e.g., cu126), replace the index URL accordingly.

Notes:
- You do NOT need to install the full CUDA Toolkit or cuDNN for inference; PyTorch wheels bundle the runtime.
- If you plan to compile custom CUDA ops later, install the full toolkit matching the wheel version.

---

## Step 6 — Install NLP Stack (Transformers + helpers)
```
pip install transformers accelerate datasets sentencepiece
```

Optional utilities:
```
pip install pandas requests beautifulsoup4
```

---

## Step 7 — Verify CUDA Availability
Quick check in the active venv:
```
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-gpu'); \
print('PyTorch:', torch.__version__)"
```
Expected:
- `CUDA available: True`
- Your GPU name (e.g., NVIDIA GeForce RTX 4080)

---

## Step 8 — First Run Timing Test (CPU vs GPU)
Save as `gpu_timing_check.py` on the Shadow desktop and run inside the venv:
```
import time, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "ProsusAI/finbert"
texts = ["Market outlook remains positive despite inflation concerns."] * 30

def run_inference(device_str):
    device = torch.device(device_str)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    torch.backends.cudnn.benchmark = True if device.type == "cuda" else False
    with torch.no_grad():
        # Warmup
        for _ in range(3):
            _ = model(**batch).logits
        start = time.time()
        _ = model(**batch).logits
        end = time.time()
    return end - start

cpu_time = run_inference("cpu")
gpu_time = run_inference("cuda" if torch.cuda.is_available() else "cpu")
print(f"CPU batch time: {cpu_time:.4f}s")
print(f"GPU batch time: {gpu_time:.4f}s (CUDA available: {torch.cuda.is_available()})")
```
Run:
```
python gpu_timing_check.py
```

---

## Optional — WSL2 Ubuntu Variant (if you prefer Linux inside Windows)
- Install WSL2: https://learn.microsoft.com/windows/wsl/install
  - Open PowerShell (Admin): `wsl --install`
- After Ubuntu is ready:
  - Update driver in Windows (already done) — WSL2 uses it.
  - Inside Ubuntu:
    ```
    sudo apt update
    sudo apt install -y python3-venv python3-pip
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -U pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate datasets sentencepiece
    python -c "import torch; print(torch.cuda.is_available())"
    ```

---

## Troubleshooting / Common Pitfalls
- `CUDA available: False` → Update NVIDIA driver and ensure you used the CUDA wheel (`--index-url .../cu121`). The default `pip install torch` is CPU-only.
- Python version mismatch → Use Python 3.10–3.12; avoid 3.13 until PyTorch publishes wheels.
- VRAM errors (OOM) → Reduce batch size, use mixed precision (`torch.autocast(device_type="cuda")`).
- Slow throughput → Ensure batching, `torch.no_grad()`, and `torch.backends.cudnn.benchmark = True`.

---

## Quick Checklist (copy to Perplexity as needed)
1) Install NVIDIA Studio Driver → reboot.
2) Install VC++ Runtime.
3) Install Python 3.11 → create venv → upgrade pip.
4) Install GPU PyTorch via CUDA wheels (use index URL).
5) Install Transformers + helpers.
6) Verify CUDA.
7) Run timing test.

That’s it — you’re GPU-ready on Shadow Windows.
