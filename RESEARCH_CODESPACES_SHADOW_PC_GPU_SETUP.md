# CODESPACES + SHADOW PC GPU SETUP GUIDE
## Research Material - December 23, 2025

**Status:** Infrastructure research for binary workflow
**Critical:** GitHub Codespaces GPU deprecated August 2025

---

## THE 2025 REALITY

```
❌ GitHub Codespaces GPU = DEPRECATED (August 29, 2025)
   - Azure NCv3-series VM retirement
   - No longer available

✅ New Best Practice:
   Codespaces (coding/CPU) + Shadow PC (GPU work)
```

---

## THE WORKFLOW

```
┌─────────────────────────────────────────┐
│ GitHub Codespaces (CPU Dev)             │
│ ├─ Code editing (VS Code in browser)    │
│ ├─ Git management                       │
│ ├─ Testing on CPU                       │
│ └─ Free tier: 120 core-hours/month      │
└─────────────────────────────────────────┘
              ↓ PUSH CODE ↓
┌─────────────────────────────────────────┐
│ Shadow PC Pro (RTX GPU)                 │
│ ├─ GPU acceleration (CUDA)              │
│ ├─ Training/inference                   │
│ ├─ Heavy processing                     │
│ └─ $60-80/month for RTX 3090            │
└─────────────────────────────────────────┘
              ↓ PUSH RESULTS ↓
┌─────────────────────────────────────────┐
│ Production (wherever needed)            │
└─────────────────────────────────────────┘
```

---

## SHADOW PC GPU SETUP

### Step 1: Verify GPU
```bash
nvidia-smi  # Should show GPU, driver, CUDA version
```

### Step 2: Install CUDA Toolkit
```bash
# Download from nvidia.com/cuda-downloads
nvcc --version  # Verify installation
```

### Step 3: Python Environment with GPU
```bash
python -m venv venv_cuda
.\venv_cuda\Scripts\activate  # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Step 4: Verify GPU Works
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Should show GPU name
```

---

## CRITICAL NOTES

### GPU Memory Management
```python
# After each batch
torch.cuda.empty_cache()

# Monitor usage
nvidia-smi -l 1  # Updates every second
```

### 30-Minute Timeout (Shadow PC)
```bash
# Use tmux for long-running jobs
tmux new-session -d -s training
tmux send-keys -t training "python train.py" Enter
```

### .devcontainer.json (Codespaces auto-setup)
```json
{
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "postCreateCommand": "pip install -r requirements.txt"
}
```

---

## COST BREAKDOWN

| Service | Cost | Notes |
|---------|------|-------|
| Codespaces | $0-30/month | Free tier covers most dev work |
| Shadow PC | $60-80/month | RTX 3090 recommended |
| GitHub Copilot | $10/month | Recommended |
| **Total** | **$70-100/month** | Professional setup |

---

## AI DEV AGENTS

### Claude Code (FREE - Use This)
- Ctrl+I in VS Code
- Integrated, instant feedback
- Perfect for GPU code

### Devin ($500/month - Optional)
- Autonomous for hours
- Creates PRs directly
- Overkill for most projects

---

## CHECKLIST

```
□ Shadow PC Pro subscription ($60-80/month)
□ NVIDIA drivers installed (nvidia-smi works)
□ CUDA Toolkit installed (nvcc --version works)
□ Python venv with GPU PyTorch
□ torch.cuda.is_available() = True
□ Codespaces configured
□ .devcontainer.json created
□ Binary workflow: Codespaces ↔ Shadow PC
```

---

**Saved:** December 23, 2025
**Status:** Infrastructure research complete
