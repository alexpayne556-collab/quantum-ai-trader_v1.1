# 📤 UPLOAD TO GOOGLE COLAB PRO

## Step 1: Upload Bundle
1. Open Google Colab: https://colab.research.google.com
2. Runtime → Change runtime type → T4 GPU or A100 GPU
3. Create new notebook or upload COLAB_PRO_VISUAL_NUMERICAL_TRAINER.ipynb
4. Upload this entire folder to Colab:
   ```python
   from google.colab import files
   uploaded = files.upload()  # Upload all files from colab_training_bundle/
   ```

## Step 2: Run Training
1. Execute all cells in order
2. Training will take ~15-30 minutes on T4 GPU
3. Watch for accuracy improvements and convergence

## Step 3: Download Trained Models
After training completes, download:
- `best_cnn_model.pth` (Visual pattern CNN)
- `best_numerical_model.pkl` (Technical indicator model)
- `feature_scaler.pkl` (Feature normalization)
- `optimized_ensemble_config_v2.json` (Final config)

## Step 4: Integrate Locally
Run: `python integrate_colab_models.py`

---

**Estimated Time:** 1-2 hours total
**Cost:** ~$0.50 (Colab Pro compute units)
**Expected Improvement:** +5-10% accuracy
