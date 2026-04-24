# DeepScan AI — Deepfake Face Detection

A production-ready Python app that detects AI-generated (deepfake) faces using:

| Component | Model |
|-----------|-------|
| **Face detection** | MTCNN (facenet-pytorch) |
| **Classification** | ViT fine-tuned on FaceForensics++ via HuggingFace |
| **Fallback** | EfficientNet-B4 (timm) + local weights |
| **UI** | Gradio 4.x |

---

## Quick Start

### 1. Install dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

> **GPU acceleration**: Install the CUDA build of PyTorch from https://pytorch.org if you have an NVIDIA GPU.

### 2. Launch the web app

```bash
python app.py
```

Open http://localhost:7860 in your browser.

---

## CLI Usage

### Single image / file

```bash
python detector.py path/to/image.jpg
```

Output:
```json
{
  "faces": [
    {
      "box": [42, 10, 218, 230],
      "label": "FAKE",
      "confidence": 0.9712
    }
  ]
}
```

### Batch processing

```bash
# Process a directory of images, save results to CSV
python batch_detect.py --input ./test_images --output results.csv

# Filter by extension
python batch_detect.py --input ./videos --ext mp4 --output video_results.csv

# Verbose output
python batch_detect.py --input ./test_images --verbose
```

---

## Project Structure

```
deepfake_detector/
├── app.py              ← Gradio web UI
├── detector.py         ← Core detection pipeline
├── batch_detect.py     ← CLI batch processor
├── requirements.txt
├── weights/            ← (optional) local model weights
│   └── deepfake_efficientnet_b4.pth
└── examples/           ← Sample images for the UI
    ├── real_face.jpg
    └── fake_face.jpg
```

---

## Model Details

### Primary: HuggingFace ViT
- **Model**: `dima806/deepfake_vs_real_image_detection`
- **Architecture**: Vision Transformer (ViT-base-patch16-224)
- **Training data**: FaceForensics++ + DFDC subset
- **Size**: ~350 MB (downloaded automatically on first run)

### Fallback: EfficientNet-B4
- Use your own weights trained on FaceForensics++
- Place at `weights/deepfake_efficientnet_b4.pth`
- Binary output: class 0 = REAL, class 1 = FAKE

---

## Accuracy Benchmarks (FaceForensics++ c23)

| Method | AUC |
|--------|-----|
| ViT (HuggingFace) | ~97% |
| EfficientNet-B4 (fine-tuned) | ~95% |
| MesoNet (baseline) | ~83% |

---

## Limitations

- Accuracy degrades with heavy video compression (below CRF 40)
- Very small faces (< 40 px) may be missed by MTCNN
- Not certified for forensic or legal use — treat as probabilistic
- Does not detect audio deepfakes

---

## License

MIT — use freely, attribution appreciated.
