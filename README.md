<div align="center">

<img src="https://img.shields.io/badge/NeuraScan_AI-Brain_Powered_Analysis-6C3FC9?style=for-the-badge&logo=brain&logoColor=white" alt="NeuraScan AI" />

# 🧠 NeuraScan AI

### *AI-Powered Learning Disorder Detection from Handwriting*

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-CPU-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![Deploy on Render](https://img.shields.io/badge/Deploy-Render-46E3B7?style=flat-square&logo=render&logoColor=white)](https://render.com)

---

**NeuraScan AI** is a production-ready REST API that analyses children's handwriting samples to detect early indicators of **dyslexia** and **dysgraphia** using a dual-engine approach: a trained **Convolutional Neural Network (CNN)** for letter-reversal-based dyslexia detection and a **computer-vision feature extractor** for dysgraphia risk scoring. It also powers an adaptive **quiz engine** with performance analytics to support personalised learning.

[🚀 Quick Start](#-quick-start) · [📡 API Reference](#-api-reference) · [🤖 How It Works](#-how-it-works) · [🐳 Docker](#-docker-deployment) · [🛠️ Development](#️-development)

</div>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🚀 Quick Start](#-quick-start)
- [⚙️ Configuration](#️-configuration)
- [📡 API Reference](#-api-reference)
- [🤖 How It Works](#-how-it-works)
- [🐳 Docker Deployment](#-docker-deployment)
- [☁️ Cloud Deployment](#️-cloud-deployment)
- [🧪 Testing](#-testing)
- [📁 Project Structure](#-project-structure)
- [🛠️ Development](#️-development)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🔍 Handwriting Analysis
- **Dyslexia Detection** via trained CNN (letter reversal patterns)
- **Dysgraphia Risk Scoring** via spatial feature extraction
- **Image Validation** — two-stage ML + rule-based paper check
- Supports PNG, JPG, JPEG, PDF uploads (up to 16 MB)

</td>
<td width="50%">

### 🎓 Adaptive Quiz Engine
- Topic-based quiz generation
- External AI quiz API integration with local fallback
- **Quiz Performance Analytics** — gap analysis, strong/weak areas
- Actionable learning recommendations per student

</td>
</tr>
<tr>
<td>

### 🤖 ML Infrastructure
- TensorFlow CNN for dyslexia (trained on handwriting datasets)
- PyTorch MobileNetV2 document classifier (trained on RVL-CDIP)
- Graceful CPU-only inference with TensorFlow fallback
- Model hot-loading with in-memory caching

</td>
<td>

### 🏭 Production-Ready
- Gunicorn multi-worker WSGI server
- Docker-containerised, Render & HuggingFace Spaces ready
- Configurable via environment variables
- Automatic temp file cleanup after every request

</td>
</tr>
</table>

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        NeuraScan AI                         │
│                      Flask REST API                         │
└────────────┬──────────────────────────────┬────────────────┘
             │                              │
   ┌─────────▼──────────┐        ┌─────────▼──────────┐
   │  Handwriting        │        │   Quiz Engine       │
   │  Analysis Pipeline  │        │                     │
   │  ─────────────────  │        │  /quiz/generate     │
   │  1. paper_validator │        │  /quiz/analyze      │
   │     ├─ PyTorch ML   │        └─────────────────────┘
   │     └─ Rule-Based   │
   │  2. ml_models.py    │
   │     ├─ CNN Dyslexia │
   │     └─ Dysgraphia   │
   │        Features     │
   └────────────────────┘
```

### Core Components

| Module | Role |
|---|---|
| `app.py` | Flask routes, request handling, response formatting |
| `ml_models.py` | CNN-based dyslexia detection + dysgraphia feature extraction |
| `paper_validator.py` | Two-stage image validation (PyTorch ML + rule-based heuristics) |
| `model_pipeline.py` | OCR pipeline, quiz generation, external API integration |
| `gunicorn_config.py` | Production server configuration |
| `training/` | Jupyter notebooks for model training (Colab-ready) |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.9 + |
| Tesseract OCR | 4.x + |
| Docker *(optional)* | 20.x + |

### 1 · Clone the repository

```bash
git clone https://github.com/Niwas-Kumar/neurascan-python-ai.git
cd neurascan-python-ai
```

### 2 · Install system dependencies

```bash
# Ubuntu / Debian
sudo apt-get update && sudo apt-get install -y tesseract-ocr libgl1 libglib2.0-0

# macOS (Homebrew)
brew install tesseract
```

### 3 · Set up a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

### 4 · Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5 · Run the server

```bash
python app.py
```

The API will be live at **`http://localhost:5000`**.

> **Tip:** Use `gunicorn --config gunicorn_config.py app:app` for production-grade serving.

---

## ⚙️ Configuration

All options are controlled through **environment variables** — no code changes needed.

| Variable | Default | Description |
|---|---|---|
| `PORT` | `5000` | Server listening port |
| `TESSERACT_CMD` | `tesseract` | Path to the Tesseract binary |
| `EXTERNAL_AI_API_URL` | *(unset)* | URL of an external AI analysis service |
| `EXTERNAL_QUIZ_API_URL` | *(unset)* | URL of an external quiz generation service |
| `EXTERNAL_AI_API_KEY` | *(unset)* | Bearer token for external API authentication |

---

## 📡 API Reference

### Base URL

```
http://localhost:5000
```

---

### `GET /health`

Health check — useful for load balancers and uptime monitors.

**Response `200`**
```json
{ "status": "healthy" }
```

---

### `POST /analyze`

Analyse a handwriting image for dyslexia and dysgraphia indicators.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` | ✅ | Handwriting image (PNG / JPG / JPEG / PDF) |
| `text` | `string` | ❌ | Optional pre-extracted OCR text |
| `skip_validation` | `string` | ❌ | Set to `"true"` to bypass image validation |

**Response `200`**
```json
{
  "dyslexia_score": 72.5,
  "dyslexia_details": {
    "confidence": "HIGH",
    "recommendation": "Strong reversal patterns detected. Professional evaluation recommended.",
    "predicted_class": "Dyslexia",
    "class_probabilities": { "Normal": 27.5, "Dyslexia": 72.5 },
    "letters_analyzed": 34
  },
  "dysgraphia_score": 45.2,
  "dysgraphia_details": {
    "confidence": "MODERATE",
    "recommendation": "Some motor coordination challenges noted.",
    "indicators": ["Inconsistent letter height", "Irregular spacing"],
    "primary_concern": "Inconsistent letter height"
  },
  "analysis_type": "Real ML Analysis (Trained CNN - 92.0% accuracy)",
  "features_extracted": {
    "height_variation_cv": 0.42,
    "baseline_deviation": 0.31,
    "spacing_uniformity": 0.55,
    "fragmentation_index": 0.28,
    "stroke_width_variance": 0.19
  },
  "success": true
}
```

**Error `400` — invalid image**
```json
{
  "error": "Invalid image",
  "validation_error": true,
  "reason": "No text lines detected",
  "confidence": 80.0,
  "message": "Please upload a clear image of handwriting on paper."
}
```

---

### `POST /analyze/external`

Identical to `/analyze` but designed for external client integrations. Accepts the same fields and returns the same schema.

---

### `POST /validate`

Pre-validate an image before sending it for full analysis.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` | ✅ | Image to validate |

**Response `200`**
```json
{
  "is_valid": true,
  "confidence": 88.5,
  "reason": "Image validated as test paper with handwriting",
  "details": { "ml_classifier": {}, "rule_validation": {} },
  "message": "Image is valid for analysis"
}
```

---

### `POST /quiz/generate`

Generate a set of quiz questions from a topic or passage of text.

**Request** — `application/json`

```json
{
  "topic": "Grammar",
  "text": "A noun is a word that identifies a person, place, thing, or idea.",
  "question_count": 5
}
```

**Response `200`**
```json
{
  "quiz_id": "a1b2c3d4-...",
  "topic": "Grammar",
  "generated_at": "...",
  "source": "local-fallback",
  "questions": [
    {
      "question": "A ____ is a word that identifies a person, place, thing, or idea.",
      "options": ["noun", "example", "placeholder", "option"],
      "answer": "noun"
    }
  ]
}
```

---

### `POST /quiz/analyze`

Analyse a student's quiz attempt and return personalised learning insights.

**Request** — `application/json`

```json
{
  "studentId": "student123",
  "quizId": "quiz456",
  "topic": "Grammar",
  "totalQuestions": 5,
  "correctAnswers": 3,
  "score": 60.0,
  "totalTimeMs": 180000,
  "questionResponses": [
    {
      "questionId": "q1",
      "questionText": "What is a noun?",
      "correctAnswer": "A",
      "studentAnswer": "B",
      "isCorrect": false,
      "responseTimeMs": 30000
    }
  ]
}
```

**Response `200`**
```json
{
  "performanceLevel": "NEEDS_IMPROVEMENT",
  "learningGapSummary": "Some Grammar concepts need more practice. Scored 60% with 2 incorrect answers.",
  "strongAreas": ["Efficient test-taking skills"],
  "weakAreas": ["Core Grammar concepts need review"],
  "recommendation": "Schedule dedicated study time for Grammar. Review class notes and try practice problems daily.",
  "metrics": {
    "accuracy": 60.0,
    "avgTimePerQuestion": 36.0,
    "fastCorrectAnswers": 1,
    "slowResponses": 2
  }
}
```

---

### Score Interpretation

| Score Range | Level | Meaning |
|---|---|---|
| 0 – 39 | 🟢 Low | Within typical range |
| 40 – 69 | 🟡 Moderate | Some indicators — further assessment suggested |
| 70 – 100 | 🔴 High | Strong indicators — professional evaluation recommended |

| Performance Level | Score Range |
|---|---|
| 🏆 EXCELLENT | ≥ 90% |
| ✅ GOOD | 70 – 89% |
| ⚠️ NEEDS_IMPROVEMENT | 50 – 69% |
| 🆘 STRUGGLING | < 50% |

> ⚠️ **Disclaimer:** NeuraScan AI scores are screening indicators only and do **not** constitute a clinical diagnosis. Always refer to a qualified specialist for a formal assessment.

---

## 🤖 How It Works

### Dyslexia Detection — CNN Model

```
Handwriting Image
       │
       ▼
  Greyscale + Otsu Threshold
       │
       ▼
  Contour Detection (individual letters)
       │
       ▼
  Each letter → resized to 64×64 → CNN prediction
       │
       ▼
  Average probability across all letters
       │
       ▼
  Dyslexia Score (0 – 100)
```

The trained CNN classifies each detected letter as **Normal** or **Dyslexia** based on reversal patterns (e.g., b/d, p/q confusion). Scores are aggregated and reported with a confidence tier.

---

### Dysgraphia Detection — Feature Extraction

Six spatial and morphological features are extracted and weighted:

| Feature | Threshold | Weight |
|---|---|---|
| Letter height variation (CV) | > 0.30 | 30 pts |
| Baseline deviation | > 0.25 | 25 pts |
| Spacing uniformity | > 0.40 | 20 pts |
| Fragmentation index | > 0.20 | 15 pts |

---

### Image Validation — Two-Stage Pipeline

Before any analysis, the upload is screened by:

1. **ML Stage** — PyTorch MobileNetV2 trained on the [RVL-CDIP](https://adamharley.com/rvl-cdip/) dataset classifies the image as a valid paper document.
2. **Rule-Based Stage** — Five heuristic checks (background uniformity, text line structure, colour analysis, document structure, texture complexity) must pass ≥ 3/5.

Both stages must pass for the image to be accepted.

---

## 🐳 Docker Deployment

### Build & Run

```bash
# Build the image
docker build -t neurascan-ai .

# Run the container
docker run -p 5000:5000 neurascan-ai
```

### With environment variables

```bash
docker run -p 5000:5000 \
  -e EXTERNAL_AI_API_URL=https://your-ai-service/analyze \
  -e EXTERNAL_AI_API_KEY=your_api_key \
  neurascan-ai
```

The Dockerfile uses `python:3.9-slim`, installs Tesseract OCR and all system dependencies, and starts Gunicorn with 2 workers and 120-second request timeout.

---

## ☁️ Cloud Deployment

### Render

The repo ships with a `render.yaml` blueprint for one-click deployment:

```yaml
services:
  - type: web
    name: neuroscan-python-ai
    env: docker
    healthCheckPath: /health
```

1. Fork this repo.
2. Connect the fork to [Render](https://render.com).
3. Render will automatically detect `render.yaml` and deploy.

### HuggingFace Spaces

A `Dockerfile.huggingface` and `README.huggingface.md` are included for deployment to [HuggingFace Spaces](https://huggingface.co/spaces) with port 7860.

---

## 🧪 Testing

### Run the included API test

```bash
# Start the server first
python app.py &

# Run the test script (creates a synthetic handwriting image and calls /analyze)
python test_api.py
```

### Manual cURL examples

```bash
# Health check
curl http://localhost:5000/health

# Analyse a handwriting image
curl -X POST http://localhost:5000/analyze \
  -F "file=@/path/to/handwriting.jpg"

# Validate an image without full analysis
curl -X POST http://localhost:5000/validate \
  -F "file=@/path/to/handwriting.jpg"

# Generate a quiz
curl -X POST http://localhost:5000/quiz/generate \
  -H "Content-Type: application/json" \
  -d '{"topic": "Spelling", "text": "...", "question_count": 5}'
```

---

## 📁 Project Structure

```
neurascan-python-ai/
├── app.py                          # Flask application & all routes
├── ml_models.py                    # CNN dyslexia + dysgraphia feature extractor
├── model_pipeline.py               # OCR pipeline, quiz generation, external API
├── paper_validator.py              # Two-stage image validation
├── gunicorn_config.py              # Production server config
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Production container
├── Dockerfile.huggingface          # HuggingFace Spaces container
├── render.yaml                     # Render deployment blueprint
├── models/                         # Trained model files (not in VCS)
│   ├── dyslexia_model.h5 / .keras  # CNN for dyslexia detection
│   ├── document_classifier_pytorch.pth  # PyTorch paper validator
│   └── config.json                 # Model configuration
├── training/                       # Model training notebooks
│   ├── NeuroScan_Training_Colab.ipynb
│   ├── NeuroScan_IAM_Training.ipynb
│   └── NeuroScan_Combined_Training.ipynb
├── test_api.py                     # API smoke test
└── test_sample.png                 # Sample test image
```

---

## 🛠️ Development

### Installing all dependencies

```bash
pip install -r requirements.txt
```

### Key dependencies

| Package | Version | Purpose |
|---|---|---|
| `flask` | 3.0.0 | Web framework |
| `opencv-python-headless` | 4.8.1 | Image processing |
| `tensorflow-cpu` | ≥ 2.13, < 2.17 | CNN inference |
| `torch` / `torchvision` | ≥ 2.0 | Document classifier |
| `pytesseract` | 0.3.10 | OCR text extraction |
| `scikit-learn` | 1.3.2 | Supporting ML utilities |
| `numpy` | 1.26.2 | Numerical computing |
| `Pillow` | 10.1.0 | Image I/O |
| `gunicorn` | 21.2.0 | Production WSGI server |

### Adding your own models

Place trained model files in the `models/` directory:

| File | Used by |
|---|---|
| `models/dyslexia_model.h5` or `.keras` | `ml_models.py` — CNN dyslexia detector |
| `models/document_classifier_pytorch.pth` | `paper_validator.py` — Paper validator |
| `models/document_classifier_traced.pt` | `paper_validator.py` — TorchScript variant |
| `models/config.json` | Both — image size, class names, accuracy |

If no models are present, the system gracefully falls back to rule-based heuristics.

### Training

Jupyter notebooks in `training/` cover the full model training pipeline and are optimised for **Google Colab**:

- `NeuroScan_Training_Colab.ipynb` — Core CNN training
- `NeuroScan_IAM_Training.ipynb` — IAM handwriting dataset training
- `NeuroScan_Combined_Training.ipynb` — Combined dataset pipeline

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/my-feature`
3. **Commit** your changes: `git commit -m "Add my feature"`
4. **Push** to your branch: `git push origin feature/my-feature`
5. **Open** a Pull Request

Please ensure your code follows existing patterns and that any new endpoints include error handling consistent with the rest of the codebase.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

Built with ❤️ to support early learning disorder detection

**[⬆ Back to top](#-neurascan-ai)**

</div>
