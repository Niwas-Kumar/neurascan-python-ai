---
title: NeuroScan AI
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# NeuroScan AI Service

AI-powered dyslexia and dysgraphia detection from handwriting samples.

## API Endpoints

- `POST /analyze` - Analyze handwriting image
- `POST /analyze/external` - Analyze with validation
- `GET /health` - Health check
- `POST /quiz/generate` - Generate quiz questions

## Usage

Upload a handwriting sample image to get dyslexia/dysgraphia risk scores.
