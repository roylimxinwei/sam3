to start server type 
```
fastapi dev main.py --host 0.0.0.0 --port 8000
```

# Food Classifier - Modular Architecture

A modularized food classification application supporting both FastAPI (for React Native) and Gradio interfaces.

## Project Structure

```
food_classifier/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration management
│   ├── dependencies.py         # Dependency injection
│   │
│   ├── api/                    # API Layer
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── classification.py   # Food classification endpoints
│   │       ├── nutrition.py        # Nutrition lookup endpoints
│   │       └── segmentation.py     # SAM3 segmentation endpoints
│   │
│   ├── core/                   # Core business logic
│   │   ├── __init__.py
│   │   ├── classifier.py       # Food classification service
│   │   ├── segmentation.py     # SAM3 segmentation service
│   │   └── nutrition.py        # Nutrition scraping service
│   │
│   ├── models/                 # Pydantic models & schemas
│   │   ├── __init__.py
│   │   ├── classification.py   # Classification request/response models
│   │   ├── nutrition.py        # Nutrition request/response models
│   │   └── segmentation.py     # Segmentation request/response models
│   │
│   ├── services/               # External service integrations
│   │   ├── __init__.py
│   │   ├── model_loader.py     # ONNX model loading
│   │   └── web_scraper.py      # Nutrition web scraping wrapper
│   │
│   └── utils/                  # Utility functions
│       ├── __init__.py
│       ├── image_processing.py # Image preprocessing utilities
│       └── exceptions.py       # Custom exceptions
│
├── gradio_ui/                  # Gradio Interface
│   ├── __init__.py
│   └── app.py                  # Gradio application
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_classification.py
│   ├── test_nutrition.py
│   └── test_segmentation.py
│
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
└── README.md                   # This file
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Running FastAPI Server

```bash
# Development
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Running Gradio Interface

```bash
python -m gradio_ui.app
```

### Running Both (Development)

```bash
# Terminal 1: FastAPI
uvicorn app.main:app --reload --port 8000

# Terminal 2: Gradio
python -m gradio_ui.app
```

## API Endpoints

### Classification
- `POST /api/v1/classify` - Classify a single image
- `POST /api/v1/classify/batch` - Classify multiple images

### Nutrition
- `GET /api/v1/nutrition/search?query={food_name}` - Search for nutrition info
- `GET /api/v1/nutrition/{index}?query={food_name}` - Get detailed nutrition

### Segmentation (SAM3)
- `POST /api/v1/segment` - Segment food items in an image

## React Native Integration

See `docs/react_native_integration.md` for detailed integration guide.

## Environment Variables

Copy `.env.example` to `.env` and configure:

```env
MODEL_PATH=./models/sgfood233_convnext_base.onnx
LABELS_PATH=./models/foodsg233_labels.json
SAM3_ENABLED=true
CORS_ORIGINS=["http://localhost:3000","exp://localhost:19000"]
```
