"""
FastAPI application entry point.
Configures the application, middleware, and routes.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.core.classifier import initialize_classifier
from app.core.segmentation import initialize_segmentation
from app.api.routes import (
    classification_router,
    nutrition_router,
    segmentation_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Handles startup and shutdown events.
    """
    # Startup
    print("🚀 Starting Food Classifier API...")
    
    # Initialize classifier
    classifier_loaded = initialize_classifier()
    if classifier_loaded:
        print("✅ Classifier initialized")
    else:
        print("⚠️ Classifier failed to load - check model paths")
    
    # Initialize segmentation (optional)
    if settings.sam3_enabled:
        seg_loaded = initialize_segmentation()
        if seg_loaded:
            print("✅ SAM3 segmentation initialized")
        else:
            print("⚠️ SAM3 not available - segmentation disabled")
    
    print("🎉 Application started successfully!")
    
    yield
    
    # Shutdown
    print("👋 Shutting down...")


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
    Food Classifier API for React Native and web applications.
    
    ## Features
    - **Classification**: Identify food items from images
    - **Nutrition**: Look up nutritional information
    - **Segmentation**: Detect and segment food items (SAM3)
    
    ## Authentication
    Currently no authentication required.
    """,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS for React Native and web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with API version prefix
API_V1_PREFIX = "/api/v1"

app.include_router(classification_router, prefix=API_V1_PREFIX)
app.include_router(nutrition_router, prefix=API_V1_PREFIX)
app.include_router(segmentation_router, prefix=API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    from app.core.classifier import get_classifier
    from app.core.segmentation import get_segmentation_service
    
    classifier = get_classifier()
    segmentation = get_segmentation_service()
    
    return {
        "status": "healthy",
        "classifier": {
            "loaded": classifier.is_loaded,
            "model": classifier.model_name,
            "classes": classifier.num_classes
        },
        "segmentation": {
            "available": segmentation.is_available
        }
    }


# For running with: python -m app.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
