import json
from pathlib import Path
import gradio as gr
from PIL import Image
import pandas as pd
import numpy as np
import re
import sys
import os

import onnxruntime as ort
from torchvision import transforms
import torch

# Add testing folder to path for web_scrape import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import web_scrape

# Add parent folder to path for sam3 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Paths for pre-loaded model and labels ---
SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "sgfood233_convnext_base.onnx"
DEFAULT_LABELS_PATH = SCRIPT_DIR / "foodsg233_labels.json"

# --- Global state for models ---
models = {}  # {name: (ort_session, class_labels)}
nutrition_cache = {}

# --- SAM3 segmentation model ---
sam3_processor = None
SAM3_AVAILABLE = False

def initialize_sam3():
    """Initialize SAM3 segmentation model."""
    global sam3_processor, SAM3_AVAILABLE
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        
        print("Loading SAM3 model...")
        model = build_sam3_image_model()
        sam3_processor = Sam3Processor(model)
        SAM3_AVAILABLE = True
        print("✅ SAM3 model loaded successfully")
    except Exception as e:
        print(f"⚠️ SAM3 not available: {str(e)}")
        SAM3_AVAILABLE = False

# Preprocessing transform (same as training)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def preprocess_image(image):
    """Preprocess PIL image for ONNX ViT model using torchvision transforms."""
    img = image.convert("RGB")
    input_tensor = transform(img).unsqueeze(0).numpy()
    return input_tensor


def load_class_labels(labels_path=None):
    """Load class labels from JSON file."""
    if labels_path is None:
        labels_path = DEFAULT_LABELS_PATH
    
    with open(labels_path, "r") as f:
        return json.load(f)


def initialize_model():
    """Load the pre-configured ONNX model and labels on startup."""
    global models
    
    try:
        # Load class labels
        labels = load_class_labels(DEFAULT_LABELS_PATH)
        
        # Create ONNX Runtime session with GPU if available
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        session = ort.InferenceSession(str(DEFAULT_MODEL_PATH), providers=providers)
        
        model_name = DEFAULT_MODEL_PATH.stem
        models[model_name] = (session, labels)
        
        print(f"✅ Successfully loaded model: {model_name} with {len(labels)} classes")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False


def get_food_segmentation(image, prompt, confidence_threshold=0.5):
    """
    Run SAM3 segmentation on an image with a text prompt.
    
    Args:
        image: PIL Image
        prompt: Text prompt (e.g., "food", "egg tart", etc.)
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        tuple: (annotated_image, results_df, num_items)
    """
    if not SAM3_AVAILABLE or sam3_processor is None:
        return None, None, "SAM3 model not available. Please check installation."
    
    if image is None:
        return None, None, "Please upload an image."
    
    if not prompt or not prompt.strip():
        return None, None, "Please enter a text prompt."
    
    try:
        # Set confidence threshold
        sam3_processor.confidence_threshold = confidence_threshold
        
        # Run SAM3 inference
        inference_state = sam3_processor.set_image(image)
        output = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt.strip())
        
        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]
        
        if len(masks) == 0:
            return np.array(image), None, f"No '{prompt}' items detected."
        
        # Create annotated image
        img_array = np.array(image).copy()
        
        # Generate colors for each detection
        np.random.seed(42)
        colors = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255)) 
                  for _ in range(len(masks))]
        
        # Draw masks and bounding boxes
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            color = colors[i]
            
            # Apply mask overlay
            mask_np = mask.cpu().numpy().squeeze()
            overlay = img_array.copy()
            overlay[mask_np > 0] = [color[0], color[1], color[2]]
            img_array = np.where(mask_np[:, :, None] > 0, 
                                  (0.6 * img_array + 0.4 * overlay).astype(np.uint8), 
                                  img_array)
            
            # Draw bounding box
            x1, y1, x2, y2 = box.cpu().numpy().astype(int)
            # Draw rectangle (thicker lines)
            thickness = 3
            img_array[y1:y1+thickness, x1:x2] = color  # Top
            img_array[y2-thickness:y2, x1:x2] = color  # Bottom
            img_array[y1:y2, x1:x1+thickness] = color  # Left
            img_array[y1:y2, x2-thickness:x2] = color  # Right
        
        # Build results dataframe
        results = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.cpu().numpy()
            results.append({
                "Item": i + 1,
                "Confidence": f"{float(score):.4f}",
                "Box (x1, y1, x2, y2)": f"({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})"
            })
        
        results_df = pd.DataFrame(results)
        status = f"✅ Found {len(masks)} '{prompt}' item(s)"
        
        return img_array, results_df, status
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def predict_single(image, topk):
    """Run prediction on a single image with the pre-loaded model."""
    if not models:
        return None, "Model not loaded. Please restart the application."
    
    if image is None:
        return None, "Please upload an image."
    
    results = []
    first_prediction = None
    
    # Preprocess image once
    input_tensor = preprocess_image(image)
    
    for model_name, (session, labels) in models.items():
        try:
            # Run inference
            outputs = session.run(None, {"input": input_tensor})
            predictions = outputs[0]
            
            # Get predicted class
            predicted_class = np.argmax(predictions, axis=1)[0]
            
            # Apply softmax to get probabilities
            logits = predictions[0]
            probs = np.exp(logits - np.max(logits))
            probs = probs / probs.sum()
            
            # Get top-k predictions
            topk_idx = np.argsort(probs)[::-1][:int(topk)]
            
            if first_prediction is None:
                first_prediction = labels[topk_idx[0]]
            
            for i, idx in enumerate(topk_idx):
                results.append({
                    "Model": model_name,
                    "Rank": i + 1,
                    "Prediction": labels[idx],
                    "Confidence": f"{float(probs[idx]):.4f}"
                })
        except Exception as e:
            results.append({
                "Model": model_name,
                "Rank": 1,
                "Prediction": f"ERROR: {str(e)}",
                "Confidence": "0.0"
            })
    
    df = pd.DataFrame(results)
    return df, first_prediction


def search_nutrition(search_term):
    """Search for nutrition info and return top 5 results."""
    global nutrition_cache
    
    if not search_term:
        return gr.update(choices=[], value=None), "Please enter a search term."
    
    try:
        results = web_scrape.top5_scrape(search_term=search_term, headless=True)
        nutrition_cache["search_term"] = search_term
        nutrition_cache["results"] = results
        
        if not results:
            return gr.update(choices=[], value=None), "No results found."
        
        choices = [f"{i+1}. {name}" for i, name in enumerate(results)]
        return gr.update(choices=choices, value=choices[0]), f"Found {len(results)} results for '{search_term}'"
    except Exception as e:
        return gr.update(choices=[], value=None), f"Error: {str(e)}"


def get_nutrition_details(selected_item):
    """Get detailed nutrition info for selected item."""
    global nutrition_cache
    
    if not selected_item or "results" not in nutrition_cache:
        return None, "Please search and select a food item first."
    
    try:
        # Extract index from selection
        idx = int(selected_item.split(".")[0]) - 1
        search_term = nutrition_cache["search_term"]
        
        nutrition_data = web_scrape.scrape_nutrition_by_index(
            search_term=search_term,
            chosen_idx=idx,
            headless=True
        )
        
        if not nutrition_data:
            return None, "Failed to fetch nutrition data."
        
        # Build output
        output_text = f"**{nutrition_data['name']}**\n\n"
        output_text += f"*{nutrition_data['description']}*\n\n"
        
        # Nutrition table
        if nutrition_data['nutrition']:
            nutrition_df = pd.DataFrame(
                list(nutrition_data['nutrition'].items()),
                columns=["Nutrient", "Per 100g"]
            )
            
            # Calculate per serving if available
            default_size = nutrition_data["extra_info"].get("Default Serving Size", "")
            output_text += f"**Default Serving Size:** {default_size}\n\n"
            
            serving_grams = re.search(r"(\d+(?:\.\d+)?)\s*g", default_size)
            if serving_grams:
                grams = float(serving_grams.group(1))
                
                # Filter key nutrients
                keep_nutrients = [
                    "Energy (kcal)", "Protein (g)", "Carbohydrate (g)",
                    "Total Fat (g)", "Dietary Fibre (g)"
                ]
                
                estimated = []
                for nutrient, value in nutrition_data['nutrition'].items():
                    if nutrient in keep_nutrients:
                        try:
                            val = float(value) if value.replace('.', '', 1).isdigit() else 0.0
                            est = round(val * grams / 100, 2)
                            estimated.append({"Nutrient": nutrient, "Per Serving": est})
                        except:
                            pass
                
                if estimated:
                    est_df = pd.DataFrame(estimated)
                    return nutrition_df, est_df
            
            return nutrition_df, None
        
        return None, output_text
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def predict_batch(files, topk):
    """Run predictions on multiple images."""
    if not models:
        return None, "Model not loaded. Please restart the application."
    
    if not files:
        return None, "Please upload images."
    
    all_results = []
    
    for file in files:
        try:
            img = Image.open(file.name).convert("RGB")
            input_tensor = preprocess_image(img)
            result = {"File": Path(file.name).name}
            
            for model_name, (session, labels) in models.items():
                # Run inference
                outputs = session.run(None, {"input": input_tensor})
                predictions = outputs[0]
                
                # Get predicted class
                pred_idx = np.argmax(predictions, axis=1)[0]
                
                # Apply softmax for confidence
                logits = predictions[0]
                probs = np.exp(logits - np.max(logits))
                probs = probs / probs.sum()
                
                result[f"{model_name}_prediction"] = labels[pred_idx]
                result[f"{model_name}_confidence"] = f"{float(probs[pred_idx]):.4f}"
            
            all_results.append(result)
        except Exception as e:
            result = {"File": Path(file.name).name}
            for model_name in models:
                result[f"{model_name}_prediction"] = "ERROR"
                result[f"{model_name}_confidence"] = "0.0"
            all_results.append(result)
    
    df = pd.DataFrame(all_results)
    return df, f"Processed {len(files)} images"


# --- Gradio Interface ---
# Initialize models on startup
initialize_model()
initialize_sam3()

with gr.Blocks(title="Food Classifier") as demo:
    gr.Markdown("# 🍱 Food Classifier (ViT Small - FoodSG233)")
    gr.Markdown(f"**Model:** `{DEFAULT_MODEL_PATH.name}` | **Classes:** {len(models.get('vit_small', (None, []))[1])} food categories")
    if SAM3_AVAILABLE:
        gr.Markdown("**SAM3 Segmentation:** ✅ Available")
    else:
        gr.Markdown("**SAM3 Segmentation:** ❌ Not available")
    
    # Settings Section
    with gr.Row():
        topk_slider = gr.Slider(1, 10, value=5, step=1, label="Top-k predictions")
    
    # Tabs for Single and Batch
    with gr.Tabs():
        # Single Image Tab
        with gr.TabItem("Single Image"):
            with gr.Row():
                with gr.Column():
                    single_image = gr.Image(label="Upload an image", type="pil")
                
                with gr.Column():
                    prediction_results = gr.Dataframe(
                        label="Prediction Results",
                        headers=["Model", "Rank", "Prediction", "Confidence"]
                    )
                    top_prediction = gr.Textbox(label="Top Prediction (for nutrition search)", interactive=False)
            
            # Auto-predict when image is uploaded or top-k changes
            single_image.change(
                predict_single,
                inputs=[single_image, topk_slider],
                outputs=[prediction_results, top_prediction]
            )
            topk_slider.change(
                predict_single,
                inputs=[single_image, topk_slider],
                outputs=[prediction_results, top_prediction]
            )
            
            # Nutrition Section
            gr.Markdown("---")
            gr.Markdown("## 🥗 Nutrition Lookup")
            
            with gr.Row():
                with gr.Column():
                    search_term_input = gr.Textbox(label="Search Term", placeholder="Enter food name...")
                    use_prediction_btn = gr.Button("Use Top Prediction")
                    search_btn = gr.Button("🔍 Search Nutrition", variant="primary")
                
                with gr.Column():
                    search_status = gr.Textbox(label="Search Status", interactive=False)
                    food_choices = gr.Radio(label="Select a food item", choices=[])
            
            use_prediction_btn.click(
                lambda x: x,
                inputs=[top_prediction],
                outputs=[search_term_input]
            )
            
            search_btn.click(
                search_nutrition,
                inputs=[search_term_input],
                outputs=[food_choices, search_status]
            )
            
            with gr.Row():
                get_nutrition_btn = gr.Button("✅ Get Nutrition Info", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    nutrition_per_100g = gr.Dataframe(label="Nutrition per 100g")
                with gr.Column():
                    nutrition_per_serving = gr.Dataframe(label="Estimated per Serving")
            
            get_nutrition_btn.click(
                get_nutrition_details,
                inputs=[food_choices],
                outputs=[nutrition_per_100g, nutrition_per_serving]
            )
        
        # Batch Images Tab
        with gr.TabItem("Batch Images"):
            batch_files = gr.File(
                label="Upload multiple images",
                file_types=["image"],
                file_count="multiple"
            )
            batch_predict_btn = gr.Button("🔍 Predict All", variant="primary")
            batch_status = gr.Textbox(label="Status", interactive=False)
            batch_results = gr.Dataframe(label="Batch Results")
            download_csv = gr.File(label="Download CSV")
            
            def process_batch_and_save(files, topk):
                df, status = predict_batch(files, topk)
                if df is not None:
                    csv_path = "/tmp/batch_predictions.csv"
                    df.to_csv(csv_path, index=False)
                    return df, status, csv_path
                return df, status, None
            
            batch_predict_btn.click(
                process_batch_and_save,
                inputs=[batch_files, topk_slider],
                outputs=[batch_results, batch_status, download_csv]
            )
        
        # Food Segmentation Tab (SAM3)
        with gr.TabItem("🎯 Food Segmentation"):
            gr.Markdown("### SAM3 Prompted Segmentation")
            gr.Markdown("Use text prompts to detect and segment food items in images.")
            
            with gr.Row():
                with gr.Column():
                    seg_image = gr.Image(label="Upload an image", type="pil")
                    seg_prompt = gr.Textbox(
                        label="Text Prompt", 
                        placeholder="Enter what to detect (e.g., 'food', 'egg tart', 'rice')...",
                        value="food"
                    )
                    seg_confidence = gr.Slider(
                        0.1, 1.0, value=0.5, step=0.05, 
                        label="Confidence Threshold"
                    )
                    seg_btn = gr.Button("🎯 Segment", variant="primary")
                
                with gr.Column():
                    seg_output_image = gr.Image(label="Segmentation Result")
                    seg_status = gr.Textbox(label="Status", interactive=False)
                    seg_results = gr.Dataframe(
                        label="Detection Results",
                        headers=["Item", "Confidence", "Box (x1, y1, x2, y2)"]
                    )
            
            # Use top prediction as prompt button
            with gr.Row():
                use_classifier_btn = gr.Button("📋 Use Classifier Prediction as Prompt")
            
            use_classifier_btn.click(
                lambda x: x if x else "food",
                inputs=[top_prediction],
                outputs=[seg_prompt]
            )
            
            seg_btn.click(
                get_food_segmentation,
                inputs=[seg_image, seg_prompt, seg_confidence],
                outputs=[seg_output_image, seg_results, seg_status]
            )


if __name__ == "__main__":
    demo.launch()
