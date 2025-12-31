"""
Gradio UI for Food Classifier.
Uses the same core services as the FastAPI backend.
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import pandas as pd
import numpy as np

from app.config import settings
from app.core.classifier import get_classifier, initialize_classifier
from app.core.segmentation import get_segmentation_service, initialize_segmentation
from app.core.nutrition import get_nutrition_service


# Initialize services
print("Initializing services...")
initialize_classifier()
initialize_segmentation()

classifier = get_classifier()
segmentation_service = get_segmentation_service()
nutrition_service = get_nutrition_service()


def predict_single(image, topk: int):
    """Run prediction on a single image."""
    if image is None:
        return None, "Please upload an image."
    
    if not classifier.is_loaded:
        return None, "Model not loaded. Please restart the application."
    
    try:
        predictions, top_prediction = classifier.predict(image, topk=int(topk))
        
        # Format results for display
        results = []
        for pred in predictions:
            results.append({
                "Model": classifier.model_name,
                "Rank": pred["rank"],
                "Prediction": pred["label"],
                "Confidence": f"{pred['confidence']:.4f}"
            })
        
        df = pd.DataFrame(results)
        return df, top_prediction
        
    except Exception as e:
        return None, f"Error: {str(e)}"


def get_food_segmentation(image, prompt: str, confidence_threshold: float):
    """Run SAM3 segmentation on an image."""
    if not segmentation_service.is_available:
        return None, None, "SAM3 model not available."
    
    if image is None:
        return None, None, "Please upload an image."
    
    if not prompt or not prompt.strip():
        return None, None, "Please enter a text prompt."
    
    try:
        result = segmentation_service.segment(
            image=image,
            prompt=prompt.strip(),
            confidence_threshold=confidence_threshold
        )
        
        if result["num_detections"] == 0:
            return np.array(image), None, f"No '{prompt}' items detected."
        
        # Build results dataframe
        df_data = []
        for det in result["detections"]:
            box = det["bounding_box"]
            df_data.append({
                "Item": det["item_id"],
                "Confidence": f"{det['confidence']:.4f}",
                "Box (x1, y1, x2, y2)": f"({box['x1']}, {box['y1']}, {box['x2']}, {box['y2']})"
            })
        
        results_df = pd.DataFrame(df_data) if df_data else None
        
        return result["annotated_image"], results_df, result["message"]
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def search_nutrition(search_term: str):
    """Search for nutrition info."""
    if not search_term:
        return gr.update(choices=[], value=None), "Please enter a search term."
    
    try:
        results = nutrition_service.search(search_term)
        
        if not results:
            return gr.update(choices=[], value=None), "No results found."
        
        choices = [f"{r['index']+1}. {r['name']}" for r in results]
        return gr.update(choices=choices, value=choices[0]), f"Found {len(results)} results"
        
    except Exception as e:
        return gr.update(choices=[], value=None), f"Error: {str(e)}"


def get_nutrition_details(selected_item: str, search_term: str):
    """Get detailed nutrition info for selected item."""
    if not selected_item:
        return None, None, "Please search and select a food item first."
    
    try:
        # Extract index from selection
        idx = int(selected_item.split(".")[0]) - 1
        
        details = nutrition_service.get_nutrition_details(search_term, idx)
        
        if not details:
            return None, None, "Failed to fetch nutrition data."
        
        # Build nutrition per 100g table
        per_100g = details.get("nutrition_per_100g", {})
        if per_100g:
            nutrition_df = pd.DataFrame(
                list(per_100g.items()),
                columns=["Nutrient", "Per 100g"]
            )
        else:
            nutrition_df = None
        
        # Build per serving table if available
        per_serving = details.get("nutrition_per_serving")
        if per_serving:
            serving_df = pd.DataFrame(
                list(per_serving.items()),
                columns=["Nutrient", "Per Serving"]
            )
        else:
            serving_df = None
        
        status = f"**{details['name']}**\n\nServing: {details.get('default_serving_size', 'N/A')}"
        
        return nutrition_df, serving_df, status
        
    except Exception as e:
        return None, None, f"Error: {str(e)}"


def predict_batch(files, topk: int):
    """Run predictions on multiple images."""
    if not classifier.is_loaded:
        return None, "Model not loaded."
    
    if not files:
        return None, "Please upload images."
    
    from PIL import Image
    
    images = []
    for file in files:
        try:
            img = Image.open(file.name).convert("RGB")
            images.append((Path(file.name).name, img))
        except Exception:
            images.append((Path(file.name).name, None))
    
    results = classifier.predict_batch(images, topk=int(topk))
    
    df = pd.DataFrame(results)
    return df, f"Processed {len(files)} images"


def process_batch_and_save(files, topk: int):
    """Process batch and save to CSV."""
    df, status = predict_batch(files, topk)
    
    if df is not None:
        csv_path = "/tmp/batch_predictions.csv"
        df.to_csv(csv_path, index=False)
        return df, status, csv_path
    
    return df, status, None


# Build Gradio Interface
with gr.Blocks(title="Food Classifier") as demo:
    gr.Markdown("# 🍱 Food Classifier")
    
    model_info = f"**Model:** `{classifier.model_name or 'Not loaded'}` | **Classes:** {classifier.num_classes}"
    sam3_info = "✅ Available" if segmentation_service.is_available else "❌ Not available"
    
    gr.Markdown(f"{model_info}")
    gr.Markdown(f"**SAM3 Segmentation:** {sam3_info}")
    
    # Settings
    with gr.Row():
        topk_slider = gr.Slider(1, 10, value=5, step=1, label="Top-k predictions")
    
    # Store search term for nutrition lookup
    search_term_state = gr.State("")
    
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
                    top_prediction = gr.Textbox(label="Top Prediction", interactive=False)
            
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
            
            def search_and_store(term):
                result, status = search_nutrition(term)
                return result, status, term
            
            search_btn.click(
                search_and_store,
                inputs=[search_term_input],
                outputs=[food_choices, search_status, search_term_state]
            )
            
            with gr.Row():
                get_nutrition_btn = gr.Button("✅ Get Nutrition Info", variant="secondary")
            
            with gr.Row():
                with gr.Column():
                    nutrition_per_100g = gr.Dataframe(label="Nutrition per 100g")
                with gr.Column():
                    nutrition_per_serving = gr.Dataframe(label="Estimated per Serving")
            
            nutrition_info_status = gr.Markdown()
            
            get_nutrition_btn.click(
                get_nutrition_details,
                inputs=[food_choices, search_term_state],
                outputs=[nutrition_per_100g, nutrition_per_serving, nutrition_info_status]
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
            
            batch_predict_btn.click(
                process_batch_and_save,
                inputs=[batch_files, topk_slider],
                outputs=[batch_results, batch_status, download_csv]
            )
        
        # Food Segmentation Tab
        with gr.TabItem("🎯 Food Segmentation"):
            gr.Markdown("### SAM3 Prompted Segmentation")
            gr.Markdown("Use text prompts to detect and segment food items in images.")
            
            with gr.Row():
                with gr.Column():
                    seg_image = gr.Image(label="Upload an image", type="pil")
                    seg_prompt = gr.Textbox(
                        label="Text Prompt",
                        placeholder="Enter what to detect (e.g., 'food', 'egg tart')...",
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


def main():
    """Launch the Gradio application."""
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )


if __name__ == "__main__":
    main()
