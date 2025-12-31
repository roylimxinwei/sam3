import gradio as gr
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import sys
import os
import json
import pandas as pd

import onnxruntime as ort
from torchvision import transforms
import torch

# Add parent folder to path for sam3 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Paths for pre-loaded model and labels ---
SCRIPT_DIR = Path(__file__).parent
DEFAULT_MODEL_PATH = SCRIPT_DIR / "sgfood233_convnext_base.onnx"
DEFAULT_LABELS_PATH = SCRIPT_DIR / "foodsg233_labels.json"

sam3_processor = None
SAM3_AVAILABLE = False
models = {}  # Dictionary to store loaded ONNX models

def initialize_sam3():
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

def _draw_overlay(image_pil, masks, boxes, scores, selected=None, highlight_mode=False):
    """
    Draw overlay with masks.
    
    If highlight_mode=True:
      - Draw ALL items, but selected shown with full opacity and checkmark
      - Unselected items shown dimmed (for interactive selection UI)
    
    If highlight_mode=False:
      - ONLY draw selected items (for final confirmed output)
    """
    img_array = np.array(image_pil).copy()
    n = len(masks)
    
    if selected is None:
        selected = list(range(n))
    
    selected_set = set(selected)

    # Consistent colors
    np.random.seed(42)
    colors = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
              for _ in range(n)]

    # Determine which indices to draw
    if highlight_mode:
        # Draw all items (selected will be highlighted, unselected will be dimmed)
        indices_to_draw = range(n)
    else:
        # Only draw selected items
        indices_to_draw = selected

    # Draw masks
    for i in indices_to_draw:
        mask = masks[i]
        box = boxes[i]
        color = colors[i]
        
        is_selected = i in selected_set
        
        # Determine opacity based on selection state
        if highlight_mode:
            mask_opacity = 0.5 if is_selected else 0.15
        else:
            mask_opacity = 0.4

        mask_np = mask.detach().cpu().numpy().squeeze()
        overlay = img_array.copy()
        overlay[mask_np > 0] = [color[0], color[1], color[2]]
        img_array = np.where(
            mask_np[:, :, None] > 0,
            ((1 - mask_opacity) * img_array + mask_opacity * overlay).astype(np.uint8),
            img_array
        )

        # Draw bounding box
        x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
        
        if highlight_mode:
            thickness = 3 if is_selected else 1
            box_color = color if is_selected else tuple(int(c * 0.4) for c in color)
        else:
            thickness = 3
            box_color = color
        
        # Draw box edges
        img_array[max(0,y1):min(img_array.shape[0],y1+thickness), max(0,x1):min(img_array.shape[1],x2)] = box_color
        img_array[max(0,y2-thickness):min(img_array.shape[0],y2), max(0,x1):min(img_array.shape[1],x2)] = box_color
        img_array[max(0,y1):min(img_array.shape[0],y2), max(0,x1):min(img_array.shape[1],x1+thickness)] = box_color
        img_array[max(0,y1):min(img_array.shape[0],y2), max(0,x2-thickness):min(img_array.shape[1],x2)] = box_color

    # Convert to PIL for text drawing
    out = Image.fromarray(img_array)
    draw = ImageDraw.Draw(out)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()

    # Draw labels only for items that were drawn
    for i in indices_to_draw:
        is_selected = i in selected_set
        x1, y1, x2, y2 = boxes[i].detach().cpu().numpy().astype(int)
        color = colors[i]
        
        # Label text
        if highlight_mode:
            label = f"#{i} ✓" if is_selected else f"#{i}"
            # Dim color for unselected
            bg_color = color if is_selected else tuple(int(c * 0.5) for c in color)
        else:
            label = f"#{i}"
            bg_color = color

        tx = max(0, x1)
        ty = max(0, y1 - 24)

        bbox = draw.textbbox((tx, ty), label, font=font)
        pad = 4
        draw.rectangle(
            [bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad],
            fill=bg_color
        )
        text_color = (255, 255, 255) if not is_selected and highlight_mode else (0, 0, 0)
        draw.text((tx, ty), label, fill=text_color, font=font)

    return np.array(out)


def _extract_mask_on_black(image_pil, mask, mask_index):
    """
    Extract a single masked region and display it against a black background.
    Returns the masked region of the original image on black.
    """
    img_array = np.array(image_pil)
    mask_np = mask.detach().cpu().numpy().squeeze()
    
    # Create black background
    result = np.zeros_like(img_array)
    
    # Copy only the masked region from the original image
    result[mask_np > 0] = img_array[mask_np > 0]
    
    return result


def _draw_final_output(image_pil, masks, selected_indices):
    """
    Draw ONLY the selected masks on the image.
    No bounding boxes, no labels - just clean mask overlays.
    """
    img_array = np.array(image_pil).copy()
    n = len(masks)
    
    # Consistent colors (same seed as _draw_overlay for consistency)
    np.random.seed(42)
    colors = [(np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
              for _ in range(n)]
    
    # Only draw selected masks
    for i in selected_indices:
        if i < 0 or i >= n:
            continue
            
        mask = masks[i]
        color = colors[i]
        
        mask_np = mask.detach().cpu().numpy().squeeze()
        overlay = img_array.copy()
        overlay[mask_np > 0] = [color[0], color[1], color[2]]
        
        # Apply mask with 40% opacity
        img_array = np.where(
            mask_np[:, :, None] > 0,
            (0.6 * img_array + 0.4 * overlay).astype(np.uint8),
            img_array
        )
    
    return img_array


def find_clicked_mask_index(x, y, masks):
    """
    Find which mask contains the clicked point.
    Returns the index of the smallest mask containing the point (for overlapping masks).
    """
    candidates = []
    
    for i, mask in enumerate(masks):
        mask_np = mask.detach().cpu().numpy().squeeze()
        h, w = mask_np.shape
        
        # Check bounds
        if 0 <= int(y) < h and 0 <= int(x) < w:
            if mask_np[int(y), int(x)] > 0:
                # Calculate mask area for tie-breaking (prefer smaller/more specific masks)
                area = mask_np.sum()
                candidates.append((i, area))
    
    if not candidates:
        return None
    
    # Return the smallest mask that contains the point
    candidates.sort(key=lambda x: x[1])
    return candidates[0][0]


def get_food_segmentation(image, prompt, confidence_threshold=0.5):
    """Run segmentation and return initial results with all items selected."""
    if not SAM3_AVAILABLE or sam3_processor is None:
        return None, None, "SAM3 model not available.", None, []
    if image is None:
        return None, None, "Please upload an image.", None, []
    if not prompt or not prompt.strip():
        return None, None, "Please enter a text prompt.", None, []

    try:
        sam3_processor.confidence_threshold = confidence_threshold
        inference_state = sam3_processor.set_image(image)
        output = sam3_processor.set_text_prompt(state=inference_state, prompt=prompt.strip())

        masks = output["masks"]
        boxes = output["boxes"]
        scores = output["scores"]

        if len(masks) == 0:
            return np.array(image), [], f"No '{prompt}' items detected.", None, []

        # Initially all items are selected
        all_indices = list(range(len(masks)))
        preview = _draw_overlay(image, masks, boxes, scores, selected=all_indices, highlight_mode=True)

        # Build table rows
        rows = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
            sc = float(score) if not hasattr(score, "item") else float(score.item())
            rows.append([f"#{i}", f"{sc:.4f}", f"({x1}, {y1}, {x2}, {y2})", "✓ Selected"])
            
        cache = {
            "image": image,
            "masks": masks,
            "boxes": boxes,
            "scores": scores,
            "prompt": prompt.strip()
        }

        status = f"✅ Found {len(masks)} '{prompt.strip()}' item(s). Click on items to toggle selection."
        return preview, rows, status, cache, all_indices

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error: {str(e)}", None, []


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


def predict_segmented_items(cache, selected_indices=None, topk=3):
    """
    Run classification on each segmented item.
    
    Args:
        cache: Dictionary containing 'image', 'masks', 'boxes' from segmentation
        selected_indices: Optional list of indices to classify (default: all)
        topk: Number of top predictions to return per item
    
    Returns:
        List of predictions for each item
    """
    if not cache or "masks" not in cache:
        return []
    
    image = cache["image"]  # Original PIL image
    masks = cache["masks"]
    boxes = cache["boxes"]
    
    # If no selection specified, classify all items
    if selected_indices is None:
        selected_indices = list(range(len(masks)))
    
    results = []
    
    for idx in selected_indices:
        if idx < 0 or idx >= len(masks):
            continue
            
        mask = masks[idx]
        box = boxes[idx]
        
        # Get bounding box coordinates
        x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
        
        # Crop the image to the bounding box
        cropped_img = image.crop((x1, y1, x2, y2))
        
        # Apply mask to only show the food item (white background)
        mask_np = mask.detach().cpu().numpy().squeeze()
        cropped_mask = mask_np[y1:y2, x1:x2]
        
        # Create masked crop (food item only, white background)
        img_array = np.array(cropped_img)
        masked_img = np.where(cropped_mask[:, :, None] > 0, img_array, 255)
        cropped_pil = Image.fromarray(masked_img.astype(np.uint8))
        
        # Run classification
        df, top_prediction = predict_single(cropped_pil, topk=topk)
        
        # Get confidence from dataframe
        confidence = "N/A"
        if df is not None and len(df) > 0:
            confidence = df.iloc[0]["Confidence"]
        
        results.append({
            "item_index": idx,
            "prediction": top_prediction if top_prediction else "Unknown",
            "confidence": confidence,
            "details": df
        })
        
        print(f"Item #{idx}: {top_prediction} (conf: {confidence})")
    
    return results


def handle_image_click(cache, selected_indices, evt: gr.SelectData):
    """Handle click on the segmented image to toggle selection."""
    if not cache or "masks" not in cache:
        return None, [], "Run segmentation first.", selected_indices
    
    # Get click coordinates
    x, y = evt.index
    
    masks = cache["masks"]
    boxes = cache["boxes"]
    scores = cache["scores"]
    image = cache["image"]
    
    # Find which mask was clicked
    clicked_idx = find_clicked_mask_index(x, y, masks)
    
    if clicked_idx is not None:
        # Toggle selection
        if clicked_idx in selected_indices:
            selected_indices = [i for i in selected_indices if i != clicked_idx]
            action = f"Deselected item #{clicked_idx}"
        else:
            selected_indices = selected_indices + [clicked_idx]
            action = f"Selected item #{clicked_idx}"
    else:
        action = "No item at click location"
    
    # Re-render with updated selection
    preview = _draw_overlay(image, masks, boxes, scores, selected=selected_indices, highlight_mode=True)
    
    # Update table
    rows = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
        sc = float(score) if not hasattr(score, "item") else float(score.item())
        status_text = "✓ Selected" if i in selected_indices else "○ Unselected"
        rows.append([f"#{i}", f"{sc:.4f}", f"({x1}, {y1}, {x2}, {y2})", status_text])
    
    status = f"{action}. {len(selected_indices)}/{len(masks)} items selected."
    
    return preview, rows, status, selected_indices


def select_all(cache, selected_indices):
    """Select all detected items."""
    if not cache or "masks" not in cache:
        return None, [], "Run segmentation first.", selected_indices
    
    masks = cache["masks"]
    boxes = cache["boxes"]
    scores = cache["scores"]
    image = cache["image"]
    
    selected_indices = list(range(len(masks)))
    preview = _draw_overlay(image, masks, boxes, scores, selected=selected_indices, highlight_mode=True)
    
    rows = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
        sc = float(score) if not hasattr(score, "item") else float(score.item())
        rows.append([f"#{i}", f"{sc:.4f}", f"({x1}, {y1}, {x2}, {y2})", "✓ Selected"])
    
    return preview, rows, f"Selected all {len(masks)} items.", selected_indices


def deselect_all(cache, selected_indices):
    """Deselect all items."""
    if not cache or "masks" not in cache:
        return None, [], "Run segmentation first.", []
    
    masks = cache["masks"]
    boxes = cache["boxes"]
    scores = cache["scores"]
    image = cache["image"]
    
    selected_indices = []
    preview = _draw_overlay(image, masks, boxes, scores, selected=selected_indices, highlight_mode=True)
    
    rows = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = box.detach().cpu().numpy().astype(int)
        sc = float(score) if not hasattr(score, "item") else float(score.item())
        rows.append([f"#{i}", f"{sc:.4f}", f"({x1}, {y1}, {x2}, {y2})", "○ Unselected"])
    
    return preview, rows, "Deselected all items. Click on items to select.", selected_indices


def confirm_selection(cache, selected_indices):
    """Confirm the current selection, classify each item, and produce individual mask outputs."""
    if not cache:
        return [], "Run segmentation first.", []

    image = cache["image"]
    masks = cache["masks"]
    prompt = cache.get("prompt", "item")
    
    print(f"DEBUG confirm_selection: selected_indices = {selected_indices}, total masks = {len(masks)}")

    if not selected_indices:
        return [], "No items selected. Click on items to select them first.", []

    # Run classification on each selected item
    predictions = predict_segmented_items(cache, sorted(selected_indices), topk=3)
    
    # Build prediction lookup
    pred_lookup = {p["item_index"]: p for p in predictions}

    # Generate individual mask images on black background with predictions
    individual_images = []
    prediction_rows = []
    
    for idx in sorted(selected_indices):
        if 0 <= idx < len(masks):
            mask_img = _extract_mask_on_black(image, masks[idx], idx)
            
            # Get prediction for this item
            pred = pred_lookup.get(idx, {})
            pred_label = pred.get("prediction", "Unknown")
            pred_conf = pred.get("confidence", "N/A")
            
            # Gallery caption with prediction
            caption = f"#{idx}: {pred_label} ({pred_conf})"
            individual_images.append((mask_img, caption))
            
            # Add to prediction table
            prediction_rows.append([f"#{idx}", pred_label, pred_conf])
    
    status = f"✅ Classified {len(selected_indices)} '{prompt}' item(s)"
    return individual_images, status, prediction_rows


# --- Gradio Interface ---
initialize_sam3()
initialize_model()

with gr.Blocks(title="SAM3 Click-to-Select Segmentation") as segmentation_demo:
    gr.Markdown("## 🎯 SAM3 Prompted Segmentation")
    gr.Markdown("**Click directly on segmented items to toggle their selection!**")

    det_cache = gr.State(value=None)
    selected_state = gr.State(value=[])

    with gr.Row():
        with gr.Column(scale=1):
            seg_image = gr.Image(label="Upload an image", type="pil")
            seg_prompt = gr.Textbox(
                label="Text Prompt",
                placeholder="e.g., 'food', 'egg tart', 'rice'...",
                value="food"
            )
            seg_confidence = gr.Slider(
                0.1, 1.0, value=0.5, step=0.05, 
                label="Confidence Threshold"
            )
            seg_btn = gr.Button("🎯 Run Segmentation", variant="primary", size="lg")

            gr.Markdown("---")
            gr.Markdown("### Selection Controls")
            with gr.Row():
                select_all_btn = gr.Button("✓ Select All", size="sm")
                deselect_all_btn = gr.Button("✗ Clear All", size="sm")
            
            confirm_btn = gr.Button("✅ Confirm Selection", variant="primary", size="lg")

        with gr.Column(scale=1):
            seg_output_image = gr.Image(
                label="📍 Click on items to toggle selection", 
                interactive=True,
            )
            seg_status = gr.Textbox(label="Status", interactive=False)
            
            seg_results = gr.Dataframe(
                label="Detection Results",
                headers=["Item", "Confidence", "Bounding Box", "Status"],
                interactive=False
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### ✅ Individual Mask Results")
            gr.Markdown("*Each selected item displayed against a black background with classification*")
            final_output = gr.Gallery(
                label="Individual Masks with Predictions",
                columns=3,
                rows=2,
                height="auto",
                object_fit="contain"
            )
            gr.Markdown("### 🍽️ Food Classification Results")
            prediction_results = gr.Dataframe(
                label="Predictions",
                headers=["Item", "Prediction", "Confidence"],
                interactive=False
            )
            confirm_status = gr.Textbox(label="Confirmation", interactive=False)
            
            

    # Event handlers
    seg_btn.click(
        get_food_segmentation,
        inputs=[seg_image, seg_prompt, seg_confidence],
        outputs=[seg_output_image, seg_results, seg_status, det_cache, selected_state]
    )

    seg_output_image.select(
        handle_image_click,
        inputs=[det_cache, selected_state],
        outputs=[seg_output_image, seg_results, seg_status, selected_state]
    )

    select_all_btn.click(
        select_all,
        inputs=[det_cache, selected_state],
        outputs=[seg_output_image, seg_results, seg_status, selected_state]
    )
    
    deselect_all_btn.click(
        deselect_all,
        inputs=[det_cache, selected_state],
        outputs=[seg_output_image, seg_results, seg_status, selected_state]
    )

    confirm_btn.click(
        confirm_selection,
        inputs=[det_cache, selected_state],
        outputs=[final_output, confirm_status, prediction_results]
    )

if __name__ == "__main__":
    segmentation_demo.launch()