import onnxruntime as ort
import numpy as np
from PIL import Image
import json

# --- Load model and labels ---
session = ort.InferenceSession("vit_small.onnx")
with open("foodsg233_labels.json", "r") as f:
    labels = json.load(f)

# --- Preprocessing (must match fastai training) ---
def preprocess(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))  # Match training size
    
    # Convert to numpy and normalize (ImageNet stats used by fastai)
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # Reorder: HWC → CHW and add batch dimension
    img_array = img_array.transpose(2, 0, 1)  # [3, 224, 224]
    img_array = np.expand_dims(img_array, axis=0)  # [1, 3, 224, 224]
    
    return img_array.astype(np.float32)

# --- Inference ---
def predict(image_path, top_k=5):
    input_tensor = preprocess(image_path)
    
    # Run inference
    outputs = session.run(None, {"input": input_tensor})
    logits = outputs[0][0]  # Shape: [233]
    
    # Apply softmax to get probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    # Get top-k predictions
    top_indices = np.argsort(probs)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "class": labels[idx],
            "probability": float(probs[idx])
        })
    
    return results

# --- Example usage ---
if __name__ == "__main__":
    results = predict("../assets/images/egg tarts.jpg", top_k=5)
    
    print("🍱 Predictions:")
    for r in results:
        print(f"   {r['class']}: {r['probability']:.4f}")