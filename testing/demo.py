import torch
#################################### For Image ####################################
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
# Load the model
model = build_sam3_image_model()
processor = Sam3Processor(model)
# Load an image
image = Image.open("../assets/images/egg tarts.jpg")
inference_state = processor.set_image(image)
# Prompt the model with text
output = processor.set_text_prompt(state=inference_state, prompt="food")

# Get the masks, bounding boxes, and scores
masks, boxes, scores = output["masks"], output["boxes"], output["scores"]

# Print results
print(f"Found {len(masks)} food items")
print(f"Confidence scores: {[f'{s:.2f}' for s in scores.tolist()]}")

# Visualize the results
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Original image
axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Image with masks overlay
axes[1].imshow(image)
for i, mask in enumerate(masks):
    # Create colored mask
    color = np.random.random(3)
    mask_np = mask.cpu().numpy().squeeze()  # Remove extra dimensions
    colored_mask = np.zeros((*mask_np.shape, 4))
    colored_mask[mask_np > 0] = [*color, 0.5]
    axes[1].imshow(colored_mask)
axes[1].set_title(f"Detected Egg Tarts: {len(masks)}")
axes[1].axis("off")

plt.tight_layout()
import os
os.makedirs("output", exist_ok=True)
plt.savefig("output/output.png")
plt.show()
print("Output saved to output/output.png")