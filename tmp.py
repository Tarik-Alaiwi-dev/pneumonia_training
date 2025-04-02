import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def apply_clahe(image, clip_limit=4.0, grid_size=(7,7)):  
    img_np = np.array(image, dtype=np.uint8)  
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    enhanced = clahe.apply(img_np)

    return Image.fromarray(enhanced)

def plot_image(image_path):
    # Open the image using PIL and convert to grayscale
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    print(f"Original image mode: {image.mode}")  # Debugging
    
    # Apply CLAHE
    image = apply_clahe(image)
    
    # Plot the processed image
    plt.figure(figsize=(8, 6))
    plt.imshow(image, cmap='gray')  # Ensure grayscale colormap
    plt.axis('off')  # Hide axes
    plt.show()

# Example usage
image_path = 'r1.png'  # Replace with your image path
plot_image(image_path)
