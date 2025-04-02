import torch
import torchvision.transforms as transforms
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
import seaborn as sns

from PIL import Image

class VitUtilities:

    @staticmethod
    def apply_clahe(image, clip_limit=4.0, grid_size=(7,7)):  
        img_np = np.array(image, dtype=np.uint8)  
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        enhanced = clahe.apply(img_np)

        return Image.fromarray(enhanced)


    @staticmethod
    def get_transforms(label):
        base_transforms = [
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat to 3 channels
            transforms.Normalize([0.5]*3, [0.5]*3),
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]

        if label == 0:  # Pneumonia (Stronger augmentation)
            aug_transforms = [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.2, 0.2), 
                    scale=(0.85, 1.15),  
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0  
                )
            ]
        else:  # Normal (Lighter augmentation)
            aug_transforms = [
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.2, 0.2), 
                    scale=(0.85, 1.15),  
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    fill=0  
                )
            ]

        return transforms.Compose([
            transforms.RandomApply(aug_transforms),
            *base_transforms
        ])

    @staticmethod
    def train_vit(model, train_loader, val_loader, optimizer, criterion, device, epochs):
        model.to(device)

        all_preds = []
        all_labels = []

        for epoch in range(epochs):
                # Training
                model.train()
                train_loss = 0.0
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
                
                for images, labels in progress_bar:
                    images, labels = images.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    train_loss += loss.item()
                    
                    progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze()
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        cm = confusion_matrix(all_labels, all_preds)
        
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'])
        plt.title('Confusion Matrix (Threshold=0.5)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        tn, fp, fn, tp = cm.ravel()
        plt.text(0.5, -0.2, 
                f'Sensitivity: {tp/(tp+fn):.2%} | Specificity: {tn/(tn+fp):.2%}\nAccuracy: {(tp+tn)/(tp+tn+fp+fn):.2%}',
                ha='center', va='center', transform=plt.gca().transAxes)
        
        plt.savefig('confusion_matrix.png', bbox_inches='tight')
        plt.close()
        
        print("Training Complete!")
        print(f"Final Confusion Matrix:\n{cm}")



    def compute_auc(model, dataloader, device):
        model.eval()  # Set model to evaluation mode
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
                
                all_labels.extend(labels.cpu().numpy())  
                all_preds.extend(probs.cpu().numpy())  # Store probabilities

        # Convert lists to numpy arrays
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        # Compute AUC for each class
        auc_scores = {}
        for i, class_name in enumerate(["Pneumonia", "Unsure", "Normal"]):
            auc_scores[class_name] = roc_auc_score((all_labels == i).astype(int), all_preds[:, i])

        return auc_scores



    @staticmethod
    def prediction_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),  # Ensure same input size as training
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Same normalization as training
        ])

    @staticmethod
    def predict_vit(model, image_path, device):
        model.eval()
        image = Image.open(image_path).convert("RGB")
        transform = VitUtilities.prediction_transforms()  # Apply normal class transforms for inference
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image_tensor)
            prediction = torch.argmax(output, dim=1).item()

        class_labels = {0: "Pneumonia", 1: "Unsure", 2: "Normal"}
        return class_labels[prediction]
    
    

    @staticmethod
    def visualize_gradcam(model, image_path, device):
        model.eval()
        
        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        transform = VitUtilities.prediction_transforms()
        image_tensor = transform(image).unsqueeze(0).to(device)
        image_tensor.requires_grad_(True)
        
        # Initialize CAM extractor
        cam_extractor = SmoothGradCAMpp(model)
        
        # Forward pass
        output = model(image_tensor)
        predicted_class = output.squeeze(0).argmax().item()
        
        # Generate activation map
        activation_map = cam_extractor(predicted_class, output)
        
        # Convert activation map to PIL Image
        mask = activation_map[0].squeeze().cpu().numpy()  # Convert to numpy array
        mask = Image.fromarray((mask * 255).astype('uint8'))  # Convert to PIL Image
        
        # Resize mask to match original image size
        mask = mask.resize(image.size, Image.BILINEAR)
        
        # Visualization
        result = overlay_mask(image, mask, alpha=0.5)

        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.imshow(result)
        plt.title(f"Grad-CAM Visualization (Class: {predicted_class})")
        plt.axis("off")

        plt.show()
        
        image_tensor.detach_()




