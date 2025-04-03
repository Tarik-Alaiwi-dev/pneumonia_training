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
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
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
                model.train()
                train_loss = 0.0
                progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
                
                for images, labels in progress_bar:
                    images, labels = images.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images).squeeze()
                    loss = criterion(outputs, labels)
                    loss.backward()
                    
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


    @staticmethod
    def compute_auc(model, dataloader, device):
        model.eval() 
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)  
                
                all_labels.extend(labels.cpu().numpy())  
                all_preds.extend(probs.cpu().numpy())  

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)

        auc_scores = {}
        for i, class_name in enumerate(["Pneumonia", "Unsure", "Normal"]):
            auc_scores[class_name] = roc_auc_score((all_labels == i).astype(int), all_preds[:, i])

        return auc_scores

    @staticmethod
    def predict_vit(model, image_path, device):
        transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1), 
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)), 
                transforms.Normalize([0.5]*3, [0.5]*3),
                transforms.Resize(256),
                transforms.CenterCrop(224)
            ])

        image = Image.open(image_path).convert("L")
        image = VitUtilities.apply_clahe(image)
        image_tensor = transform(image).unsqueeze(0).to(device) 

        model.eval()

        with torch.no_grad():
            output = model(image_tensor).squeeze()
            probability = torch.sigmoid(output).item()  

        predicted_class = "Pneumonia" if probability < 0.5 else "Normal"
        probability = 1 - probability if probability < 0.5 else probability

        return predicted_class, probability

    @staticmethod
    def reshape_transform(tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), 
                                    height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    @staticmethod
    def visualize_gradcam(model, image_path, device):
        model.eval()
        
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  
            transforms.Normalize([0.5]*3, [0.5]*3),
            transforms.Resize(224),
            transforms.CenterCrop(224)
        ])

        original_image = Image.open(image_path).convert("RGB")
        
        image = Image.open(image_path).convert("L")
        image = VitUtilities.apply_clahe(image)
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        target_layer = model.vit.blocks[-1].norm1
        
        cam = GradCAM(model=model,
                    target_layers=[target_layer],
                    reshape_transform=VitUtilities.reshape_transform)
        
        grayscale_cam = cam(input_tensor=image_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]
        
        input_image = image_tensor.squeeze().cpu().detach().numpy()
        input_image = np.transpose(input_image, (1, 2, 0))
        input_image = (input_image * 0.5 + 0.5)  
        
        visualization = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
        inference = VitUtilities.predict_vit(model, image_path, device)
        inference[1]
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"Grad-CAM Visualization - {inference[0]} {round(inference[1] * 100, 2)}%")
        plt.axis("off")

        plt.show()



