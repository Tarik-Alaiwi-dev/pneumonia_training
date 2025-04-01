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

    # @staticmethod
    # def histogram_equalization(img):
    #     img_np = np.array(img.convert("L"))  # Convert to grayscale
    #     img_equalized = cv2.equalizeHist(img_np)  # Apply histogram equalization
    #     return Image.fromarray(img_equalized)  # Convert back to PIL Image
    
    # @staticmethod
    # def get_transforms(label):
    #     """
    #     Data augmentation balancing strategy:
    #     - Pneumonia (22%) → Strong augmentation (100%)
    #     - Normal (33%) → Moderate augmentation (60%)
    #     - Unsure (44%) → Light augmentation (30%)
    #     """

    #     base_transforms = [
    #         transforms.Lambda(lambda img: VitUtilities.histogram_equalization(img)),  # Apply histogram equalization
    #         transforms.Resize((224, 224)),
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.5], std=[0.5]),
    #         transforms.RandomAutocontrast(),
    #     ]

    #     if label == 0:  # Pneumonia (underrepresented, needs strong augmentation)
    #         aug_transforms = [
    #             transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    #             transforms.RandomAdjustSharpness(sharpness_factor=2),
    #             transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    #             transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.RandomRotation(30), 
    #         ]
    #         prob = 1.0  # Apply augmentation to 100% of samples

    #     elif label == 1:  # Unsure (overrepresented, minimal augmentation)
    #         aug_transforms = [
    #             transforms.RandomAdjustSharpness(sharpness_factor=2),
    #             transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.RandomRotation(30), 
    #         ]
    #         prob = 0.5  # Apply augmentation to 30% of samples

    #     else:  # Normal (medium augmentation)
    #         aug_transforms = [
    #             transforms.RandomAdjustSharpness(sharpness_factor=2),
    #             transforms.RandomHorizontalFlip(p=0.5),
    #             transforms.RandomRotation(30), 
    #         ]
    #         prob = 0.8  # Apply augmentation to 60% of samples

    #     return transforms.Compose([
    #         transforms.RandomApply(aug_transforms, p=prob),
    #         *base_transforms
    #     ])

    @staticmethod
    def get_transforms(label):
        """
        Adjusted augmentation probabilities to balance dataset with:
        - Normal = 60%
        - Pneumonia = 100%
        - Unsure = 10%
        """
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ]

        if label == 0:  # Pneumonia (needs strong augmentation)
            aug_transforms = [
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
            ]
            prob = 1.0  # Apply 100% augmentation
        elif label == 1:  # Unsure (overrepresented, minimal augmentation)
            aug_transforms = [
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
            ]
            prob = 0.1  # Apply augmentations 10% of the time
        else:  # Normal (medium augmentation)
            aug_transforms = [
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
            ]
            prob = 0.6  # Apply augmentations 60% of the time

        return transforms.Compose([
            transforms.RandomApply(aug_transforms, p=prob),
            *base_transforms
        ])

    @staticmethod
    def train_vit(model, train_loader, val_loader, optimizer, criterion, device, epochs):
        model.to(device)

        # Initialize scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=1)

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct, total = 0, 0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for images, labels in progress_bar:
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                # Mixed precision forward pass
                with torch.autocast("cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)

                progress_bar.set_postfix(loss=running_loss / (total // images.shape[0]), acc=100 * correct / total)

            train_acc = 100 * correct / total
            epoch_loss = running_loss / len(train_loader)

            # Validation phase
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            # Lists to store all predictions & true labels
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    with torch.autocast("cuda"):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                    val_total += labels.size(0)

                    # Store for confusion matrix
                    all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

            val_acc = 100 * val_correct / val_total
            val_epoch_loss = val_loss / len(val_loader)

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Step the scheduler
            scheduler.step(val_epoch_loss)

            # Compute Confusion Matrix
            cm = confusion_matrix(all_labels, all_preds)
            plt.figure(figsize=(6,5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pneumonia", "Unsure", "Normal"], 
                        yticklabels=["Pneumonia", "Unsure", "Normal"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.title(f"Confusion Matrix - Epoch {epoch+1}")
            plt.show()

        print("Training Complete!")



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






    # @staticmethod
    # def train_vit(model, train_loader, val_loader, optimizer, criterion, device, epochs):
    #     model.train()
        
    #     for epoch in range(epochs):
    #         running_loss = 0.0
    #         correct, total = 0, 0

    #         # Progress bar for each epoch
    #         progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

    #         for images, labels in progress_bar:
    #             images, labels = images.to(device), labels.to(device)

    #             optimizer.zero_grad()
    #             outputs = model(images)

    #             loss = criterion(outputs, labels)

    #             loss.backward()
    #             optimizer.step()

    #             running_loss += loss.item()
    #             correct += (outputs.argmax(dim=1) == labels).sum().item()
    #             total += labels.size(0)

    #             # Update progress bar
    #             progress_bar.set_postfix(loss=running_loss / (total // images.shape[0]), acc=100 * correct / total)

    #         train_acc = 100 * correct / total
    #         epoch_loss = running_loss / len(train_loader)


    #         model.eval()
    #         val_loss, val_correct, val_total = 0.0, 0, 0

    #         with torch.no_grad():
    #             for images, labels in val_loader:
    #                 images, labels = images.to(device), labels.to(device)

    #                 outputs = model(images)
    #                 loss = criterion(outputs, labels)

    #                 val_loss += loss.item()
    #                 val_correct += (outputs.argmax(dim=1) == labels).sum().item()
    #                 val_total += labels.size(0)

    #         val_acc = 100 * val_correct / val_total
    #         val_epoch_loss = val_loss / len(val_loader)

    #         print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%, "
    #             f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_acc:.2f}%")

    #     print("Training Complete!")

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
        
        # Clean up
        image_tensor.detach_()

# Example usage
# visualize_gradcam(model, "path/to/test_image.jpg")


