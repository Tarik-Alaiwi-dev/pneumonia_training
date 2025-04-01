import os
import shutil
import random
import pydicom
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image


class DataProcessor:
    
    @staticmethod
    def reorder_files(df, input_dir, output_dirs):
        '''
        Reorders the files from an input directory to some output directories based on the labels from a CSV file,
        which is given as a DataFrame.
        '''
        for dir_name in output_dirs.values():
            os.makedirs(dir_name, exist_ok=True)

        existing_files = set(os.listdir(input_dir))

        moved_count = 0
        skipped_count = 0
        for _, row in df.iterrows():
            file_name = row["patientId"]
            
            if not file_name.endswith(".png"):
                file_name += ".png"

            label = row["class"]
            dest_dir = output_dirs.get(label)
            
            if dest_dir:
                src_path = os.path.join(input_dir, file_name)
                dest_path = os.path.join(dest_dir, file_name)
                if file_name in existing_files and os.path.exists(src_path):
                    shutil.move(src_path, dest_path)
                    moved_count += 1
                else:
                    skipped_count += 1  

        print(f"Moved {moved_count} files.")
        print(f"Skipped {skipped_count} missing files.")

    @staticmethod
    def split_data(train_dirs, val_dirs):
        """
        Splits dataset into exactly 80% training and 20% validation data.
        """
        for category, train_dir in train_dirs.items():
            val_dir = val_dirs[category]
            os.makedirs(val_dir, exist_ok=True)

            files = [f for f in os.listdir(train_dir) if f.endswith(".png")]
            random.shuffle(files)  # Shuffle files to ensure randomness

            split_idx = int(0.8 * len(files))  # 80% train, 20% validation
            val_files = files[split_idx:]  # Last 20% go to validation

            # Move validation files
            for file_name in val_files:
                shutil.move(os.path.join(train_dir, file_name), os.path.join(val_dir, file_name))

            print(f"Moved {len(val_files)}/{len(files)} files from {train_dir} -> {val_dir}")

    @staticmethod
    def dcm_to_png(directory):
        '''
        Converts ALL .dcm files to .png in the input directory.
        '''
        for file_name in os.listdir(directory):
            if file_name.endswith(".dcm"):
                dcm_path = os.path.join(directory, file_name)
                png_path = os.path.join(directory, file_name.replace(".dcm", ".png"))
                
                DataProcessor.dcm_to_png_single(dcm_path, png_path)

    @staticmethod
    def dcm_to_png_single(dcm_path, png_path):
        try:
            dicom_data = pydicom.dcmread(dcm_path)
            image_array = dicom_data.pixel_array 
            image = Image.fromarray((image_array / image_array.max() * 255).astype(np.uint8))
            
            image.save(png_path)
            os.remove(dcm_path)
            print(f"Converted: {os.path.basename(dcm_path)} -> {os.path.basename(png_path)}")
        except Exception as e:
            print(f"Error converting {dcm_path}: {e}")

    @staticmethod
    def data_distribution_info(dirs):
        file_counts = {} 
        total_files = 0  

        for dir in dirs:
            if os.path.exists(dir) and os.path.isdir(dir):
                num_files = len([f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))])
                file_counts[dir] = num_files
                total_files += num_files
            else:
                file_counts[dir] = 0  

        for dir, count in file_counts.items():
            percentage = (count / total_files) * 100 if total_files > 0 else 0
            print(f"{dir}: {count} files ({percentage:.2f}%)")

        print(f"\nTotal files: {total_files}")

    @staticmethod
    def preprocess_image(image_path):
        """Preprocess image for U-Net input"""
        img = Image.open(image_path).convert("RGB")  # Convert to RGB
        img = img.resize((256, 256))  # Resize to match training input size

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize for RGB
        ])

        img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return img_tensor.to(device), img
    
    @staticmethod
    def image_segmentation(image_path, model):
        input_tensor, original_pil_img = DataProcessor.preprocess_image(image_path)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

        # Convert output tensor to numpy array
        output_mask = output.squeeze().cpu().numpy()
        output_mask = (output_mask > 0.5).astype(np.uint8)  # Apply thresholding

        # Convert to RGBA where white pixels in the mask are transparent
        mask_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        mask_rgba[:, :, 0] = 0  # No Red
        mask_rgba[:, :, 1] = 0  # No Green
        mask_rgba[:, :, 2] = 0  # No Blue
        mask_rgba[:, :, 3] = (1 - output_mask) * 255  # Invert mask: White → Transparent, Black → Opaque

        # Convert original image to RGBA
        original_pil_img = original_pil_img.resize((256, 256)).convert("RGBA")
        # Convert numpy array to PIL image
        mask_pil = Image.fromarray(mask_rgba, "RGBA")
        # Overlay mask on original image
        overlayed_image = Image.alpha_composite(original_pil_img, mask_pil)

        return overlayed_image

    @staticmethod
    def data_segmentation(dirs, model):
        """
        Generates segmentation masks for all images in the given directories,
        deletes the original images, and saves the masks in the same directory.
        """
        for category, dir_path in dirs.items():
            for file_name in os.listdir(dir_path):
                if file_name.endswith(".png"): 
                    img_path = os.path.join(dir_path, file_name)

                    mask = DataProcessor.image_segmentation(img_path, model)

                    if isinstance(mask, torch.Tensor):
                        mask = mask.squeeze().cpu().numpy()  
                        mask = (mask * 255).astype(np.uint8)  
                        mask = Image.fromarray(mask) 

                    elif not isinstance(mask, Image.Image):
                        raise TypeError(f"Unsupported mask type: {type(mask)}")

                    mask.save(img_path)

        print("Data segmentation complete!")


