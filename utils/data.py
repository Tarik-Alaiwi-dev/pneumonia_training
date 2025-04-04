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
        Splits dataset into 80% training and 20% validation data.
        """
        for category, train_dir in train_dirs.items():
            val_dir = val_dirs[category]
            os.makedirs(val_dir, exist_ok=True)

            files = [f for f in os.listdir(train_dir) if f.endswith(".png")]
            random.shuffle(files)  

            split_idx = int(0.8 * len(files))  
            val_files = files[split_idx:]

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
            dir_name = os.path.basename(dir) 
            print(f"{dir_name}: {count} files ({percentage:.2f}%)")

        print(f"\nTotal files: {total_files}")


    



