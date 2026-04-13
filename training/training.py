from ultralytics import YOLO
import albumentations as A
import cv2
import numpy as np
from datetime import datetime
import os
#import torch

data_yaml_path = "./data.yaml"
base_dir = "."
model_save_dir = "../models"

custom_transforms = A.Compose([
        A.PadIfNeeded(min_height=640, min_width=640, p=1.0),
        A.AtLeastOneBBoxRandomCrop(width=640, height=640, erosion_factor=0.0, p=1.0),
        A.SquareSymmetry(),
        A.ConstrainedCoarseDropout( 
            num_holes_range=(1, 4),
            hole_height_range=(0.1, 0.3),
            hole_width_range=(0.1, 0.3),
            bbox_labels=list(range(4)),
            p=0.5, 
            fill="random_uniform"),
        A.OneOf([
            A.ToGray(p=1.0),
            A.ChannelDropout(p=1.0)
        ], p=0.05),
        A.Affine(
            scale=(0.8, 1.2),
            rotate=(-5, 5),
            p=0.25
        ),
        A.Perspective(
            scale=(0.01, 0.5),
            keep_size=True,
            fit_output=False,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.25
        ),
        A.RandomBrightnessContrast(p=0.1),
        A.GaussianBlur(p=0.1),
        A.Normalize(
            mean = (0, 0, 0),
            std = (1, 1, 1),
            normalization="image_per_channel")
    ], bbox_params=A.BboxParams(coord_format='yolo',
                               label_fields=['class_labels']
                               ))

def train_yolo_model(epochs=100, batch_size=8, img_size=640, lr0=0.01):
    # Check for CUDA availability
    #device = '0' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    # Define timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f'train_{timestamp}'
    
    # Load the model
    try:
        model = YOLO('yolo26s.pt')
        model_type = 'yolo26s'
    except Exception:
        model = YOLO('yolov8n.pt')
        model_type = 'yolov8n'
    
    # Train the model
    results = model.train(
        data=data_yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        patience=10,
        save=True,
        device=device,
        project=os.path.join(base_dir, 'runs'),
        name=run_name,
        lr0=lr0,
        lrf=0.01,
        plots=True,
        save_period=5,
        augment=True,
        pretrained=True,
        freeze = list(range(0, 23)),
        augmentations=custom_transforms,
        scale=0.5,
        multi_scale=0.25,
        classes = [0, 1, 2, 3]
    )
    
    # Save the model
    model_save_path = os.path.join(model_save_dir, f"{model_type}_{timestamp}.pt")
    
    try:
        model.model.save(model_save_path)
    except:
        try:
            model.save(model_save_path)
        except Exception:
            best_model_path = os.path.join(base_dir, 'runs', run_name, 'weights', 'best.pt')
            if os.path.exists(best_model_path):
                shutil.copy2(best_model_path, model_save_path)
    
    return model
    
def test_model(model):
    metrics = model.val(
        data=data_yaml_path,
        split='val',
        project=os.path.join(base_dir, 'runs'),
        name='val'
    )
    
    # Calculate F1 score
    f1_score = 2 * metrics.box.precision * metrics.box.recall / (metrics.box.precision + metrics.box.recall + 1e-6)
    
    return metrics
    
def main():
    # Define timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Train model
    model = train_yolo_model(epochs=100, batch_size=8, lr0=0.01)
    
    if model is not None:
        test_model(model)
        
if __name__ == '__main__':
    main()