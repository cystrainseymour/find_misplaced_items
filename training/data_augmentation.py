import albumentations as A
import cv2
import numpy as np

import sys
import os

def get_files(path, start, end):
    files = os.listdir(path)
    filtered = []
    for file in files:
        num = int(file.split("_")[1].split(".")[0])
        if num >= start and (end == -1 or num < end):
            filtered.append(file)
    return filtered

def get_classes(path):
    inp = open(path, "r")
    classes = []
    line = inp.readline().strip()
    while len(line):
        try:
            classes.append(line)
            line = inp.readline().strip()
        except:
            break
    return classes
    
def read_annotation(path):
    inp = open(path, "r")
    line = inp.readline()
    bboxes = []
    classes = []
    while len(line):
        try:
            nums = line.split()
            classes.append(int(nums[0]))
            bboxes.append(list(map(lambda s: float(s), nums[1:])))
            line = inp.readline()
        except:
            break
    
    return np.array(classes), np.array(bboxes)

def main(image_path, annotation_path, class_path, start = 0, end = -1):
    images = get_files(image_path, start, end)
    
    pairs = {}
    for i in range(len(images)):
        cat, bboxes = read_annotation(os.path.join(annotation_path, os.path.splitext(images[i])[0] + ".txt"))
        pairs[images[i]] = [bboxes, cat]
    
    classes = get_classes(class_path)

    transform = A.Compose([
        A.PadIfNeeded(min_height=640, min_width=640, p=1.0),
        A.AtLeastOneBBoxRandomCrop(width=640, height=640, erosion_factor=0.0, p=1.0),
        A.SquareSymmetry(),
        A.ConstrainedCoarseDropout( 
            num_holes_range=(1, 4),
            hole_height_range=(0.1, 0.3),
            hole_width_range=(0.1, 0.3),
            bbox_labels=list(range(len(classes))),
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
    
    new_images = []
    start_num = max(0, end)
    if end == -1:
        start_num = max(0, start)
    while os.path.exists(os.path.join(image_path, str(start_num).join(["IMG_", ".jpg"]))):
        start_num += 1
    
    for i in range(len(images)):
        cv2_image = cv2.imread(os.path.join(image_path, images[i]))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        new_image = transform(image = cv2_image, bboxes = pairs[images[i]][0], class_labels = pairs[images[i]][1])
        
        new_image_path = os.path.join(image_path, str(start_num + i).join(["IMG_", ".jpg"]))
        new_annotation_path = os.path.join(annotation_path, str(start_num + i).join(["IMG_", ".txt"]))
        
        img = new_image["image"] + -1 * np.min(new_image["image"])
        img *= 255 / np.max(img)
        img = np.asarray(img, dtype="int")        
        
        cv2.imwrite(new_image_path, img)
        annotation_outp = open(new_annotation_path, "a+")
        
        for i in range(len(new_image["bboxes"])):
            annotation_outp.write(" ".join([str(new_image["class_labels"][i])] + [str(j) for j in new_image["bboxes"][i]]) + "\n")
        annotation_outp.close()
    
    
    
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))