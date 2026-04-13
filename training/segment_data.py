import os
import random

def segment(images_path = "./images", labels_path = "./labels", frac = 0.2):
    images = os.listdir(os.path.join(images_path, "train"))
    labels = os.listdir(os.path.join(labels_path, "train"))
    
    names = [i.split(".")[0] for i in images]
    val = random.sample(names, int(frac * len(names) + 0.5))
    for name in val:
        os.rename(os.path.join(images_path, "train", name+ ".jpg"), os.path.join(images_path, "val", name+ ".jpg"))
        os.rename(os.path.join(labels_path, "train", name+ ".txt"), os.path.join(labels_path, "val", name+ ".txt"))
        
if __name__ == "__main__":
    segment("./images", "./labels", 0.2)