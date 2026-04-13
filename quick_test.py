from ultralytics import YOLO
import torch
import cv2
import json
from datetime import datetime
from sort_tracker import *
import numpy as np
import sys

COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
LABELS = ["phone", "glasses", "keys", "earbuds"]

def setup_model(model_path):
    model = YOLO(model_path)
    return model
    
def setup_display(window):
    cv2.namedWindow(window)
    vc = cv2.VideoCapture(0)
    return vc
    

# Source - https://stackoverflow.com/a/606154
# Posted by John Montgomery, modified by community. See post 'Timeline' for change history
# Retrieved 2026-04-10, License - CC BY-SA 4.0
        
def run(model, trackers, vc, window):
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
        print("Could not get first frame")
    while rval:
        cv2.imshow(window, frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        results = model.predict(frame, conf = 0.25, verbose = False)
        
        bboxes = [[] for i in LABELS]
        for result in results:
            
            boxes = result.boxes
            
            for i in range(0, boxes.shape[0]):
                #print(str(i) + "\n" + str(boxes[i]) + "\n" + str(boxes[i].xyxy) + "\n###")
                x1 = float(boxes[i].xyxy[0, 0])
                y1 = float(boxes[i].xyxy[0, 1])
                x2 = float(boxes[i].xyxy[0, 2])
                y2 = float(boxes[i].xyxy[0, 3])
                cls = int(boxes[i].cls)
                conf = float(boxes[i].conf)
            
                bboxes[cls] = trackers[cls].update(np.array([x1, y1, x2, y2, conf], ndmin=2))
                
        for i in range(len(trackers)):
            for box in bboxes[i]:
                frame = cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), COLORS[i], 5)
                frame = cv2.putText(frame, LABELS[i], (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLORS[i], 2)
                
            #make_new_record(frame, boxes[i], record_path, img_path)
            
        if key == 27: # exit on ESC
            break
    vc.release()
    cv2.destroyWindow("preview")

def start(model_path):
    window = "preview"
    model = setup_model(model_path)
    vc = setup_display(window)
    trackers = [Sort() for i in LABELS]
    run(model, trackers, vc, window)
    
def main():
    args = sys.argv[1:]
    
    model_path = []
    record_path = []
    image_path = []
    
    reading = model_path
    i = 0
    
    while i < len(args):
        if args[i] == "-m":
            reading = model_path
        else:
            reading.append(args[i])
        i += 1
        
    if(not len(model_path)):
        print("No model provided")
        exit()
        
        
    if(len(model_path) > 1):
        print("More than 1 model provided")
        exit()
    
    start(model_path[0])
    
if __name__ == '__main__':
    main()