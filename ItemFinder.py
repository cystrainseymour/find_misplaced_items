#from ultralytics import YOLO
import datetime
from sort_tracker import *
import numpy as np
import cv2
import threading

# for evaluation
import psutil
import os

class ItemFinder:
    def __init__(self, model, conf = 0.25, vc = cv2.VideoCapture(0), name = "", room = ""):
        COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255), (0, 0, 0)]
        
        self.model = model
        self.conf = conf
        self.labels = model.names
        self.trackers = [Sort() for i in self.labels]
        self.bboxes = [[] for i in self.labels]
        self.frames = [{} for i in self.labels]
        self.vc = vc
        if not self.vc.isOpened():
            raise ValueError
        
        self.colors = []
        for i in range(len(self.model.names)):
            self.colors.append(COLORS[i % len(COLORS)])
        
        self.name = name
        self.room = room
        
        self.monitoring = True
        
        self.thread = threading.Thread(target = self._monitor)
        self.thread.start()
        
        self.ave_speed = 0 # for evaluation
        self.n_preds = 0
        self.max_mem = 0
        self.max_cpu = process.cpu_percent()

    '''def _make_new_record(self, frame, prediction):
        recs = os.listdir(self.record_path)[prediction.cls]
        imgs = os.listdir(self.image_path)[prediction.cls]
        
        rec_outp = open(recs, "a", encoding = "utf-8")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_outp.write(" ".join(prediction.data))
        rec_outp.write("\n")
        rec_outp.close()
        
    def _lookup(self, cls):
        recs = os.listdir(self.record_path)[prediction.cls]
        imgs = os.listdir(self.image_path)[prediction.cls]
        
        rec_inp = open(recs, "r", encoding = "utf-8")
        pred = np.array([float(i) for i in rec_inp.readline().split()])'''
        
    def shutdown(self):
        self.monitoring = False
        self.thread.join()
        
        self.vc.release()
        return self.ave_speed, self.n_preds, self.max_mem, self.max_cpu
        
    def get_labels(self):
        return self.model.names
        
    def get_name(self):
        return self.name
        
    def get_room(self):
        return self.room
        
    def _monitor(self):
        process = psutil.Process(os.getpid())
        while self.monitoring:
            rval, frame = self.vc.read()
            
            t1 = datetime.datetime.now()
            results = self.model.predict(frame, conf = self.conf, verbose = False)
            t2 = datetime.datetime.now()
            
            # to evaluate memory and cpu
            self.max_mem = max(self.max_mem, process.memory_info().rss)
            self.max_cpu = max(self.max_mem, process.cpu_percent())
            
            # to evaluate speed
            self.n_preds += 1
            self.ave_speed += (t2 - t1)/self.n_preds
            
            
            
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
                    
                    self.bboxes[cls] = [self.trackers[cls].update(np.array([x1, y1, x2, y2, conf], ndmin=2)), datetime.datetime.now()]
                    save_frame = cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), self.colors[cls], 5)
                    save_frame = cv2.putText(save_frame, f'{self.labels[cls]} conf: {str(round(conf * 100, 2))}%', (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors[cls], 2)
                    self.frames[cls] = save_frame

    def find(self, cls):
        '''rval, frame = self.vc.read()
        if not rval:
            return [], [], False, False
        
        results = self.model.predict(frame, conf = self.conf)
        
        ret = []
        old_boxes = []
        if len(self.bboxes[cls]):
            old_boxes = self.boxes[cls]
        
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
                
                self.bboxes[cls] = [trackers[cls].update(np.array([x1, y1, x2, y2, conf], ndmin=2)), datetime.datetime.now()]
                    
        for box in self.bboxes[cls]:
            x1 = int(box[0])
            y1 = int(box[1])
            x2 = int(box[2])
            y2 = int(box[3])
            conf = box[4]
            
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), self.colors[cls], 5)
            frame = cv2.putText(frame, self.labels[cls], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors[cls], 2)
        
        if self.bboxes[cls]:
            self.frames[cls].clear()
            self.frames[cls][np.array([x1, y1, x2, y2, conf])] = frame'''

        self.monitoring = False
        self.thread.join()
        ret = self.bboxes[cls]
        frames = self.frames[cls]
        found = len(ret) >= 1
        
        '''if(not len(ret)):
            #ret = _lookup(cls)
            ret = old_boxes
            frames = [self.frames[cls][box[0]] for box in old_boxes]
            cur = False'''
            
        #else:
        #   _ make_new_record(frame, ret, record_path, img_path)
        self.monitoring = True
        
        self.thread = threading.Thread(target = self._monitor)
        self.thread.start()
        return frames, ret, found

    def clear_record(self, cls):
        self.bboxes[cls] = []
        self.frames[cls] = {}