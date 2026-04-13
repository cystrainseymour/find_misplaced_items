from ItemFinder import ItemFinder
from ultralytics import YOLO
import cv2
import traceback

class QueryDisplay:
    def __init__(self, model, c = 0.25, streams = {}, cams = []):
        self.model = YOLO(model)
        self.finders = []
        self.threads = []
        for key, entry in streams.items():
            cap = cv2.VideoCapture(key)
            if cap.isOpened():
                finder = ItemFinder(model = self.model, conf = c, vc = cap, name = str(key), room = str(entry))
                self.finders.append(finder)
            else:
                print(f'stream {str(key)} not opened')
        
        if not len(self.finders):
            for cam in cams:
                cap = cv2.VideoCapture(cam)
                if cap.isOpened():
                    finder = ItemFinder(self.model, c, cap, cam)
                    self.finders.append(finder)
                else:
                    print(f'camera {str(cam)} not opened')
        
        #[ItemFinder(model, image_path, i) for i in range(n_cams)]
        self.labels = self.model.names
        
        try:
            self._start()
        except Exception as e:
            traceback.print_exc()
            
        for finder in self.finders:
            print(finder.shutdown())
        
    def _query(self, cls):
        '''
        Query all itemfinders (one per camera) and return a list of results, including the coordinates of the bounding boxes,
        annotated frames, time stamp, and index of the itemfinder. The list of results is sorted primarily by time (most recent
        first) and secondarily by confidence.
        '''
        
        results = []
        for i in range(len(self.finders)):
            frame, pred, found = self.finders[i].find(cls)
            if found:
                time = pred[1]
                pred = pred[0]
                conf = 0
                try:
                    conf = pred[0, 4]
                except:
                    print(pred)
                    try:
                        conf = pred[4]
                    except:
                        conf = pred[3]
                coords = pred[0, :4]
                results.append({"frame": frame, "coords": coords, "conf": conf, "time": time, "cam_n": i, "cam": self.finders[i].get_name(), "room": self.finders[i].get_room()})
        if len(results):
            results.sort(key = lambda r: r["conf"]) # sort by confidence
            results.sort(key = lambda r: r["time"], reverse = True) # sort by time
        return results
        
    def _display(self, result, cls):
        cv2.namedWindow(cls)
        
        frame = cv2.putText(result["frame"], f'camera {result["cam"]}, {result["time"].strftime("%Y%m%d_%H%M%S")}', (0, 0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
        cv2.imshow(cls, frame)
        print("Displaying frame")
        key = cv2.waitKey(200)
        
    def _start(self):
        while True:
            inp = input("\nWhat do you need to find?\n\t").lower().split()
            
            if "quit" in inp or "exit" in inp or "stop" in inp or "nothing" in inp or "nada" in inp or "done" in inp:
                break
                
            if "options" in inp or "classes" in inp:
                print(self.labels)
            
            items = []
            for idx, label in self.labels.items():
                if label in inp:
                    items.append(idx)
            
            if not len(items):
                print("Item not recognized")
            else:
                for i in items:
                    found = False
                    cur = 0
                    results = self._query(i)
                    if not len(results):
                        print("Sorry, no records of item found")
                    while not found and cur < len(results):
                        result = results[cur]
                        room_str = ""
                        if len(result["room"]): # get the label for the room the camera is in
                            room_str = f' in room "{result["room"]}"'
                             
                        time_str = f' at time {result["time"].strftime("%Y%m%d_%H%M%S")}'
                        print(f'\n\n{self.labels[i].upper()} was last seen by camera {result["cam"]}' + room_str + time_str)
                        self._display(result, self.labels[i])
                        
                        cur += 1
                        
                        inp = input("Has the item been found?\n\t")
                        found = "y" in inp.lower()
                        if not found:
                            self.finders[result["cam_n"]].clear_record(i)
                            
                    cv2.destroyAllWindows()