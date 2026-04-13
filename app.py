from QueryDisplay import QueryDisplay
import json
import sys

def run(model_path, conf = 0.25, streams = {}, cams = []):
    qd = QueryDisplay(model_path, conf, streams, cams)

def main():
    args = sys.argv[1:]
    
    model_path = []
    streams_path = []
    cams = []
    conf_level = []
    
    reading = model_path
    i = 0
    
    while i < len(args):
        if args[i] == "-m":
            reading = model_path
        elif args[i] == "-r" or args[i] == "-s":
            reading = rooms_path
        elif args[i] == "-c":
            reading = cams
        elif args[i] == "-l":
            reading = conf_level
        else:
            reading.append(args[i])
        i += 1
    #print(model_path)
        
    if(not len(model_path)):
        print("No model provided")
        exit()
        
    if(len(model_path) > 1):
        print("More than 1 model provided")
        exit()
        
    if(len(streams_path) > 1):
        print("More than 1 streams path provided")
        exit()
        
        
    conf = 0.25
    if(len(conf_level) > 1):
        print("More than 1 confidence level provided")
        exit()
    elif(len(conf_level)):
        conf = float(conf[0])
    
    cams_new = []
    for cam in cams:
        try:
            cams_new.append(int(cam))
        except:
            cams_new.append(cam)
    
    streams = {}
    if(len(streams_path)):
        with open(streams_path[0], "r", encoding = "utf-8") as streams_inp:
            streams = json.loads(streams_inp.read())
    
    run(model_path[0], conf, streams, cams_new)
    
if __name__ == "__main__":
    main()