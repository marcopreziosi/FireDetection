import cv2, os, argparse, random
import numpy as np
import cv2
from ultralytics import YOLO
import os 
import random
import sys
from striprtf.striprtf import rtf_to_text
import glob
import os

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='results/', help="Results folder")
    args = parser.parse_args()
    return args

def send_image(botToken, imageFile, chat_id):
    botToken = str(botToken)
    chat_id = str(chat_id)
    command = 'curl -s -v -X POST https://api.telegram.org/bot' + botToken + '/sendPhoto -F chat_id=' + chat_id + " -F photo=@" + imageFile + " -F caption='Fire Detection!'"
    os.system(command)
    return

args = init_parameter()

model = YOLO('FireNet.pt')
videos = sorted(glob.glob(os.path.join(args.videos, '*')), key=os.path.getsize)
result_dir = args.results
botToken = "" # YOUR TOKEN
chat_id = ""  # YOUR CHAT ID

for video in (videos):
    video_id = video.split("/")[-1].split(".")[0]
    video_path = args.videos+video_id+".mp4"

    f = open(result_dir+video_id+".txt", "w")
    cap = cv2.VideoCapture(video_path)
    i = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1 / fps
    flag_detection = False
    detection_cons = [[None, -1]]*3

    while cap.isOpened() and not flag_detection:
        success, frame = cap.read()
        if success:
            results = model(frame, conf=0.4, verbose=False)
            time_frame = int(i*frame_duration)
            annotated_frame = results[0].plot()

            

            for result in results:
                cls = result.boxes.cls
                if len(cls) > 0:

                    detection_cons.pop()
                    detection_cons.insert(0,[True, time_frame])
                    
                    if all(element[0] is True for element in detection_cons):
                        result = f"{detection_cons[-1][1]}"
                        cv2.imwrite("detection.jpg", annotated_frame)
                        imageFile = "detection.jpg"
                        send_image(botToken, imageFile, chat_id)
                        f.write(str(result))
                        flag_detection = True 
                        detection_cons = [[None, -1]]*3
                else:
                    detection_cons.pop()
                    detection_cons.insert(0,[False, time_frame])        
            i += 1
        else:
            # Break the loop if the end of the video is reached
            break 
    cap.release()
    f.close()


