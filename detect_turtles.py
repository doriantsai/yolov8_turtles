from ultralytics import YOLO
import os
import glob

# load model
model = YOLO('weights/20230430_yolov8x_turtlesonly_best.pt')

# get inputs as a glob string
img_dir = '/home/dorian/Code/turtles/turtle_datasets/job10_041219-0-1000/split_data/test/images'
img_list = glob.glob(os.path.join(img_dir, '*.PNG'))

pred = model.predict(source = img_list,
                     save=True,
                     save_txt = True,
                     save_conf=True,
                     imgsz=640,
                     conf=0.5)

import code
code.interact(local=dict(globals(), **locals()))