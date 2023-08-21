from ultralytics import YOLO

model = YOLO('yolov8n-cls.pt')

model.train(data='/home/dorian/Code/turtles/turtle_datasets/classification',
            pretrained=True,
            epochs=200,
            imgsz=64,
            workers=15,
            cache=True,
            amp=False,
            batch=100,
            lr0=0.03
            )
            
