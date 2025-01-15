from ultralytics import YOLO

class BoxDetection():
  def __init__(self):
    self.model_path = r"./model/bubble.pt"
    self.model = YOLO(self.model_path)
  
  def predict(self, image=None, *, conf =0.5, iou =0.4 ) -> tuple:
    results = self.model(image, conf=conf, iou=iou)
    detections = results[0].boxes
    output = []
    for box in detections:
      output.append([int(x) for x in box.xyxy[0].tolist()])
    return output
  