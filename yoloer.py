from ultralytics import YOLO

class BoxDetection():
  def __init__(self, model="bubble.pt"):
    self.model_path = f"./model/{model}"
    self.model = YOLO(self.model_path)
  
  def predict(self, image=None, *, conf =0.5, iou =0.4 ) -> tuple:
    results = self.model(image, conf=conf, iou=iou)
    detections = results[0].boxes
    output = []
    for box in detections:
      output.append([int(x) for x in box.xyxy[0].tolist()])
    return output
  
class PanelDetection():
  def __init__(self, model="panel.pt"):
    self.model_path = f"./model/{model}"
    self.model = YOLO(self.model_path)
  
  def predict(self, image=None ):
    results = self.model(image)
    detections = results[0].boxes.xywhn
    output = []
    for i in range(len(detections)):
      output.append(detections[i].tolist())
    return output
  