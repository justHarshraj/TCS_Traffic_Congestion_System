from ultralytics import YOLO

class VehicleTracker:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)

    def track(self, frame, tracker="bytetrack.yaml", persist=True, verbose=False):
        """
        Wraps the YOLOv8 track method.
        Returns the results object from YOLOv8.
        tracker: 'bytetrack.yaml' or 'botsort.yaml'
        """
        # classes=[2, 3, 5, 7] correspond to car, motorcycle, bus, truck in COCO dataset
        results = self.model.track(frame, tracker=tracker, persist=persist, verbose=verbose, classes=[2, 3, 5, 7])
        return results[0]
