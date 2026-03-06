from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # lightweight model

def detect_vehicles(frame):
    results = model(frame)

    vehicles = []

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["car", "truck", "bus"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicles.append((x1, y1, x2, y2))

    return vehicles
