import time

class CongestionDetector:
    def __init__(self, threshold=10, duration=10):
        self.threshold = threshold
        self.duration = duration
        self.elapsed_time = 0
        self.start_time = None
        self.congestion = False
        self.alert_sent = False

    def update(self, vehicle_count):
        if vehicle_count > self.threshold:
            if self.start_time is None:
                self.start_time = time.time()

            self.elapsed_time = time.time() - self.start_time

            if self.elapsed_time > self.duration:
                self.congestion = True
        else:
            self.start_time = None
            self.elapsed_time = 0
            self.congestion = False
            self.alert_sent = False

        return self.congestion
