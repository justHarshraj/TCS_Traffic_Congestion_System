import cv2
import numpy as np

def draw_text(img, text, pos, font_scale=1, font_thickness=2, text_color=(0, 255, 0), text_color_bg=(0, 0, 0)):
    """
    Draws text with a background box for better visibility.
    """
    x, y = pos
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    
    cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y), text_color_bg, -1)
    cv2.putText(img, text, (x, y - 5), font, font_scale, text_color, font_thickness)

def draw_roi(img, roi_points, color=(255, 0, 0), thickness=2):
    """
    Draws the Region of Interest (ROI) polygon on the frame.
    roi_points should be a list of (x, y) tuples or a numpy array of shape (N, 1, 2)
    """
    if isinstance(roi_points, list):
        roi_points = np.array(roi_points, np.int32)
        roi_points = roi_points.reshape((-1, 1, 2))
        
    cv2.polylines(img, [roi_points], isClosed=True, color=color, thickness=thickness)

def is_inside_roi(point, roi_points):
    """
    Checks if a point (x,y) is inside the ROI polygon.
    """
    if isinstance(roi_points, list):
        roi_points = np.array(roi_points, np.int32)
        
    # cv2.pointPolygonTest returns positive if inside, negative if outside, 0 if on edge
    result = cv2.pointPolygonTest(roi_points, point, False) 
    return result >= 0
