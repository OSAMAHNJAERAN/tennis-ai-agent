import numpy as np
import cv2

def draw_stats_panel(frame: np.ndarray, stats: dict, position: str = 'bottom_right') -> np.ndarray:
    """
    Draws a semi-transparent stats panel.
    """
    img = frame.copy()
    h, w = img.shape[:2]
    
    panel_w, panel_h = 300, 200
    
    if position == 'bottom_right':
        x1, y1 = w - panel_w - 20, h - panel_h - 20
    elif position == 'top_left':
        x1, y1 = 20, 20
    else:
        x1, y1 = 20, 20
        
    x2, y2 = x1 + panel_w, y1 + panel_h
    
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    
    y_text = y1 + 30
    cv2.putText(img, "MATCH STATS", (x1 + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_text += 25
    
    for key, val in stats.items():
        text = f"{key.replace('_', ' ').title()}: {val}"
        cv2.putText(img, text, (x1 + 10, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_text += 20
        
    return img
