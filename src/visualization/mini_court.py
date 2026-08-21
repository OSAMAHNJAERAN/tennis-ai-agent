import numpy as np
import cv2
from typing import Optional

class MiniCourt:
    """
    Mini-court top-down visualization.
    Converts canonical court meters to mini-court pixels and draws court/players/ball.
    """
    def __init__(self, width: int = 250, height: int = 500, margin: int = 20):
        self.width = width
        self.height = height
        self.margin = margin
        
        # ITF Standard court dimensions in meters
        self.court_length_meters = 23.77
        self.court_width_meters = 10.97
        self.singles_width_meters = 8.23
        self.service_length_meters = 6.40
        
        # Scaling factors
        self.scale_x = (self.width - 2 * self.margin) / self.court_width_meters
        self.scale_y = (self.height - 2 * self.margin) / self.court_length_meters
        
    def draw_court(self) -> np.ndarray:
        """
        Draws the court lines on a blank canvas.
        Returns:
            np.ndarray: Mini-court image with drawn lines.
        """
        canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        
        line_color = (0, 0, 0)
        net_color = (255, 0, 0)
        thickness = 2
        
        # Draw doubles court (outer boundary)
        pts_doubles = [
            self.court_to_mini_court(0, 0),
            self.court_to_mini_court(self.court_width_meters, 0),
            self.court_to_mini_court(self.court_width_meters, self.court_length_meters),
            self.court_to_mini_court(0, self.court_length_meters)
        ]
        cv2.rectangle(canvas, pts_doubles[0], pts_doubles[2], line_color, thickness)
        
        # Draw singles lines
        singles_margin = (self.court_width_meters - self.singles_width_meters) / 2
        cv2.line(canvas, self.court_to_mini_court(singles_margin, 0), 
                 self.court_to_mini_court(singles_margin, self.court_length_meters), line_color, thickness)
        cv2.line(canvas, self.court_to_mini_court(self.court_width_meters - singles_margin, 0), 
                 self.court_to_mini_court(self.court_width_meters - singles_margin, self.court_length_meters), line_color, thickness)
        
        # Draw service lines
        service_y_top = (self.court_length_meters / 2) - self.service_length_meters
        service_y_bottom = (self.court_length_meters / 2) + self.service_length_meters
        cv2.line(canvas, self.court_to_mini_court(singles_margin, service_y_top), 
                 self.court_to_mini_court(self.court_width_meters - singles_margin, service_y_top), line_color, thickness)
        cv2.line(canvas, self.court_to_mini_court(singles_margin, service_y_bottom), 
                 self.court_to_mini_court(self.court_width_meters - singles_margin, service_y_bottom), line_color, thickness)
        
        # Draw center service line
        cv2.line(canvas, self.court_to_mini_court(self.court_width_meters / 2, service_y_top), 
                 self.court_to_mini_court(self.court_width_meters / 2, service_y_bottom), line_color, thickness)
        
        # Draw net
        cv2.line(canvas, self.court_to_mini_court(0, self.court_length_meters / 2), 
                 self.court_to_mini_court(self.court_width_meters, self.court_length_meters / 2), net_color, thickness + 1)
                 
        return canvas

    def draw_positions(self, court_image: np.ndarray, player1_pos: Optional[tuple], player2_pos: Optional[tuple], ball_pos: Optional[tuple]) -> np.ndarray:
        """
        Draws players and ball on the mini-court.
        """
        img = court_image.copy()
        
        if player1_pos:
            p1 = self.court_to_mini_court(player1_pos[0], player1_pos[1])
            cv2.circle(img, p1, 5, (255, 0, 0), -1)  # Blue
            
        if player2_pos:
            p2 = self.court_to_mini_court(player2_pos[0], player2_pos[1])
            cv2.circle(img, p2, 5, (0, 0, 255), -1)  # Red
            
        if ball_pos:
            b = self.court_to_mini_court(ball_pos[0], ball_pos[1])
            cv2.circle(img, b, 3, (0, 255, 255), -1)  # Yellow
            
        return img
        
    def court_to_mini_court(self, court_x: float, court_y: float) -> tuple[int, int]:
        """
        Convert canonical meters to mini-court pixels.
        """
        x_px = int(self.margin + court_x * self.scale_x)
        y_px = int(self.margin + court_y * self.scale_y)
        return (x_px, y_px)
