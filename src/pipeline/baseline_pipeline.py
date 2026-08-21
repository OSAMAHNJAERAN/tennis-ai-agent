import os
import json
import time
from typing import Dict
import cv2
import yaml
from tqdm import tqdm
import numpy as np

from src.visualization.mini_court import MiniCourt
from src.visualization.video_annotator import VideoAnnotator
from src.visualization.stats_overlay import draw_stats_panel

class BaselinePipeline:
    def __init__(self, config: dict):
        self.config = config
        
        self.mini_court = MiniCourt(
            width=self.config['visualization'].get('mini_court_width', 250),
            height=self.config['visualization'].get('mini_court_height', 500)
        )
        self.annotator = VideoAnnotator()

    def run(self, input_video: str, output_dir: str) -> dict:
        print(f"Running baseline pipeline on {input_video}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Read video + metadata
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_video = None
        if self.config['output'].get('save_annotated_video', True):
            fourcc = cv2.VideoWriter_fourcc(*self.config['output'].get('video_codec', 'mp4v'))
            out_video = cv2.VideoWriter(
                os.path.join(output_dir, "output.mp4"),
                fourcc, fps, (width, height)
            )

        start_time = time.time()
        
        for i in tqdm(range(total_frames), desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Pipeline steps would be here:
            # 2. Detect court keypoints on first frame
            # 3. Compute homography
            # 4. Detect/track players across all frames
            # 5. Detect ball across all frames
            # 6. Filter players to court players
            # 7. Interpolate ball trajectory (gap-limited)
            # 8. Map player positions to court coordinates
            # 9. Map ball positions to court coordinates
            # 10. Calculate player distance/speed
            # 11. Calculate ball trajectory analytics
            # 12. Generate annotated video with mini-court overlay
            
            if out_video:
                mc_img = self.mini_court.draw_court()
                stats = {"fps": f"{fps:.2f}", "frame_number": i}
                frame = self.annotator.compose_frame(frame, mc_img, stats)
                frame = draw_stats_panel(frame, stats)
                out_video.write(frame)
                
        cap.release()
        if out_video:
            out_video.release()
            
        print(f"Pipeline completed in {time.time() - start_time:.2f}s")
        
        # 13. Save JSON results
        results = {"status": "success", "frames_processed": total_frames, "fps": fps}
        if self.config['output'].get('save_json', True):
            with open(os.path.join(output_dir, "results.json"), 'w') as f:
                json.dump(results, f)
                
        # 14. Return summary dict
        return results
