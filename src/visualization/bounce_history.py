"""Fixed raster history for a forward render of completed owned event evidence.

Only new source frames are read. The raster preserves the original opaque
circle layering over the current court, people and active zone polygons.
It does not infer or qualify a bounce, and is rebuilt for every analysis.
"""
import math

import cv2
import numpy as np


class MiniCourtBounceHistory:
    def __init__(self, court, source, bounce_type):
        self.court, self.source, self.bounce_type = court, source, bounce_type
        self._geometry = self._geometry_key()
        if (type(court.width) is not int or type(court.height) is not int
                or court.width <= 0 or court.height <= 0
                or not all(math.isfinite(value) for value in self._geometry)):
            raise ValueError('Mini-court history requires finite fixed raster geometry')
        self._generation = source.events_generation
        self._colors = np.zeros((court.height, court.width, 3), dtype=np.uint8)
        self._mask = np.zeros((court.height, court.width), dtype=np.uint8)
        self._through = -1
        self._failure = None

    def _geometry_key(self):
        court = self.court
        return court.width, court.height, court.margin, court.scale_x, court.scale_y

    def draw(self, image, frame):
        if self._failure is not None:
            raise RuntimeError('Failed mini-court history must be rebuilt') from self._failure
        if type(frame) is not int or frame < self._through or frame < 0:
            raise ValueError('Mini-court history requires forward source frames')
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.shape != self._colors.shape or self._geometry_key() != self._geometry):
            raise ValueError('Mini-court history requires its original raster geometry')
        if self.source.events_generation != self._generation:
            raise ValueError('Changed source events require a rebuilt mini-court history')
        iterator = None
        primary = None
        try:
            if frame > self._through:
                iterator = iter(self.source.events_by_frame(self._through + 1, frame, event_type=self.bounce_type))
                for event in iterator:
                    if event.event_type != self.bounce_type or not event.court_position_m or event.frame_index > frame:
                        continue
                    x, y = self.court.court_to_mini_court(*event.court_position_m)
                    cv2.circle(self._colors, (x, y), 5, (0, 165, 255), -1)
                    cv2.circle(self._colors, (x, y), 7, (255, 255, 255), 1)
                    cv2.circle(self._mask, (x, y), 5, 255, -1)
                    cv2.circle(self._mask, (x, y), 7, 255, 1)
            if self.source.events_generation != self._generation:
                raise ValueError('Changed source events require a rebuilt mini-court history')
            cv2.copyTo(self._colors, self._mask, image)
            self._through = frame
            return image
        except BaseException as error:
            self._failure = primary = error
            raise
        finally:
            if iterator is not None:
                try:
                    iterator.close()
                except BaseException as cleanup:
                    self._failure = cleanup
                    if primary is None:
                        raise
                    primary.add_note(f'Mini-court source reader cleanup also failed: {cleanup}')
