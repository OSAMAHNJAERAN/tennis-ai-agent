# Independent Tennis Match Event Annotation Protocol

## 1. Scope and Guiding Principles
This protocol governs the independent, manual ground truth annotation of physical tennis match events directly from raw, unprocessed video frames.

### Fundamental Scientific Rule
**NO MODEL PREDICTIONS OR KALMAN OUTPUTS MAY BE USED AS GROUND TRUTH.**
All annotations (event timestamps, frame intervals, player attributions, and pixel coordinates) must be derived purely by manual visual inspection of raw frames using video player software or frame extraction tools.

---

## 2. Event Taxonomy & Physical Criteria

### 2.1 `SERVE_CONTACT`
- **Definition:** The single frame where the server's tennis racket strings physically strike the airborne tennis ball at the peak of the toss to initiate a point.
- **Visual Landmark:** Point of maximum overhead extension; instant of ball trajectory reversal from upward toss to downward launch.
- **Attributes:**
  - `player_id`: Integer ID of serving player (1 or 2).
  - `frame_best`: Exact contact frame.
  - `frame_min` / `frame_max`: 1-frame uncertainty interval ($\pm 1$ frame).

### 2.2 `BOUNCE`
- **Definition:** The single frame where the tennis ball physically contacts the court surface plane ($Z = 0$) before rebounding upward.
- **Visual Landmark:** Lowest vertical pixel coordinate before upward rebound; compression/shadow contact on the court surface.
- **Attributes:**
  - `player_id`: `null` (Environment interaction).
  - `frame_best`: Ground impact contact frame.
  - `frame_min` / `frame_max`: Temporal contact window.
  - `ball_center_px`: `[x, y]` manually annotated center of the tennis ball at the exact contact frame.

### 2.3 `PLAYER_1_HIT` / `PLAYER_2_HIT`
- **Definition:** The frame where a player's racket strings contact the tennis ball to return it across the net during live rally play.
- **Visual Landmark:** Racket string-bed contact; sharp trajectory inflection within the player's reach zone.
- **Attributes:**
  - `player_id`: 1 (near court player) or 2 (far court player).
  - `frame_best`: Contact frame.
  - `ball_center_px`: `[x, y]` manual pixel location of the ball at contact.

### 2.4 `POINT_END`
- **Definition:** The frame where the rally point officially terminates (e.g. ball double-bounces, lands out of bounds, or receiver fails to return).
- **Visual Landmark:** Second bounce or cessation of competitive movement by players.

---

## 3. Temporal Annotation Uncertainty Standard
At 30 frames per second ($33.3\text{ ms}$ per frame), rapid tennis ball travel ($20\text{--}50\text{ m/s}$) creates motion blur across $1\text{--}2$ frames. To represent annotation uncertainty honestly:
- Every event must record:
  - `frame_best`: The most probable physical frame.
  - `frame_min`: The earliest possible frame consistent with visual evidence.
  - `frame_max`: The latest possible frame consistent with visual evidence.

---

## 4. Manual Ground Truth Coordinate Measurement
- Ball center pixel coordinates $(x_{px}, y_{px})$ are marked manually on raw, uncompressed $1920\times 1080$ video frames.
- Metric court coordinates $(X_m, Y_m)$ are derived using the **Manual Reference Homography** $\mathbf{H}_{ref}$ computed from manually annotated court line intersections.
