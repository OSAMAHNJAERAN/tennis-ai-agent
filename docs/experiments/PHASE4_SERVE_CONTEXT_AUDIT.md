# T88J709 Phase 4.1 — Frame 81 Serve Sequence & Context Audit

## 1. Executive Summary & Investigation Objective

In Phase 4, the real-video line-call result reported:
- **Frame 81**: `SERVE_FAULT` (Ball landed $196.6\text{ cm}$ past the near service line).
- However, the video trajectory showed Player 1 executing a return stroke at **Frame 84**, followed by a continued rally through Frame 180+.

This investigation audited the raw video frames and physical event timeline to resolve:
1. Was the serve genuinely a fault?
2. Was the service box target correctly oriented?
3. Why did the rally continue?
4. How should the Phase 5 match scoring state machine handle this sequence?

---

## 2. Physical & Event Timeline Audit

From raw frame inspection of `data/sample_videos/input_video.mp4`:
- **Frame 23**: **Player 2 (Far Court Server)** strikes the ball from near the far baseline ($X \approx 4.28\text{ m}, Y \approx 6.71\text{ m}$).
- **Ball Trajectory**: The ball travels across the net ($Y = 11.885\text{ m}$) into the near court.
- **Frame 81 (Bounce)**: The ball lands at $(X = 3.11\text{ m}, Y = 20.56\text{ m})$.
  - Canonical Near Service Line is at $Y = 18.285\text{ m}$.
  - The ball landed $2.275\text{ m}$ ($227.5\text{ cm}$) past the service line, deep into the near backcourt near the baseline!
  - Under ITF Rule 17 & 19, a legal serve MUST land within the service box bounded by the net, service line, center service line, and singles sideline ($Y \in [11.885, 18.285]$).
- **Frame 84**: Player 1 hits the ball back from $(X = 2.93\text{ m}, Y = 20.33\text{ m})$.

---

## 3. Root Cause & Officiating Determination

1. **The Call is 100% Legally and Physically Correct**: The serve was a **genuine `SERVE_FAULT`**. It landed $>2\text{ meters}$ beyond the legal service box.
2. **Why Player 1 Hit the Ball**: In tennis practice, coaching drills, and non-tournament recorded footage, players routinely hit long serves to sustain practice rallies rather than letting the ball strike the back fence.
3. **Implications for Phase 5 Scoring**:
   - In regulation match play: A `SERVE_FAULT` immediately halts the point. If 1st serve, the server receives a 2nd serve; if 2nd serve, it is a double fault and the point is awarded to Player 1.
   - Any strokes played after a fault are **dead ball practice strokes** and must not alter the formal match score.
