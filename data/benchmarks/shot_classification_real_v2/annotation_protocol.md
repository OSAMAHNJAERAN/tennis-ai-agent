# Benchmark V2 Annotation Protocol

## 1. Annotation Standards
- Every stroke is manually labeled by examining raw unannotated video frames.
- Contact frame window [frame_min, frame_max] spans the take-back, impact, and initial follow-through (typically 7 frames total).
- True physical contact is defined at frame_hit.
- Hitting player ID, court perspective (NEAR vs FAR), and handedness are verified from raw footage.
- Shot classification (FOREHAND, BACKHAND, SERVE, UNKNOWN) follows standard tennis biomechanics.
- Dead balls and post-fault returns are explicitly tagged is_dead_ball: true.
