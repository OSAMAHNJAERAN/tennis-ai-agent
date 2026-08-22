# Independent Line Calling Annotation Protocol

## 1. Scope & Provenance
This protocol defines the manual ground truth annotation standard for tennis line calls (IN, OUT, SERVE_IN, SERVE_FAULT, REVIEW_REQUIRED) derived directly from raw video frames and verified court geometry.

### Rule of Independence
**NO MODEL PREDICTIONS OR AUTOMATED ENGINE OUTPUTS MAY BE USED AS GROUND TRUTH.**

---

## 2. Event Classification Rules

### 2.1 Serve Contacts
- **`SERVE_IN`**: Serve bounce contact lands completely within or touching the perimeter of the diagonally opposite service box.
- **`SERVE_FAULT`**: Serve bounce contact lands outside the service line, past the center service line, or wide of the singles sideline.

### 2.2 In-Rally Groundstrokes
- **`IN`**: Ball footprint at bounce contact intersects or lies inside the singles boundary lines ($X \in [1.37, 9.60\text{ m}]$, $Y \in [0.0, 23.77\text{ m}]$).
- **`OUT`**: Ball footprint at bounce contact lies completely outside all legal boundary lines with zero line contact.

### 2.3 Abstention & Ambiguity
- **`REVIEW_REQUIRED`**: When the estimated ball margin to the line is smaller than the spatial uncertainty of the camera viewpoint or when the ball is occluded.
