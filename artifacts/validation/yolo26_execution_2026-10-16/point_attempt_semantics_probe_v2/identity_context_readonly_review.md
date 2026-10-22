# Independent identity, end and clock review

Read `diagnosis_and_fix_proposal.md` before tracing the current production
structures. This review used source text only: no production imports, media,
memory snapshot, model or GPU operation. Source edits by the other owners can
move these line numbers; references describe the inspected state on 16 October.

## Existing evidence and its limits

No current candidate-path record qualifies a stable match identity to a court
end for service adjudication.

- `src/pipeline/phase6_pipeline.py:241`–259 constructs identity observations from
  the bounding-box foot proxy and always sets `ground_point_usable=False`.
  `candidate_frames` stores interval, identity audit and landmarks at line604.
- `src/tracking/match_identity.py:96`–118 defines reviewed A/B anchors, but has no
  current court-end declaration or calibration revision. `correct()` sets
  ground usability false at159–162. Records at279–285 retain local `track_key`,
  epoch, segment, identity revision, A/B state and anchor ID, not an eligible end.
  Camera/gap resets clear local bindings at224–228. Colour reconnection remains
  provisional and is not a new identity confirmation.
- `src/tracking/court_player_tracker.py:3`–5 explicitly defines segment-local
  near/far roles; line37 assigns role1 from the near half of the ground proxy.
  `src/tracking/player_identity.py:3`–6 describes its whole-segment presence
  prior as offline role selection, not causal A/B identity. Phase6:706–711 takes
  these role1/role2 boxes into the event and shot paths. Numeric event player IDs
  are therefore not an established mapping to stable scoring participants.
- `src/utils/frame_registration_sequence.py:18`–27 carries calibration and
  anchor transform only. `src/court/calibration.py:43`–59 explicitly describes
  supplied-landmark fit consistency and sets independent accuracy false; neither
  structure supplies an identity/end qualification revision.
- `src/court/camera_intervals.py:221`–252 already supplies reviewed normal-play
  and view gates, original PTS/time base, segment, anchor frame, eligibility and
  reasons. A fresh registration can change the anchor at206–211 without a new
  segment, so segment/epoch alone cannot identify a calibration revision.

The inspected private candidate config has empty interval reviews and identity
anchors at `configs/evaluation/yolo26.private-reviewed20-match172-2026-10-16.yaml:125`–128.
It and production retain `authoritative_enabled:false` at85/86. Phase6:663–664
rejects dynamic registration with event authority enabled. Preserve that guard:
the new semantic timeline is not independent event or geometry qualification.

## Constraints sent to the analytics owner

The early API uses typed source-clock/camera/identity context and an optional
explicitly reviewed participant/end provider. Missing provider withholds the
service target; it does not infer one from A/B-only anchors. This is appropriate
with the following required details.

Bind each declaration to analysis/source fingerprint, exact frame/PTS/time base,
current segment, identity epoch/revision and anchor/local track key, and current
calibration artifact/revision or fingerprint plus anchor frame. It must agree
with confirmed current identity evidence or an explicit superseding reviewed
identity revision. A flat eligibility boolean or arbitrary anchor coordinate is
insufficient. Capture the server, official service side and end proof before the
attempt/event mutates rule state. A service bounce must belong to that same
continuous attempt; a new endpoint-only binding after an intervening cut does
not validate the earlier attempt.

Identity authorization is needed for every contact, not just service targets.
`src/scoring/point_outcome_resolver.py:166`–173 derives the hitter from the
`PLAYER_1_HIT`/`PLAYER_2_HIT` token itself. Changing only its `player_id` argument
cannot correctly remap swapped ends. Preserve the raw physical role token and
derive a separate scoring token/participant through eligible current
role→track-key→A/B→configured-participant evidence. Unknown contact association
must withhold/freeze consequential score progression. In particular, passing
`player_id=None` for a serve is insufficient: line100 substitutes the configured
server. Physical observations can remain in diagnostics without scoring them.

For timing, use candidate interval PTS/time-base tokens and exact rational
subtraction before converting to float. Phase6:553 and558–561 converts absolute
source time to float for identity; that field is not an exact elapsed-time
clock at large offsets. Check every transition after the opening frame through
the actual terminal event. The opening frame itself may correctly have no
incoming transition. Cut, zoom, reversed/missing PTS, normal-speed uncertainty,
unknown identity/end or stale calibration makes authoritative service context
unavailable; full physical rally duration stays null. A separately labeled
source-playback duration must not be presented as verified playing time.

## Additional named-box geometry finding

The existing service-box convention also needs a separate correction before a
future qualified target can be trusted. Canonical axes are x-right/y-down in
`src/court/court_geometry.py:6`–7. However,
`src/line_calling/line_geometry.py:98`–101 puts both near and far DEUCE boxes in
the same left x half, while `src/scoring/match_state.py:178`–181 selects the
same named side when the server changes ends. A 180-degree rotation of the
FAR_DEUCE center `(3.4275,8.685)` gives `(7.5425,15.085)`, which the current
polygons label NEAR_AD. This conflicts with the rotation of the server's right
half and diagonal target. The serving-half/diagonal requirement was checked
against [ITF2026 Rule17, PDF page10](https://www.itftennis.com/media/7221/2026-rules-of-tennis-english.pdf).
No geometry change or real line-call accuracy claim was made in this review.
