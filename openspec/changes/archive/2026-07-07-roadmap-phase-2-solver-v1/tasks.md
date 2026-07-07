## 1. Pipeline contracts and module scaffolding

- [x] 1.1 Create solver v1 module layout for detection, matching, pose, overlay, and benchmark capabilities
- [x] 1.2 Define typed payload contracts between pipeline stages (detections, hypotheses, pose fit, debug layers)
- [x] 1.3 Add shared solver result model with confidence, ambiguity flag, accepted/rejected status, and diagnostics

## 2. Star detection baseline

- [x] 2.1 Implement deterministic star-candidate detector for synthetic frames with stable output ordering
- [x] 2.2 Add configurable truth-matching tolerance and per-frame precision/recall computation
- [x] 2.3 Add aggregate detection metric calculation across dataset splits

## 3. Pattern matching baseline

- [x] 3.1 Implement baseline catalog pattern matching over detector outputs
- [x] 3.2 Add confidence scoring for match hypotheses and ranking
- [x] 3.3 Implement ambiguity and low-confidence handling (no-accept path)

## 4. Pose estimation and robustness

- [x] 4.1 Implement camera orientation estimation from matched correspondences
- [x] 4.2 Add robust outlier filtering (RANSAC-like or equivalent) before final pose acceptance
- [x] 4.3 Add explicit failure diagnostics when robust fit cannot produce a stable pose

## 5. Overlay and debug visualization

- [x] 5.1 Implement constellation overlay rendering for accepted pose outputs
- [x] 5.2 Add debug rendering layer for detected star candidates
- [x] 5.3 Add debug rendering layers for inlier/outlier correspondences

## 6. Benchmark harness and reproducibility

- [x] 6.1 Implement benchmark runner for dataset split execution with reproducible configuration handling
- [x] 6.2 Emit aggregate report including detection and pose-quality indicators
- [x] 6.3 Implement worst-case ranking and export of frames with diagnostics
- [x] 6.4 Add reproducibility check procedure to validate stable metrics and ranking within tolerance
