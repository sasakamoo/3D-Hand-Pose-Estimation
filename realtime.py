# -*- coding: utf-8 -*-
"""
realtime.py -- Live webcam hand pose estimation with sdf_best_model.pt
=======================================================================
Usage:
    python realtime.py
    python realtime.py --model sdf_best_model.pt --camera 0

Controls:
    Q or ESC  -- quit
    S         -- save current frame to realtime_captures/
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from model_sdf import SDFHandPoseNet, IMG_SIZE

# MediaPipe HandLandmarker — BlazePalm detector (~8 MB .task file).
# Runs at 60+ FPS on CPU in VIDEO mode (tracks across frames).
# Used only for bounding-box detection; our SDF model predicts the joint positions.
# Requires hand_landmarker.task in the working directory (auto-downloaded on first run).
try:
    import mediapipe as mp
    from mediapipe.tasks.python        import BaseOptions         as _BaseOptions
    from mediapipe.tasks.python.vision import HandLandmarker      as _HandLandmarker
    from mediapipe.tasks.python.vision import HandLandmarkerOptions as _HandLandmarkerOptions
    from mediapipe.tasks.python.vision import RunningMode          as _RunningMode
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False

HAND_LANDMARKER_MODEL = 'hand_landmarker.task'

# ---------------------------------------------------------------------------
# Skeleton
# ---------------------------------------------------------------------------

CONNECTIONS = [
    [0,1],[1,2],[2,3],[3,4],
    [0,5],[5,6],[6,7],[7,8],
    [0,9],[9,10],[10,11],[11,12],
    [0,13],[13,14],[14,15],[15,16],
    [0,17],[17,18],[18,19],[19,20],
]

# BGR colors per joint (finger groups)
JOINT_BGR = [
    (200, 200, 200),  # 0  wrist
    (0, 230, 0),      # 1-4  index   green
    (0, 230, 0),
    (0, 230, 0),
    (0, 230, 0),
    (255, 128, 0),    # 5-8  middle  blue
    (255, 128, 0),
    (255, 128, 0),
    (255, 128, 0),
    (0, 200, 255),    # 9-12  ring   yellow
    (0, 200, 255),
    (0, 200, 255),
    (0, 200, 255),
    (0, 100, 255),    # 13-16 pinky  orange
    (0, 100, 255),
    (0, 100, 255),
    (0, 100, 255),
    (255, 0, 230),    # 17-20 thumb  magenta
    (255, 0, 230),
    (255, 0, 230),
    (180, 0, 230),
]


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(frame_bgr: np.ndarray,
               hand_box: tuple | None = None) -> tuple[torch.Tensor, tuple]:
    """
    Crop the webcam frame and resize to IMG_SIZE x IMG_SIZE.

    If `hand_box` is provided as (x0, y0, x1, y1) in frame pixel coords
    (e.g. from a hand detector), that region is cropped and padded to a square.
    This matches the FreiHAND training distribution (tight hand crops) and
    fixes the 2D scaling issue where a full-frame crop causes the model to
    predict joints clustered near the frame centre.

    Without a hand_box the fallback is a centre-square crop of the full frame —
    this works when the hand already fills most of the frame.

    Returns:
        tensor     : (1, 3, IMG_SIZE, IMG_SIZE) float32 in [0, 1]
        crop_info  : (x0, y0, side) crop origin and size in frame coords
    """
    h, w = frame_bgr.shape[:2]

    if hand_box is not None:
        bx0, by0, bx1, by1 = hand_box
        bw, bh = bx1 - bx0, by1 - by0
        # Expand box by 20% on each side for context (matches FreiHAND crops)
        pad    = int(max(bw, bh) * 0.20)
        side   = max(bw, bh) + 2 * pad
        cx     = (bx0 + bx1) // 2
        cy     = (by0 + by1) // 2
        x0     = max(0, cx - side // 2)
        y0     = max(0, cy - side // 2)
        # Clamp to frame boundaries
        x0     = min(x0, w - side) if x0 + side <= w else max(0, w - side)
        y0     = min(y0, h - side) if y0 + side <= h else max(0, h - side)
        side   = min(side, min(w, h))   # never exceed frame
    else:
        # Fallback: centre-square crop
        side = min(h, w)
        x0   = (w - side) // 2
        y0   = (h - side) // 2

    crop    = frame_bgr[y0:y0+side, x0:x0+side]
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor  = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor, (x0, y0, side)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_skeleton(canvas: np.ndarray, kpts_px: np.ndarray,
                  scale_x: float, scale_y: float,
                  offset_x: int, offset_y: int) -> np.ndarray:
    """
    Draw the hand skeleton onto `canvas`.
    kpts_px is (21,2) in IMG_SIZE pixel space; we map back to canvas coords.
    """
    def to_canvas(pt):
        x = int(pt[0] * scale_x) + offset_x
        y = int(pt[1] * scale_y) + offset_y
        return x, y

    # Bones
    for s, e in CONNECTIONS:
        p1 = to_canvas(kpts_px[s])
        p2 = to_canvas(kpts_px[e])
        color = JOINT_BGR[s]
        cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

    # Joints
    for k, pt in enumerate(kpts_px):
        c = to_canvas(pt)
        cv2.circle(canvas, c, 4, JOINT_BGR[k], -1, cv2.LINE_AA)
        cv2.circle(canvas, c, 4, (0, 0, 0), 1, cv2.LINE_AA)  # thin black outline

    return canvas


# ---------------------------------------------------------------------------
# MediaPipe bounding box helper
# ---------------------------------------------------------------------------

def mediapipe_hand_box(result, frame_h: int, frame_w: int) -> tuple | None:
    """
    Extract a bounding box (x0, y0, x1, y1) in pixel coords from a
    MediaPipe Tasks HandLandmarkerResult.
    Returns the box for the first detected hand, or None if no hand found.
    """
    if not result.hand_landmarks:
        return None
    lm = result.hand_landmarks[0]
    xs = [l.x * frame_w for l in lm]
    ys = [l.y * frame_h for l in lm]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',    type=str, default='sdf_best_model.pt')
    parser.add_argument('--camera',   type=int, default=0,
                        help='Webcam device index (default 0)')
    parser.add_argument('--no-detect', action='store_true',
                        help='Disable MediaPipe hand detector, use centre-crop fallback')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # Load pose model
    print(f'Loading {args.model} ...')
    model = SDFHandPoseNet(num_kpts=21, pretrained_backbone=False)
    ckpt  = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    print('Model ready.')

    # Warmup
    with torch.no_grad():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))

    # MediaPipe hand detector
    use_detector = HAS_MEDIAPIPE and not args.no_detect
    if use_detector:
        # RunningMode.VIDEO: tracks across frames — faster than re-detecting every frame.
        # Requires monotonically increasing timestamps (ms).
        options = _HandLandmarkerOptions(
            base_options=_BaseOptions(model_asset_path=HAND_LANDMARKER_MODEL),
            running_mode=_RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        mp_hands = _HandLandmarker.create_from_options(options)
        print('MediaPipe hand detector: ON  (--no-detect to disable)')
    else:
        mp_hands = None
        reason = '--no-detect flag set' if args.no_detect else 'mediapipe not installed'
        print(f'MediaPipe hand detector: OFF ({reason}) — using centre-crop fallback')
    print()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')

    save_dir = Path('realtime_captures')
    save_dir.mkdir(exist_ok=True)

    fps_buf  = []
    hand_box = None   # cached bounding box from last detection
    print('Press Q or ESC to quit, S to save a frame.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera read failed.')
            break

        t0 = time.perf_counter()
        h_f, w_f = frame.shape[:2]

        # -- Hand detection (MediaPipe Tasks API, BGR→RGB) --
        if mp_hands is not None:
            rgb_frame  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(time.perf_counter() * 1000)
            result     = mp_hands.detect_for_video(mp_image, timestamp_ms)
            detected   = mediapipe_hand_box(result, h_f, w_f)
            if detected is not None:
                hand_box = detected   # update cache when hand is found

        # -- Crop + preprocess --
        inp, (x0, y0, side) = preprocess(frame, hand_box=hand_box)
        inp = inp.to(device)

        # -- Pose inference --
        with torch.no_grad():
            pose_2d, _, _, _ = model(inp)

        kpts = pose_2d[0].cpu().numpy()   # (21, 2) in IMG_SIZE space

        # -- Map keypoints back to original frame --
        scale_x = side / IMG_SIZE
        scale_y = side / IMG_SIZE

        canvas = frame.copy()
        draw_skeleton(canvas, kpts, scale_x, scale_y, x0, y0)

        # -- Draw crop / detection box --
        box_color = (0, 200, 80) if hand_box is not None else (80, 80, 80)
        cv2.rectangle(canvas, (x0, y0), (x0+side, y0+side), box_color, 1)

        # -- FPS --
        dt = time.perf_counter() - t0
        fps_buf.append(1.0 / dt)
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = np.mean(fps_buf)

        # -- Overlay text --
        det_label = 'BlazePalm+SDF' if use_detector else 'SDF (no detector)'
        cv2.putText(canvas, f'FPS: {fps:.1f}  [{det_label}]', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Q/ESC=quit  S=save', (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Hand Pose (SDF) -- realtime', canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            fname = save_dir / f'capture_{int(time.time())}.png'
            cv2.imwrite(str(fname), canvas)
            print(f'Saved {fname}')

    if mp_hands is not None:
        mp_hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
