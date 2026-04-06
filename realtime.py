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

def preprocess(frame_bgr: np.ndarray) -> tuple[torch.Tensor, tuple[int,int]]:
    """
    Centre-crop the webcam frame to a square, resize to IMG_SIZE x IMG_SIZE,
    normalise to [0,1], return tensor (1,3,H,W) and the crop box (x0,y0).
    """
    h, w = frame_bgr.shape[:2]
    side  = min(h, w)
    x0    = (w - side) // 2
    y0    = (h - side) // 2
    crop  = frame_bgr[y0:y0+side, x0:x0+side]
    resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    rgb   = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
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
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',  type=str, default='sdf_best_model.pt')
    parser.add_argument('--camera', type=int, default=0,
                        help='Webcam device index (default 0)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    # Load model
    print(f'Loading {args.model} ...')
    model = SDFHandPoseNet(num_kpts=21, pretrained_backbone=False)
    ckpt  = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    print('Model ready.\n')

    # Warmup
    with torch.no_grad():
        model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')

    save_dir = Path('realtime_captures')
    save_dir.mkdir(exist_ok=True)

    fps_buf = []
    print('Press Q or ESC to quit, S to save a frame.')

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera read failed.')
            break

        t0 = time.perf_counter()

        # -- Inference --
        inp, (x0, y0, side) = preprocess(frame)
        inp = inp.to(device)

        with torch.no_grad():
            pose_2d, _, _, _ = model(inp)

        kpts = pose_2d[0].cpu().numpy()  # (21, 2) in IMG_SIZE space

        # -- Map keypoints back to original frame --
        scale_x = side / IMG_SIZE
        scale_y = side / IMG_SIZE

        canvas = frame.copy()
        draw_skeleton(canvas, kpts, scale_x, scale_y, x0, y0)

        # -- FPS --
        dt = time.perf_counter() - t0
        fps_buf.append(1.0 / dt)
        if len(fps_buf) > 30:
            fps_buf.pop(0)
        fps = np.mean(fps_buf)

        # -- Overlay text --
        cv2.putText(canvas, f'FPS: {fps:.1f}', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Q/ESC=quit  S=save', (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Draw crop boundary
        cv2.rectangle(canvas, (x0, y0), (x0+side, y0+side), (80, 80, 80), 1)

        cv2.imshow('Hand Pose (SDF) -- realtime', canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # Q or ESC
            break
        if key == ord('s'):
            fname = save_dir / f'capture_{int(time.time())}.png'
            cv2.imwrite(str(fname), canvas)
            print(f'Saved {fname}')

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
