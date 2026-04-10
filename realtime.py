# -*- coding: utf-8 -*-
"""
realtime.py -- Live webcam hand pose estimation with sdf_best_model.pt
=======================================================================
Usage:
    python realtime.py
    python realtime.py --model sdf_best_model.pt --camera 0
    python realtime.py --show-3d --K-file webcam_K.json

Controls:
    Q or ESC  -- quit
    S         -- save current frame to realtime_captures/

K file format (webcam_K.json):
    {"fx": 614.3, "fy": 614.3, "cx": 320.0, "cy": 240.0}
    or a 3x3 nested list:  [[fx,0,cx],[0,fy,cy],[0,0,1]]
    .npy files (3x3 float array) and whitespace-delimited .txt are also accepted.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model_sdf    import SDFHandPoseNet, IMG_SIZE, N_PTS
from model        import SingleViewModel
from model_hybrid import HybridHandPoseNet

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

# RGB colors per joint (finger groups) — used for both 2D and 3D drawing
JOINT_RGB = [
    (0.78, 0.78, 0.78),  # 0  wrist
    (0.00, 0.90, 0.00),  # 1-4  index   green
    (0.00, 0.90, 0.00),
    (0.00, 0.90, 0.00),
    (0.00, 0.90, 0.00),
    (1.00, 0.50, 0.00),  # 5-8  middle  orange
    (1.00, 0.50, 0.00),
    (1.00, 0.50, 0.00),
    (1.00, 0.50, 0.00),
    (0.00, 0.78, 1.00),  # 9-12  ring   cyan
    (0.00, 0.78, 1.00),
    (0.00, 0.78, 1.00),
    (0.00, 0.78, 1.00),
    (0.00, 0.39, 1.00),  # 13-16 pinky  blue
    (0.00, 0.39, 1.00),
    (0.00, 0.39, 1.00),
    (0.00, 0.39, 1.00),
    (1.00, 0.00, 0.90),  # 17-20 thumb  magenta
    (1.00, 0.00, 0.90),
    (1.00, 0.00, 0.90),
    (0.71, 0.00, 0.90),
]

# BGR equivalents for OpenCV 2D drawing
JOINT_BGR = [(int(r*255), int(g*255), int(b*255))
             for r, g, b in [(c[2], c[1], c[0]) for c in JOINT_RGB]]


# ---------------------------------------------------------------------------
# Camera K helpers
# ---------------------------------------------------------------------------

def load_K(path: str) -> np.ndarray:
    """
    Load a 3x3 camera intrinsics matrix from:
      - .npy  : np.load
      - .json : {"fx","fy","cx","cy"} dict  OR  3x3 nested list
      - .txt  : whitespace-delimited 3x3
    Returns K as float64 (3,3).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'K file not found: {path}')
    suffix = p.suffix.lower()
    if suffix == '.npy':
        K = np.load(str(p)).astype(np.float64)
    elif suffix == '.json':
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict):
            fx = data['fx']; fy = data['fy']
            cx = data['cx']; cy = data['cy']
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        else:
            K = np.array(data, dtype=np.float64)
    else:  # .txt or other
        K = np.loadtxt(str(p), dtype=np.float64)
    assert K.shape == (3, 3), f'Expected 3x3 K, got {K.shape}'
    return K


def adjust_K_for_crop(K: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """
    Adjust intrinsics for a square crop starting at (x0, y0) with size `side`,
    then resized to IMG_SIZE x IMG_SIZE.
    """
    scale = IMG_SIZE / side
    K_crop = K.copy().astype(np.float64)
    K_crop[0, 0] *= scale          # fx
    K_crop[1, 1] *= scale          # fy
    K_crop[0, 2] = (K[0, 2] - x0) * scale   # cx
    K_crop[1, 2] = (K[1, 2] - y0) * scale   # cy
    return K_crop


# ---------------------------------------------------------------------------
# 3D reconstruction
# ---------------------------------------------------------------------------

def reconstruct_3d(kpts_px: np.ndarray, depth_rel: np.ndarray,
                   K_crop: np.ndarray, z_root: float,
                   depth_scale: float) -> np.ndarray:
    """
    Back-project 2D keypoints + root-relative depth into 3D camera space.

    Args:
        kpts_px    : (21, 2) joint positions in IMG_SIZE pixel space
        depth_rel  : (21,)   root-relative depth in normalised model units
        K_crop     : (3, 3)  camera intrinsics adjusted for the crop
        z_root     : assumed depth of the wrist joint (mm)
        depth_scale: scale factor converting depth_rel units → mm

    Returns:
        joints_3d  : (21, 3) XYZ in camera space (mm), root-centred
    """
    fx = K_crop[0, 0]; fy = K_crop[1, 1]
    cx = K_crop[0, 2]; cy = K_crop[1, 2]

    # Absolute depth per joint
    z = z_root + depth_rel * depth_scale      # (21,)

    u = kpts_px[:, 0]  # (21,)
    v = kpts_px[:, 1]

    X = (u - cx) / fx * z
    Y = (v - cy) / fy * z
    Z = z

    joints_3d = np.stack([X, Y, Z], axis=-1)  # (21, 3)
    # Root-centre so visualisation is stable regardless of z_root assumption
    joints_3d -= joints_3d[0:1]
    return joints_3d


# ---------------------------------------------------------------------------
# 3D rendering (matplotlib Agg → numpy BGR)
# ---------------------------------------------------------------------------

def make_3d_figure(size_px: int = 400):
    """Create a persistent matplotlib figure for 3D rendering."""
    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi),
                     facecolor='#111111', dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor('#111111')
    fig.tight_layout(pad=0.5)
    return fig, ax


def render_3d_frame(fig, ax, joints_3d: np.ndarray) -> np.ndarray:
    """
    Draw the hand skeleton in 3D onto `ax`, render to a BGR numpy image.
    joints_3d : (21, 3) root-centred XYZ
    """
    ax.cla()
    ax.set_facecolor('#111111')

    # Auto-range: symmetric cube centred on the data
    r = max(np.abs(joints_3d).max() * 1.1, 50.0)
    ax.set_xlim(-r, r); ax.set_ylim(-r, r); ax.set_zlim(-r, r)
    ax.set_xlabel('X', color='white', fontsize=7)
    ax.set_ylabel('Y', color='white', fontsize=7)
    ax.set_zlabel('Z', color='white', fontsize=7)
    ax.tick_params(colors='#888888', labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('#333333')

    # Bones
    for s, e in CONNECTIONS:
        xs = [joints_3d[s, 0], joints_3d[e, 0]]
        ys = [joints_3d[s, 1], joints_3d[e, 1]]
        zs = [joints_3d[s, 2], joints_3d[e, 2]]
        ax.plot(xs, ys, zs, color=JOINT_RGB[s], linewidth=1.5)

    # Joints
    for k, pt in enumerate(joints_3d):
        ax.scatter(pt[0], pt[1], pt[2],
                   color=JOINT_RGB[k], s=18, zorder=5)

    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img_rgb = buf.reshape(h, w, 3)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)


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
# 2D drawing
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
    parser.add_argument('--model',       type=str, default='sdf_best_model.pt')
    parser.add_argument('--model-type',  type=str, default='sdf',
                        choices=['heatmap', 'sdf', 'hybrid'],
                        help='Model architecture (default: sdf)')
    parser.add_argument('--camera',      type=int, default=0,
                        help='Webcam device index (default 0)')
    parser.add_argument('--no-detect',   action='store_true',
                        help='Disable MediaPipe hand detector, use centre-crop fallback')
    parser.add_argument('--show-3d',     action='store_true',
                        help='Show a second window with the 3D skeleton reconstruction')
    parser.add_argument('--K-file',      type=str, default='webcam_K.json',
                        help='Path to camera intrinsics file (json/npy/txt, default webcam_K.json)')
    parser.add_argument('--z-root',      type=float, default=600.0,
                        help='Assumed depth of the wrist in mm (default 600)')
    parser.add_argument('--depth-scale', type=float, default=100.0,
                        help='Scale: depth_rel model units → mm (default 100)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device     : {device}')
    print(f'Model type : {args.model_type}')

    # Load pose model
    print(f'Loading {args.model} ...')
    if args.model_type == 'sdf':
        model = SDFHandPoseNet(num_kpts=21, pretrained_backbone=False)
    elif args.model_type == 'hybrid':
        model = HybridHandPoseNet(num_kpts=21, pretrained_backbone=False)
    else:
        model = SingleViewModel(num_kpts=21)
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device).eval()
    print('Model ready.')

    # Warmup — use ext_pts for SDF to avoid slow dense grid
    with torch.no_grad():
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE, device=device)
        if args.model_type == 'sdf':
            model(dummy, ext_pts=torch.zeros(1, N_PTS, 3, device=device))
        else:
            model(dummy)

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

    # 3D visualisation setup
    K = None
    fig_3d = ax_3d = None
    if args.show_3d:
        K = load_K(args.K_file)
        fig_3d, ax_3d = make_3d_figure(size_px=400)
        print(f'3D view : ON  (K from {args.K_file}, z_root={args.z_root}mm, '
              f'depth_scale={args.depth_scale})')
    else:
        print('3D view : OFF  (--show-3d to enable)')
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
            if args.model_type == 'sdf':
                ext_pts = torch.empty(1, N_PTS, 3, device=device).uniform_(-1., 1.)
                pose_2d, depth_rel, _, _ = model(inp, ext_pts=ext_pts)
            elif args.model_type == 'hybrid':
                pose_2d, depth_rel, _, _, _ = model(inp)
            else:
                pose_2d, depth_rel, _, _ = model(inp)

        kpts      = pose_2d[0].cpu().numpy()    # (21, 2) in IMG_SIZE space
        depth_np  = depth_rel[0].cpu().numpy()  # (21,)

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
        det_label = f'BlazePalm+{args.model_type.upper()}' if use_detector else f'{args.model_type.upper()} (no detector)'
        cv2.putText(canvas, f'FPS: {fps:.1f}  [{det_label}]', (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, 'Q/ESC=quit  S=save', (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(f'Hand Pose ({args.model_type.upper()}) -- realtime', canvas)

        # -- 3D visualisation --
        if args.show_3d:
            K_crop    = adjust_K_for_crop(K, x0, y0, side)
            joints_3d = reconstruct_3d(kpts, depth_np, K_crop,
                                       args.z_root, args.depth_scale)
            img_3d    = render_3d_frame(fig_3d, ax_3d, joints_3d)
            cv2.imshow('3D Hand Pose', img_3d)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('s'):
            fname = save_dir / f'capture_{int(time.time())}.png'
            cv2.imwrite(str(fname), canvas)
            print(f'Saved {fname}')
            if args.show_3d:
                fname_3d = save_dir / f'capture_{int(time.time())}_3d.png'
                cv2.imwrite(str(fname_3d), img_3d)
                print(f'Saved {fname_3d}')

    if mp_hands is not None:
        mp_hands.close()
    if fig_3d is not None:
        plt.close(fig_3d)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
