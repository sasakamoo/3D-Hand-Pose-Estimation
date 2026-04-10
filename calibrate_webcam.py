"""
calibrate_webcam.py — Webcam intrinsics calibration
====================================================
Uses a printed checkerboard to estimate the camera matrix K and distortion
coefficients, then saves webcam_K.json for use with realtime.py --show-3d.

Steps:
  1. Print calibration_board.png (generated automatically, A4 / Letter)
  2. Run this script and hold the board at various angles in front of the camera
  3. Press SPACE to capture a frame (need at least 15–20 good captures)
  4. Press C to run calibration and save webcam_K.json
  5. Press Q/ESC to quit

Usage:
    python calibrate_webcam.py
    python calibrate_webcam.py --camera 1 --cols 9 --rows 6 --square 25
    python calibrate_webcam.py --preview-only   # just view the camera

Arguments:
    --camera      Webcam device index (default 0)
    --cols        Inner corners per row   (default 9, for a 10-column board)
    --rows        Inner corners per column (default 6, for a 7-row board)
    --square      Physical square size in mm (default 25 — measure your printout!)
    --output      Output JSON path (default webcam_K.json)
    --min-frames  Minimum good frames before calibration is allowed (default 15)
    --preview-only  Just open the camera without calibrating

Board generation:
    A calibration_board.png is saved automatically. Print it at 100% scale
    (no fit-to-page) and measure one square with a ruler to get --square value.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


# ── Board generation ──────────────────────────────────────────────────────────

def generate_board_image(cols: int, rows: int,
                         square_px: int = 80,
                         border_px: int = 40) -> np.ndarray:
    """
    Generate a checkerboard calibration image (cols×rows inner corners →
    (cols+1)×(rows+1) squares).
    """
    n_cols = cols + 1
    n_rows = rows + 1
    w = n_cols * square_px + 2 * border_px
    h = n_rows * square_px + 2 * border_px
    img = np.ones((h, w), dtype=np.uint8) * 255

    for r in range(n_rows):
        for c in range(n_cols):
            if (r + c) % 2 == 0:
                x0 = border_px + c * square_px
                y0 = border_px + r * square_px
                img[y0:y0+square_px, x0:x0+square_px] = 0

    # Add info text
    img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(img_bgr,
                f'Checkerboard {cols}x{rows} inner corners  —  print at 100% scale',
                (border_px, h - border_px // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 100, 100), 1, cv2.LINE_AA)
    return img_bgr


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate(obj_points, img_points, frame_size):
    """Run cv2.calibrateCamera and return (K, dist, rms)."""
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points, img_points, frame_size,
        None, None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )
    return K, dist, rms


def save_K(K: np.ndarray, dist: np.ndarray, rms: float,
           frame_size: tuple, path: str):
    data = {
        'fx': float(K[0, 0]),
        'fy': float(K[1, 1]),
        'cx': float(K[0, 2]),
        'cy': float(K[1, 2]),
        'dist': dist.flatten().tolist(),
        'rms_px': round(rms, 4),
        'frame_w': frame_size[0],
        'frame_h': frame_size[1],
        'K_matrix': K.tolist(),
    }
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\nSaved {path}')
    print(f'  fx={data["fx"]:.1f}  fy={data["fy"]:.1f}  '
          f'cx={data["cx"]:.1f}  cy={data["cy"]:.1f}')
    print(f'  RMS reprojection error: {rms:.4f} px  '
          f'(good < 0.5, acceptable < 1.0)')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera',       type=int,   default=0)
    parser.add_argument('--cols',         type=int,   default=9,
                        help='Inner corner columns (default 9)')
    parser.add_argument('--rows',         type=int,   default=6,
                        help='Inner corner rows (default 6)')
    parser.add_argument('--square',       type=float, default=25.0,
                        help='Physical square size in mm (measure your printout)')
    parser.add_argument('--output',       type=str,   default='webcam_K.json')
    parser.add_argument('--min-frames',   type=int,   default=15)
    parser.add_argument('--preview-only', action='store_true')
    args = parser.parse_args()

    COLS, ROWS = args.cols, args.rows

    # ── Generate and save board image ─────────────────────────────────────
    board_path = 'calibration_board.png'
    board_img  = generate_board_image(COLS, ROWS)
    cv2.imwrite(board_path, board_img)
    print(f'Checkerboard saved → {board_path}')
    print(f'Print at 100% scale, then measure one square and set --square <mm>\n')

    if args.preview_only:
        cap = cv2.VideoCapture(args.camera)
        print('Preview mode — press Q/ESC to quit')
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imshow('Camera preview', frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27): break
        cap.release()
        cv2.destroyAllWindows()
        return

    # ── 3D object points for one board pose ──────────────────────────────
    # (0,0,0), (1,0,0), ..., in square_mm units
    objp = np.zeros((ROWS * COLS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)
    objp *= args.square   # convert to mm

    obj_points = []   # 3D points across captures
    img_points = []   # 2D points across captures

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')

    frame_size = None
    last_capture = 0
    print(f'Board: {COLS}×{ROWS} inner corners  |  square={args.square} mm')
    print(f'SPACE=capture  C=calibrate  Q/ESC=quit')
    print(f'Need at least {args.min_frames} good captures before calibrating.\n')

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    while True:
        ret, frame = cap.read()
        if not ret:
            print('Camera read failed.')
            break

        if frame_size is None:
            h, w = frame.shape[:2]
            frame_size = (w, h)

        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (COLS, ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        display = frame.copy()
        n_good  = len(obj_points)

        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, (COLS, ROWS), corners2, found)
            status_color = (0, 220, 0)
            status_text  = f'Board detected!  SPACE to capture  [{n_good}/{args.min_frames}]'
        else:
            status_color = (0, 100, 255)
            status_text  = f'No board — tilt/move it  [{n_good}/{args.min_frames}]'

        cv2.putText(display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

        can_calibrate = n_good >= args.min_frames
        hint = 'C=calibrate' if can_calibrate else f'Need {args.min_frames - n_good} more'
        cv2.putText(display, f'SPACE=capture  {hint}  Q=quit', (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Webcam Calibration', display)
        key = cv2.waitKey(1) & 0xFF

        # ── SPACE: capture ────────────────────────────────────────────────
        if key == ord(' '):
            now = time.time()
            if found and (now - last_capture) > 0.5:   # debounce 0.5 s
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                obj_points.append(objp)
                img_points.append(corners2)
                last_capture = now
                print(f'  Captured frame {len(obj_points):>2}')
                # Flash green
                flash = display.copy()
                flash[:] = (0, 200, 0)
                cv2.addWeighted(display, 0.6, flash, 0.4, 0, display)
                cv2.imshow('Webcam Calibration', display)
                cv2.waitKey(150)
            elif not found:
                print('  No board detected — hold it steadier or adjust lighting')

        # ── C: calibrate ──────────────────────────────────────────────────
        elif key == ord('c'):
            if not can_calibrate:
                print(f'  Need at least {args.min_frames} captures '
                      f'(have {n_good}) — keep going')
            else:
                print(f'\nCalibrating with {n_good} frames ...')
                K, dist, rms = calibrate(obj_points, img_points, frame_size)
                save_K(K, dist, rms, frame_size, args.output)

                # Show undistorted preview
                print('\nShowing undistorted preview — press any key to exit')
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    undist = cv2.undistort(frame, K, dist)
                    combined = np.hstack([frame, undist])
                    cv2.putText(combined, 'Original', (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.putText(combined, 'Undistorted', (frame.shape[1]+10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    cv2.imshow('Calibration result', combined)
                    if cv2.waitKey(1) & 0xFF in (ord('q'), 27, ord(' ')):
                        break
                break

        # ── Q / ESC: quit ─────────────────────────────────────────────────
        elif key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
