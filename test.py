import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient # type: ignore
from datetime import datetime
import os
import argparse


# ---------------------------
# ROI / REGION HELPERS
# ---------------------------
def norm_poly_to_abs(w, h, poly_norm):
    return np.array([[int(x * w), int(y * h)] for (x, y) in poly_norm], dtype=np.int32)

def norm_rects_to_abs(w, h, rects_norm):
    out = []
    for (x1, y1, x2, y2) in rects_norm:
        out.append([int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)])
    return out

def point_in_poly(cx, cy, poly_abs):
    return cv2.pointPolygonTest(poly_abs, (float(cx), float(cy)), False) >= 0

def clamp_box_xyxy(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2

def box_area_xyxy(b):
    x1, y1, x2, y2 = b
    return max(0, x2 - x1) * max(0, y2 - y1)

def inter_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0
    return (ix2 - ix1) * (iy2 - iy1)

def iou(a, b):
    ia = inter_area(a, b)
    if ia <= 0:
        return 0.0
    ua = box_area_xyxy(a) + box_area_xyxy(b) - ia
    return ia / max(1, ua)


# ---------------------------
# DEFAULTS (tune as needed)
# ---------------------------
MODEL_ID_DEFAULT = "accident-detection-hlstb/1"

# Accuracy Settings
CONF_THRESHOLD_DEFAULT = 0.55
VERIFY_FRAMES_DEFAULT  = 3
ACCIDENT_HOLD_TTL_DEFAULT = 30

# Road ROI (Normalized 0..1) — adjust to your camera
ROAD_ROI_POLY_NORM_DEFAULT = [
    (0.00, 0.30),
    (1.00, 0.30),
    (1.00, 1.00),
    (0.00, 1.00),
]

# Exclude watermarks/overlays (Normalized rects x1,y1,x2,y2)
# Example: top-right logo area like "Caught On Cam"
EXCLUDE_RECTS_NORM_DEFAULT = [
    (0.78, 0.00, 1.00, 0.22),
]


# ---------------------------
# MAIN
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Input video path (e.g. v38.mov)")
    ap.add_argument("--model_id", default=MODEL_ID_DEFAULT, help="Roboflow model id")
    ap.add_argument("--api_key", default=None, help="Roboflow API key (or set env ROBOFLOW_API_KEY)")
    ap.add_argument("--conf", type=float, default=CONF_THRESHOLD_DEFAULT, help="Confidence threshold (0..1)")
    ap.add_argument("--verify_frames", type=int, default=VERIFY_FRAMES_DEFAULT, help="Consecutive frames required")
    ap.add_argument("--hold_ttl", type=int, default=ACCIDENT_HOLD_TTL_DEFAULT, help="Hold confirmed box for N frames")

    ap.add_argument("--infer_every", type=int, default=3,
                    help="Run cloud inference every N frames (speed/rate-limit control). 1 = every frame.")
    ap.add_argument("--save_annotated", action="store_true", help="Save annotated mp4")
    ap.add_argument("--out_dir", default="roboflow_cases", help="Output folder")

    ap.add_argument("--no_exclude", action="store_true", help="Disable watermark exclude rects")
    ap.add_argument("--no_roi", action="store_true", help="Disable road ROI gating")

    args = ap.parse_args()

    api_key = args.api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("Missing API key. Set ROBOFLOW_API_KEY env var or pass --api_key YOUR_KEY")

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {args.video}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0

    roi_abs = norm_poly_to_abs(w, h, ROAD_ROI_POLY_NORM_DEFAULT)
    exclude_abs = norm_rects_to_abs(w, h, EXCLUDE_RECTS_NORM_DEFAULT)

    # Output setup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    case_dir = os.path.join(args.out_dir, f"Case_{ts}")
    os.makedirs(case_dir, exist_ok=True)

    out_video = None
    if args.save_annotated:
        out_path = os.path.join(case_dir, "Annotated_Evidence.mp4")
        out_video = cv2.VideoWriter(
            out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
        )

    print(f"[SYSTEM] Monitoring {args.video} via Roboflow Cloud")
    print(f"[SYSTEM] Output: {case_dir}")
    print(f"[SYSTEM] infer_every={args.infer_every}, conf={args.conf}, verify_frames={args.verify_frames}, hold_ttl={args.hold_ttl}")

    # State variables
    consecutive_hits = 0
    confirmed_box = None
    confirmed_conf = 0.0
    ttl_counter = 0
    evidence_saved = False

    # We reuse last inference results for skipped frames
    last_best = None
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Decide whether to run inference on this frame
        run_infer = (args.infer_every <= 1) or (frame_idx % args.infer_every == 0)

        best_this_frame = None
        if run_infer:
            # Roboflow generally accepts numpy arrays; frame is BGR.
            # If your model behaves better with RGB, uncomment:
            # send_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            send_img = frame

            result = client.infer(send_img, model_id=args.model_id)
            predictions = result.get("predictions", []) or []

            # Select best prediction that passes gates
            for p in predictions:
                conf = float(p.get("confidence", 0.0))
                if conf < args.conf:
                    continue

                # Roboflow predictions: x,y,width,height are typically in PIXELS for serverless endpoints
                cx = float(p.get("x", 0.0))
                cy = float(p.get("y", 0.0))
                bw = float(p.get("width", 0.0))
                bh = float(p.get("height", 0.0))

                # ROI gate (optional)
                if not args.no_roi:
                    if not point_in_poly(cx, cy, roi_abs):
                        continue

                # Convert to xyxy
                x1 = cx - bw / 2.0
                y1 = cy - bh / 2.0
                x2 = cx + bw / 2.0
                y2 = cy + bh / 2.0
                x1, y1, x2, y2 = clamp_box_xyxy(x1, y1, x2, y2, w, h)
                box = (x1, y1, x2, y2)

                # Exclude watermark/overlay regions (optional)
                if not args.no_exclude:
                    bad = False
                    for r in exclude_abs:
                        if iou(box, r) >= 0.20:  # 20% overlap with excluded zone
                            bad = True
                            break
                    if bad:
                        continue

                if best_this_frame is None or conf > best_this_frame["confidence"]:
                    best_this_frame = {
                        "confidence": conf,
                        "box": box,
                        "center": (cx, cy)
                    }

            last_best = best_this_frame
        else:
            best_this_frame = last_best  # reuse last inference

        # Temporal verification
        if best_this_frame is not None:
            consecutive_hits += 1
        else:
            consecutive_hits = 0

        # Confirm accident after N consecutive frames
        if best_this_frame is not None and consecutive_hits >= args.verify_frames:
            confirmed_box = best_this_frame["box"]
            confirmed_conf = best_this_frame["confidence"]
            ttl_counter = args.hold_ttl

            # Save evidence once at the moment of confirmation
            if (consecutive_hits == args.verify_frames) and (not evidence_saved):
                print(f"[ALERT] Accident Confirmed at {datetime.now().strftime('%H:%M:%S')}  conf={confirmed_conf:.2f}")
                snap_path = os.path.join(case_dir, f"Impact_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                cv2.imwrite(snap_path, frame)
                evidence_saved = True

                # Optional tiny report
                with open(os.path.join(case_dir, "Event.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Confirmed: {datetime.now().isoformat()}\n")
                    f.write(f"Confidence: {confirmed_conf:.4f}\n")
                    f.write(f"VerifyFrames: {args.verify_frames}\n")
                    f.write(f"ConfThreshold: {args.conf}\n")
                    f.write(f"InferEvery: {args.infer_every}\n")

        # Draw ROI + exclude zones for visibility
        if not args.no_roi:
            cv2.polylines(frame, [roi_abs], True, (255, 255, 0), 2)
            cv2.putText(frame, "ROI", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        if not args.no_exclude:
            for r in exclude_abs:
                cv2.rectangle(frame, (r[0], r[1]), (r[2], r[3]), (255, 0, 255), 2)
                cv2.putText(frame, "EXCLUDE", (r[0], min(h - 10, r[3] + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        # Draw raw best (yellow) when available
        if best_this_frame is not None:
            x1, y1, x2, y2 = best_this_frame["box"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"RAW {best_this_frame['confidence']:.2f}",
                        (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw confirmed (red) while TTL active
        if ttl_counter > 0 and confirmed_box is not None:
            x1, y1, x2, y2 = confirmed_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
            cv2.putText(frame, f"CRASH CONFIRMED: {confirmed_conf:.0%}",
                        (x1, max(25, y1 - 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            ttl_counter -= 1
        else:
            confirmed_box = None

        # HUD
        cv2.putText(frame, f"frame={frame_idx} hits={consecutive_hits}/{args.verify_frames} infer={'Y' if run_infer else 'N'}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if out_video is not None:
            out_video.write(frame)

        cv2.imshow("Inspector AI - Roboflow Accident Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out_video is not None:
        out_video.release()
    cv2.destroyAllWindows()
    print("[DONE] Output written to:", case_dir)


if __name__ == "__main__":
    main()