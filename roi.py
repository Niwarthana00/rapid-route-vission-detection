import argparse
import cv2
import json
import sys
from pathlib import Path

CAMERA_STREAM_URL = "http://192.168.8.102:8080/video"
VIDEO_FILE_PATH = "video.webm" 
OUTPUT_FILE = Path("roi_config.json")
WINDOW_NAME = "ROI Selector"
WARMUP_FRAMES = 10
CONFIRM_DISPLAY_MS = 1500

ZONES = [
    ("outside", "OUTSIDE ZONE (door / step - not boarded yet)", (0, 165, 255)),   # orange
    ("inside",  "INSIDE ZONE (aisle - already boarded)",         (0, 255, 0)),    # green
]


def resolve_source(arg: str) -> str:
    if arg is None or arg.lower() == "camera":
        return CAMERA_STREAM_URL
    if arg.lower() == "video":
        return VIDEO_FILE_PATH
    return arg


def read_stable_frame(capture: cv2.VideoCapture) -> tuple[bool, any]:
    for _ in range(WARMUP_FRAMES):
        capture.read()
    return capture.read()


def save_zones(zones: dict) -> None:
    OUTPUT_FILE.write_text(json.dumps(zones, indent=4))
    print(f"Zones saved → {OUTPUT_FILE.resolve()}")


def draw_zones(frame, zones: dict):
    overlay = frame.copy()
    for zone_key, label, color in ZONES:
        if zone_key not in zones:
            continue
        z = zones[zone_key]
        x, y, w, h = z["x"], z["y"], z["width"], z["height"]
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        cv2.putText(overlay, zone_key.upper(), (x, max(20, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return overlay


def run(source: str) -> None:
    capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        print(f"[ERROR] Cannot open source: {source}")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("[INFO] Live feed started.")
    print("       SPACE → freeze & select zones   |   Q → quit")

    frozen_frame = None

    while True:
        grabbed, live_frame = capture.read()

        if not grabbed:
            print("[ERROR] Lost connection to source.")
            break

        display_frame = live_frame.copy()
        cv2.putText(display_frame, "SPACE: Select zones  |  Q: Quit",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 220, 0), 2)
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Quit — nothing saved.")
            break

        if key == ord(' '):
            print("[INFO] Frame frozen.")
            _, frozen_frame = read_stable_frame(capture)
            break

    if frozen_frame is None:
        capture.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return

    zones = {}
    for zone_key, label, color in ZONES:
        print(f"[INFO] Draw the {label}, then press ENTER (or SPACE) to confirm.")
        preview = draw_zones(frozen_frame, zones)
        roi = cv2.selectROI(WINDOW_NAME, preview, fromCenter=False, showCrosshair=True)

        if roi == (0, 0, 0, 0):
            print(f"[WARN] No box drawn for '{zone_key}' — quitting without saving.")
            capture.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return

        x, y, w, h = roi
        zones[zone_key] = {"x": x, "y": y, "width": w, "height": h}
        print(f"[INFO] {zone_key} zone → x={x}  y={y}  w={w}  h={h}")

        confirmed_frame = draw_zones(frozen_frame, zones)
        cv2.imshow(WINDOW_NAME, confirmed_frame)
        cv2.waitKey(CONFIRM_DISPLAY_MS)

    save_zones(zones)

    capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-zone ROI selector for camera or video source.")
    parser.add_argument(
        "--source",
        default="camera",
        help=(
            "'camera' (default, uses CAMERA_STREAM_URL), 'video' (uses VIDEO_FILE_PATH), "
            "or a custom URL/file path."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resolved_source = resolve_source(args.source)
    print(f"[INFO] Using source: {resolved_source}")
    run(resolved_source)
