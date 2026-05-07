"""
Controls:
    SPACE  - Freeze the current frame and draw the ROI box
    ENTER  - Confirm the selected ROI
    C      - Cancel / re-draw the ROI box
    Q      - Quit without saving
"""

import cv2
import json
import sys
from pathlib import Path

STREAM_URL = "http://192.168.8.102:8080/video"
OUTPUT_FILE = Path("roi_config.json")
WINDOW_NAME = "ROI Selector"
WARMUP_FRAMES = 10
CONFIRM_DISPLAY_MS = 2000


def read_stable_frame(capture: cv2.VideoCapture) -> tuple[bool, any]:
    for _ in range(WARMUP_FRAMES):
        capture.read()
    return capture.read()


def save_roi(roi: tuple[int, int, int, int]) -> None:
    x, y, w, h = roi
    payload = {"x": x, "y": y, "width": w, "height": h}
    OUTPUT_FILE.write_text(json.dumps(payload, indent=4))
    print(f"ROI saved → {OUTPUT_FILE.resolve()}")


def draw_confirmed_roi(frame, roi: tuple[int, int, int, int]):
    x, y, w, h = roi
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
    label = f"ROI  x={x}  y={y}  w={w}  h={h}"
    cv2.putText(overlay, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return overlay


def run() -> None:
    capture = cv2.VideoCapture(STREAM_URL)

    if not capture.isOpened():
        print(f"[ERROR] Cannot open stream: {STREAM_URL}")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    print("[INFO] Live feed started.")
    print("       SPACE → freeze & select ROI   |   Q → quit")

    while True:
        grabbed, live_frame = capture.read()

        if not grabbed:
            print("[ERROR] Lost connection to stream.")
            break

        display_frame = live_frame.copy()
        cv2.putText(display_frame, "SPACE: Select ROI  |  Q: Quit",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (0, 220, 0), 2)
        cv2.imshow(WINDOW_NAME, display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Quit — no ROI saved.")
            break

        if key == ord(' '):
            print("[INFO] Frame frozen. Draw your ROI box.")

            _, frozen_frame = read_stable_frame(capture)

            roi = cv2.selectROI(WINDOW_NAME, frozen_frame,
                                fromCenter=False, showCrosshair=True)

            if roi == (0, 0, 0, 0):
                print("[INFO] Selection cancelled — resuming live feed.")
                continue

            x, y, w, h = roi
            print(f"[INFO] ROI selected → x={x}  y={y}  w={w}  h={h}")

            confirmed_frame = draw_confirmed_roi(frozen_frame, roi)
            cv2.imshow(WINDOW_NAME, confirmed_frame)
            cv2.waitKey(CONFIRM_DISPLAY_MS)

            save_roi(roi)
            break

    capture.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)


if __name__ == "__main__":
    run()