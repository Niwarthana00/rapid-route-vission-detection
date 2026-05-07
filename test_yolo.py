import cv2
import json
import os
import logging
from pathlib import Path
from dataclasses import dataclass, field
from ultralytics import YOLO

STREAM_URL           = "http://192.168.8.102:8080/video"
ROI_CONFIG_FILE      = Path("roi_config.json")
MODEL_WEIGHTS        = "yolov8n.pt"
DETECTION_CONFIDENCE = 0.35
PERSON_CLASS_ID      = 0
VIDEO_FPS            = 20.0
OUTPUT_DIR           = Path("output")

HEADLESS = (os.environ.get("DISPLAY") is None) or (os.environ.get("HEADLESS", "0") == "1")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

@dataclass
class TrackState:
    zones:             dict = field(default_factory=dict)
    entry_side:        dict = field(default_factory=dict)
    in_total:          int  = 0
    out_total:         int  = 0
    persistent_inside: set  = field(default_factory=set)

    @property
    def inside_now(self) -> int:
        return len(self.persistent_inside)

def get_side(p, p1, p2):
    return (p[0] - p1[0]) * (p2[1] - p1[1]) - (p[1] - p1[1]) * (p2[0] - p1[0])

def run():
    if not ROI_CONFIG_FILE.exists():
        log.error(f"Error: {ROI_CONFIG_FILE} not found!")
        return

    with open(ROI_CONFIG_FILE, 'r') as f:
        roi = json.load(f)

    rx, ry, rw, rh = roi['x'], roi['y'], roi['width'], roi['height']

    l1_y = ry + int(rh * 0.85) # Outer Line (Red)
    l1_p1, l1_p2 = (rx, l1_y), (rx + rw, l1_y)

    l2_y = ry + rh             # Inner Line (Green)
    l2_p1, l2_p2 = (rx, l2_y), (rx + rw, l2_y)

    model  = YOLO(MODEL_WEIGHTS)
    state  = TrackState()
    writer = None

    results = model.track(
        source=STREAM_URL,
        stream=True,
        persist=True,
        conf=DETECTION_CONFIDENCE,
        classes=[PERSON_CLASS_ID],
        tracker="bytetrack.yaml"
    )

    try:
        for result in results:
            frame = result.orig_img.copy()

            if result.boxes.id is not None:
                boxes   = result.boxes.xyxy.cpu().numpy()
                ids     = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()

                for box, track_id, cls in zip(boxes, ids, classes):
                    if cls != PERSON_CLASS_ID: continue

                    cx, cy = (box[0] + box[2]) / 2, box[3]

                    if not (rx <= cx <= rx + rw and ry <= cy <= ry + rh + 20):
                        continue

                    side1 = get_side((cx, cy), l1_p1, l1_p2)
                    side2 = get_side((cx, cy), l2_p1, l2_p2)

                    if side1 > 0:
                        current_zone = "outside"
                    elif side2 < 0:
                        current_zone = "inside"
                    else:
                        current_zone = "confirm_zone"

                    prev_zone = state.zones.get(track_id)

                    if current_zone == "confirm_zone" and prev_zone != "confirm_zone":
                        state.entry_side[track_id] = prev_zone

                    if current_zone == "inside" and prev_zone == "confirm_zone":
                        if state.entry_side.get(track_id) == "outside":
                            state.in_total += 1
                            state.persistent_inside.add(track_id)
                            log.info(f"==> ENTERED: ID {track_id} | Total IN: {state.in_total}")
                            state.entry_side[track_id] = None 

                    elif current_zone == "outside" and prev_zone == "confirm_zone":
                        if state.entry_side.get(track_id) == "inside":
                            state.out_total += 1
                            state.persistent_inside.discard(track_id)
                            log.info(f"<== EXITED: ID {track_id} | Total OUT: {state.out_total}")
                            state.entry_side[track_id] = None

                    state.zones[track_id] = current_zone

                frame = result.plot()

            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 200, 0), 2)
            cv2.line(frame, l1_p1, l1_p2, (0, 0, 255), 3)
            cv2.line(frame, l2_p1, l2_p2, (0, 255, 0), 3)
            
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (350, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"IN: {state.in_total}  OUT: {state.out_total}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 100), 2)
            cv2.putText(frame, f"INSIDE NOW: {state.inside_now}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            if writer is None:
                OUTPUT_DIR.mkdir(exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(OUTPUT_DIR / "counter_output.mp4"), fourcc, VIDEO_FPS, (frame.shape[1], frame.shape[0]))

            writer.write(frame)
            if not HEADLESS:
                cv2.imshow("Advanced People Counter", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        log.info("Stopped.")
    finally:
        if writer: writer.release()
        cv2.destroyAllWindows()
        log.info(f"Final Count - IN: {state.in_total}, OUT: {state.out_total}")

if __name__ == "__main__":
    run()