import argparse
import cv2
import json
import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from ultralytics import YOLO

from kafka_publisher import TelemetryPublisher

CAMERA_STREAM_URL    = "http://192.168.8.102:8080/video"
VIDEO_FILE_PATH      = "video.webm"   # <-- change this to your video's filename/path
ROI_CONFIG_FILE      = Path("roi_config.json")
MODEL_WEIGHTS        = "yolov8n.pt"
DETECTION_CONFIDENCE = 0.4
PERSON_CLASS_ID      = 0
VIDEO_FPS            = 20.0
OUTPUT_DIR           = Path("output")

# Kafka telemetry — must match the bus_id used by the other bus repos
# (bus-seat-detection, rapid-route-telemetry-service) for the same physical bus.
BUS_ID       = os.environ.get("BUS_ID", "BUS-001")
KAFKA_BROKER = os.environ.get("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC  = os.environ.get("KAFKA_TOPIC", "iot-telemetry")  # same topic rapid-route-telemetry-service uses
KAFKA_ENABLED = os.environ.get("KAFKA_ENABLED", "1") != "0"
# Send a snapshot event every N processed frames even with no IN/OUT change,
# so the aggregator's "inside_now" never goes stale during a quiet stretch.
HEARTBEAT_EVERY_N_FRAMES = 30

# Colors for drawing the two zones (BGR)
ZONE_COLORS = {"outside": (0, 165, 255), "inside": (0, 255, 0)}

HEADLESS = (os.environ.get("DISPLAY") is None) or (os.environ.get("HEADLESS", "0") == "1")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def resolve_source(arg: str) -> str:
    """Turn a --source value into an actual video source for OpenCV/YOLO."""
    if arg is None or arg.lower() == "camera":
        return CAMERA_STREAM_URL
    if arg.lower() == "video":
        return VIDEO_FILE_PATH
    # Anything else (a URL or a custom file path) is used as-is.
    return arg


@dataclass
class TrackState:
    # track_id -> last known zone ("outside" / "inside")
    track_zone: dict = field(default_factory=dict)
    in_total:   int  = 0
    out_total:  int  = 0
    persistent_inside: set = field(default_factory=set)

    @property
    def inside_now(self) -> int:
        return len(self.persistent_inside)


def point_in_zone(cx: float, cy: float, zone: dict) -> bool:
    return zone["x"] <= cx <= zone["x"] + zone["width"] and zone["y"] <= cy <= zone["y"] + zone["height"]


def run(source: str):
    if not ROI_CONFIG_FILE.exists():
        print(f"Error: {ROI_CONFIG_FILE} not found!")
        return

    with open(ROI_CONFIG_FILE, 'r') as f:
        zones = json.load(f)

    if "outside" not in zones or "inside" not in zones:
        print("Error: roi_config.json must contain 'outside' and 'inside' zones.")
        print("       Re-run roi.py to select both zones.")
        return

    outside_zone = zones["outside"]
    inside_zone = zones["inside"]

    model = YOLO(MODEL_WEIGHTS)
    cap = cv2.VideoCapture(source)
    state = TrackState()
    writer = None

    publisher = TelemetryPublisher(
        bus_id=BUS_ID, broker=KAFKA_BROKER, topic=KAFKA_TOPIC, enabled=KAFKA_ENABLED
    )

    def publish_snapshot():
        publisher.publish({
            "in_total": state.in_total,
            "out_total": state.out_total,
            "inside_now": state.inside_now,
        })

    print(f"[INFO] Using source: {source}")
    results = model.track(source=source, stream=True, persist=True, conf=DETECTION_CONFIDENCE)

    print("System Running... Press Ctrl+C to stop.")

    frame_counter = 0
    try:
        for result in results:
            frame_counter += 1
            frame = result.plot()

            if result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.int().cpu().tolist()
                classes = result.boxes.cls.int().cpu().tolist()

                for box, track_id, cls in zip(boxes, ids, classes):
                    if cls != PERSON_CLASS_ID:
                        continue

                    foot_y = float(box[3])
                    cx = (box[0] + box[2]) / 2

                    if point_in_zone(cx, foot_y, outside_zone):
                        current_zone = "outside"
                    elif point_in_zone(cx, foot_y, inside_zone):
                        current_zone = "inside"
                    else:
                        # Not in either zone right now — ignore this frame for this person,
                        # but keep their last known zone as-is.
                        continue

                    prev_zone = state.track_zone.get(track_id)

                    if prev_zone is None:
                        state.track_zone[track_id] = current_zone
                        if current_zone == "inside":
                            state.persistent_inside.add(track_id)
                        continue

                    if prev_zone == "outside" and current_zone == "inside":
                        state.in_total += 1
                        state.persistent_inside.add(track_id)
                        log.info(f"--> ENTERED: ID {track_id}")
                        publish_snapshot()

                    elif prev_zone == "inside" and current_zone == "outside":
                        state.out_total += 1
                        state.persistent_inside.discard(track_id)
                        log.info(f"<-- EXITED: ID {track_id}")
                        publish_snapshot()

                    state.track_zone[track_id] = current_zone

            if frame_counter % HEARTBEAT_EVERY_N_FRAMES == 0:
                publish_snapshot()

            for zone_key, zone in (("outside", outside_zone), ("inside", inside_zone)):
                zx, zy, zw, zh = zone["x"], zone["y"], zone["width"], zone["height"]
                color = ZONE_COLORS[zone_key]
                cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), color, 2)
                cv2.putText(frame, zone_key.upper(), (zx, max(20, zy - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (320, 130), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            cv2.putText(frame, f"IN: {state.in_total}  OUT: {state.out_total}", (20, 50), 0, 0.9, (0, 255, 100), 2)
            cv2.putText(frame, f"INSIDE NOW: {state.inside_now}", (20, 100), 0, 0.9, (0, 255, 255), 2)

            if writer is None:
                OUTPUT_DIR.mkdir(exist_ok=True)
                path = OUTPUT_DIR / "output_video.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), VIDEO_FPS, (frame.shape[1], frame.shape[0]))

            writer.write(frame)

            if not HEADLESS:
                cv2.imshow("Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        if writer:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
        publisher.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO in/out counter for camera or video source.")
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
    run(resolve_source(args.source))
