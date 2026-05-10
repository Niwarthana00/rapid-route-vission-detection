# rapid-route-vission-detection

YOLOv8-based passenger entry/exit (in/out) counter for a bus door, using a
two-zone (outside/inside) crossing rule. See `roi.py` to select the zones
and `test_yolo.py` to run detection + counting.

## Run without Docker

```bash
pip install -r requirements.txt
BUS_ID=BUS-001 KAFKA_BROKER=localhost:9092 KAFKA_TOPIC=iot-telemetry \
    python test_yolo.py --source video
```

## Run with Docker

```bash
docker compose up --build
```

Builds the image and runs `test_yolo.py --source video` (the bundled
`video.webm`) by default — see `command:` in `docker-compose.yml` to point
it at a live camera stream instead. The container is headless (no
`cv2.imshow`), so it just writes the annotated output to `output/`.

`extra_hosts: host.docker.internal:host-gateway` lets the container reach a
Kafka broker running on your host machine (e.g. started by
`rapid-route-kafka-infra`'s `compose.yml`) at `host.docker.internal:9092` —
the same pattern `rapid-route-telemetry-service` already uses.

Start order for the full demo: **Kafka infra → telemetry-service →
bus-state-aggregator → this repo + bus-seat-detection** (either order
between the last two).

## Kafka telemetry

`test_yolo.py` publishes to the **same Kafka topic and flat JSON style**
that `rapid-route-telemetry-service` already uses (`iot-telemetry`, keyed
by `bus_id`, no envelope wrapper) — nothing about that service needed to
change. Sent on every IN/OUT change, plus a heartbeat every 30 processed
frames so `inside_now` never goes stale:

```json
{"bus_id": "BUS-001", "in_total": 4, "out_total": 1, "inside_now": 12}
```

Env vars (all optional, sensible defaults shown):

```
BUS_ID=BUS-001
KAFKA_BROKER=localhost:9092
KAFKA_TOPIC=iot-telemetry
KAFKA_ENABLED=1        # set to 0 to disable publishing
```
