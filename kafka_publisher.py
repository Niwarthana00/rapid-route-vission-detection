import json
import logging
import time

logger = logging.getLogger("kafka_publisher")

try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None


class TelemetryPublisher:
    def __init__(self, bus_id, broker="localhost:9092", topic="iot-telemetry", enabled=True):
        self.bus_id = bus_id
        self.topic = topic
        self.enabled = enabled
        self._producer = None

        if not enabled:
            return
        if KafkaProducer is None:
            logger.warning(
                "kafka-python not installed - telemetry publishing disabled. "
                "Install with: pip install kafka-python"
            )
            self.enabled = False
            return

        try:
            self._producer = KafkaProducer(
                bootstrap_servers=broker,
                key_serializer=lambda k: k.encode("utf-8"),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                linger_ms=50,
                request_timeout_ms=5000,
            )
        except Exception as exc:
            logger.warning("Kafka producer init failed (%s) - telemetry publishing disabled.", exc)
            self.enabled = False

    def publish(self, payload: dict) -> None:
        if not self.enabled or self._producer is None:
            return
        event = {"bus_id": self.bus_id, **payload}
        try:
            self._producer.send(self.topic, key=self.bus_id, value=event)
        except Exception as exc:
            logger.warning("Kafka publish failed: %s", exc)

    def close(self) -> None:
        if self._producer is not None:
            try:
                self._producer.flush(timeout=5)
                self._producer.close(timeout=5)
            except Exception:
                pass
