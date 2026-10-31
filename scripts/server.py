from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "src")

from road_accident_risk.predictor import AccidentRiskPredictor


predictor = AccidentRiskPredictor()


class PredictionHandler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send_json(204, {})

    def do_GET(self) -> None:
        if self.path == "/":
            self._send_json(200, {"service": "road-accident-risk", "status": "ready"})
        else:
            self._send_json(404, {"detail": "Not found"})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._send_json(404, {"detail": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            features = json.loads(self.rfile.read(length))
            prediction, valid, message = predictor.predict_with_validation(features)
            if not valid:
                self._send_json(422, {"detail": message})
                return
            self._send_json(200, {"accident_risk": float(prediction), "source": "model"})
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            self._send_json(400, {"detail": f"Invalid JSON request: {error}"})
        except Exception as error:
            self._send_json(500, {"detail": f"Prediction failed: {error}"})

    def log_message(self, format: str, *args: object) -> None:
        print(format % args)


if __name__ == "__main__":
    host = "0.0.0.0"
    port = 8000
    print(f"RoadGuard API listening on http://{host}:{port}")
    ThreadingHTTPServer((host, port), PredictionHandler).serve_forever()
