from __future__ import annotations

import atexit
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, send_file

from .config import load_config, validate_config
from .controller import AgriVisionController
from .hardware import make_hardware
from .inference import make_inferencer
from .state import RuntimeState, StateStore


def create_app() -> Flask:
    cfg = load_config(os.environ.get("AGRIVISION_CONFIG"))
    simulation = os.environ.get("AGRIVISION_SIMULATION", "0").lower() in {"1", "true", "yes"}
    validate_config(cfg, simulation)

    hardware = make_hardware(cfg, simulation)
    inferencer = make_inferencer(cfg, simulation)
    store = StateStore(
        RuntimeState(
            simulation=simulation,
            backend=inferencer.backend_name,
            coral_status=inferencer.coral_status,
            message="Simulation mode active" if simulation else "Hardware mode active",
        )
    )
    controller = AgriVisionController(cfg, hardware, inferencer, store, simulation=simulation)
    controller.start_background()

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["AGRIVISION_CONTROLLER"] = controller
    app.config["AGRIVISION_STATE"] = store
    app.config["AGRIVISION_CONFIG"] = cfg

    @app.get("/")
    def index():
        project = cfg.section("project")
        return render_template("index.html", project=project, simulation=simulation)

    @app.get("/api/status")
    def status():
        return jsonify(store.snapshot())

    @app.post("/api/scan")
    def scan():
        try:
            return jsonify(controller.scan_leaf())
        except Exception as exc:
            store.update(message=f"AI scan error: {exc}")
            return jsonify({"error": str(exc)}), 500

    @app.post("/api/pump/off")
    def pump_off():
        return jsonify(controller.emergency_pump_off())

    @app.get("/latest.jpg")
    def latest():
        path = controller.capture_path
        if not path.exists():
            return ("No image yet", 404)
        return send_file(path, mimetype="image/jpeg", max_age=0)

    @app.get("/health")
    def health():
        snapshot = store.snapshot()
        return jsonify(
            {
                "status": "ok",
                "simulation": simulation,
                "backend": snapshot["backend"],
                "coral_status": snapshot["coral_status"],
            }
        )

    atexit.register(controller.close)
    return app


def main() -> None:
    app = create_app()
    cfg = app.config["AGRIVISION_CONFIG"]
    server = cfg.section("server")
    app.run(
        host=str(server.get("host", "0.0.0.0")),
        port=int(server.get("port", 5000)),
        threaded=True,
        debug=False,
    )


if __name__ == "__main__":
    main()
