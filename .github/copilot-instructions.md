# GitHub Copilot Instructions — System_Conveyor

Purpose: quick, actionable guidance for an AI coding agent to be productive here.

1) Big picture (architecture)
- Flow: camera → YOLO detection → ROI preprocessing → MobileNet TFLite classification → servo/motor actuation. Entrypoints: `fruit_sorter.py` (CLI) and `run_web.py` → `web/app.py` (Flask + SocketIO).
- Major components: `hardware/` (camera, motor, servo, `gpio_config.py`, `conveyor.py`), `ai_models/` (detection & classifier), `training/` (retraining/export), `web/` (UI & streaming), `utils/` (central `config.py`, `logger.py`).

2) Quick run & test commands
- One-shot install (Raspberry Pi): `chmod +x install.sh && ./install.sh` (creates venv, installs packages).
- Activate venv: `source venv/bin/activate`.
- Run main system (CLI): `python fruit_sorter.py`.
- Run web UI: `python run_web.py` (default port 5000).
- Useful module tests: `python hardware/camera.py`, `python hardware/servo_control.py`, `python ai_models/mobilenet_classifier.py`.

3) Models & training
- Model files: `models/` — `yolov8n_fruit.pt`, `mobilenet_classifier.tflite` (production classifier).
- Training: YOLO under `training/yolo/`; MobileNet under `training/mobilenet/`. Export TFLite with `training/mobilenet/export_tflite.py`.
- Important: MobileNet expects inputs normalized to [-1,1]; preprocessing pipeline lives in `ai_models/preprocessing.py` and conversion happens in `ai_models/mobilenet_classifier.py`.

4) Conventions & important files
- Central config: `utils/config.py` — change thresholds, `FRUIT_TRAVEL_TIME`, `SERVO_MOVE_DELAY`, and model paths here.
- Hardware mapping: `hardware/gpio_config.py` — do not change defaults without documenting wiring changes.
- Logging: use `utils/logger.py` and `SystemLogger` for events, detection, and stats; logs are in `logs/`.

5) Simulation & development patterns
- The code supports simulation mode (when `picamera2` or `RPi.GPIO` are missing) — keep this behavior for local dev/CI.
- Self-test helpers: `ConveyorSystem.run_test_cycle()`, `ServoControl.test_movement()`, `MotorControl.test()`; use them in PR validation.

6) When editing AI pipelines
- Update implementation in `ai_models/*`, any preprocessing in `training/mobilenet/prepare_data.py`, and the TFLite export script if shapes or normalization change.
- Run the classifier locally (`python ai_models/mobilenet_classifier.py`) and verify outputs and thresholds in `utils/config.py`.

7) PR guidance (practical checklist)
- Preserve simulation fallback for non-Pi CI/dev.
- Add or update a short self-test (module `__main__`) when hardware or model behavior changes.
- Update `utils/config.py` defaults and document config changes in `docs/SYSTEM_SETUP.md` or `README.md`.

8) Integration points to watch
- Web UI: `web/app.py` (SocketIO events `stats_update`, `request_stats`, and `/video_feed`).
- Hardware control: `hardware/servo_control.py` and `hardware/motor_control.py` are the single place for actuation logic.

If you want this shortened further, expanded with example PR descriptions, or turned into checklists for CI, tell me which area to adjust.
