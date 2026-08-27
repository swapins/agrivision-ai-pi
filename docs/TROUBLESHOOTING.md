# Troubleshooting

| Problem | Check |
|---|---|
| App starts in simulation | Confirm `AGRIVISION_SIMULATION=1` was intended; unset it for real hardware |
| App fails outside simulation | Run `python3 scripts/fetch_models.py`; confirm `models/plant_health_edgetpu.tflite` and `models/labels.txt` exist |
| Coral not detected | Standard runtime installed, accelerator reconnected, USB 3 port, `lsusb`, official PyCoral example |
| PyCoral unavailable | Raspberry Pi OS/Python compatibility; `scripts/install_coral.sh` warns when Python is outside the common PyCoral range |
| Coral smoke test allocates but skips inference | Supply a real image path: `python3 scripts/coral_smoke_test.py path/to/leaf.jpg` |
| Camera missing | Power off, reseat CSI ribbon, enable camera interface if needed, test Picamera2 separately |
| Pi reboots when pump starts | Separate pump supply, MOSFET wiring, common control ground, motor suppression |
| Pump does not start | Check dry threshold, cooldown, `pump.max_run_seconds`, MOSFET signal pin, and dashboard pump reason |
| Pump should stop now | Press dashboard **PUMP OFF** or stop the service; the app also turns pump off during shutdown |
| Soil percentage reversed | Recheck `dry_raw` / `wet_raw`; formula supports either sensor direction |
| Model always predicts one class | Check fixed lighting, label order, manifest lineage, and whether the compiled Edge TPU model is the published artifact |
| Result changes with background | Use fixed scan background and treat PlantVillage metrics as controlled-dataset results |
| Dashboard not opening | Check `python -m agrivision.app`, then `/health`, then port 5000/firewall |
