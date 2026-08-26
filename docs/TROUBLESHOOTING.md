# Troubleshooting

| Problem | Check |
|---|---|
| Coral not detected | Standard runtime installed, accelerator reconnected, USB 3 port, official example passes |
| PyCoral unavailable | OS/Python compatibility; use the compatibility route documented for the school build |
| Camera missing | Power off, reseat CSI ribbon, list cameras, test Picamera2 separately |
| Pi reboots when pump starts | Separate pump supply, MOSFET wiring, common control ground, motor suppression |
| Soil percentage reversed | Recheck `dry_raw` / `wet_raw`; formula supports either sensor direction |
| Pump chatters | Verify hysteresis and cooldown settings |
| Model always predicts one class | Dataset balance, preprocessing, label order, quantization, lighting |
| Result changes with background | Use fixed scan background and diversify training data |
| Dashboard not opening | Check `python -m agrivision.app`, then `/health`, then port 5000/firewall |
| Model load fails | Confirm `_edgetpu.tflite` is actually compiler output and labels match |
