# Raspberry Pi deployment

1. Flash the compatibility-first Raspberry Pi OS image selected for the Coral/PyCoral stack.
2. Install Pi packages with `sudo bash scripts/install_pi.sh`.
3. Enable I²C in `raspi-config` and verify `i2cdetect -y 1`.
4. Install the Coral runtime with `sudo bash scripts/install_coral.sh`.
5. Test the Coral with Google's official PyCoral example before using the custom model.
6. Copy `config.example.yaml` to `config.yaml` and enter actual dry/wet calibration values.
7. Copy model artifacts into `models/`.
8. Run `scripts/hardware_selftest.py` with the water tank almost empty first.
9. Run `scripts/coral_smoke_test.py <known-leaf-image.jpg>`.
10. Start the dashboard with `scripts/run.sh`.

For exhibition auto-start, edit `scripts/systemd/agrivision.service` if the Pi username/path differs, then install with `sudo scripts/install_service.sh`.
