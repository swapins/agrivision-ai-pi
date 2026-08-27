# Raspberry Pi Deployment

Target hardware: Raspberry Pi 4 + Google Coral USB Accelerator.

## First Install

```bash
git clone https://github.com/swapins/agrivision-ai-pi.git
cd agrivision-ai-pi
sudo bash scripts/install_pi.sh
sudo bash scripts/install_coral.sh
cp config.example.yaml config.yaml
python3 scripts/fetch_models.py
```

Edit `config.yaml` only for real GPIO, pump thresholds, soil calibration values, and server settings. Runtime files such as `config.yaml`, `captures/latest.jpg`, `.venv`, and downloaded model artifacts are not committed.

## Soil Calibration

```bash
python3 scripts/calibrate_soil.py
```

Record the actual dry and wet readings in `config.yaml`.

## Hardware Validation

Run the safe self-test first. It does not pulse the pump unless requested.

```bash
python3 scripts/hardware_selftest.py
```

After checking MOSFET wiring, separate pump power, tubing, and water isolation:

```bash
python3 scripts/hardware_selftest.py --pump
```

Validate the Coral model with a real image:

```bash
python3 scripts/coral_smoke_test.py path/to/leaf.jpg
```

## Start Dashboard

```bash
./scripts/run.sh
```

Open:

```text
http://127.0.0.1:5000
```

or:

```text
http://<PI-IP>:5000
```

## Service Mode

If the Pi user or checkout path differs from `/home/pi/agrivision-ai-pi`, edit `scripts/systemd/agrivision.service` first.

```bash
sudo bash scripts/install_service.sh
journalctl -u agrivision.service -f
```

## Simulation Mode

On a development PC:

```bash
AGRIVISION_SIMULATION=1 python -m agrivision.app
```

Simulation mode is controlled by `AGRIVISION_SIMULATION=1`. Outside simulation mode, startup requires the deployed Edge TPU model and labels.
