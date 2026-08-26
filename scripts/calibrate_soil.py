#!/usr/bin/env python3
"""Interactive ADS1115 soil-sensor calibration utility."""
from statistics import mean
from time import sleep

try:
    import board, busio
    from adafruit_ads1x15.ads1115 import ADS1115
    from adafruit_ads1x15.analog_in import AnalogIn
except ImportError as exc:
    raise SystemExit(f"Pi sensor libraries are required: {exc}")


def sample(channel, n=20):
    values=[]
    for _ in range(n):
        values.append(float(channel.value))
        print(f"  {values[-1]:.0f}")
        sleep(0.15)
    return mean(values)


i2c=busio.I2C(board.SCL, board.SDA)
ads=ADS1115(i2c)
soil=AnalogIn(ads,0)

input("Clean the sensor, hold it in AIR / dry reference, then press Enter...")
dry=sample(soil)
print(f"DRY_RAW average = {dry:.1f}\n")
input("Insert the sensing area into FULLY WET SOIL, then press Enter...")
wet=sample(soil)
print(f"WET_RAW average = {wet:.1f}\n")
print("Copy these into config.yaml:")
print(f"  dry_raw: {round(dry)}")
print(f"  wet_raw: {round(wet)}")
