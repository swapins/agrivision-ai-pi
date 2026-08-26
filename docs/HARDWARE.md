# Hardware and wiring

## Architecture

```text
Camera Module 3 -> Raspberry Pi 4 -> Coral USB Accelerator -> prediction
Soil sensor -> ADS1115 -> Raspberry Pi -> safety logic -> MOSFET -> 5 V pump
BME280 -> Raspberry Pi -> dashboard
Raspberry Pi -> LEDs / buzzer / local Flask dashboard
```

## BCM GPIO map

| Function | Pin / bus |
|---|---|
| ADS1115 SDA | GPIO2 / SDA |
| ADS1115 SCL | GPIO3 / SCL |
| Soil analog out | ADS1115 A0 |
| BME280 SDA/SCL | shared I²C bus |
| Pump MOSFET signal | GPIO17 |
| Green LED | GPIO22 through 220–330 Ω |
| Yellow LED | GPIO23 through 220–330 Ω |
| Red LED | GPIO24 through 220–330 Ω |
| Buzzer signal | GPIO27 |
| Camera | CSI |
| Coral | USB 3 |

## Pump wiring safety

The GPIO controls only the MOSFET input. The pump is powered from a separate 5 V source sized for the pump. Join the low-voltage control ground as required by the MOSFET module. Use flyback/suppression appropriate to the pump/module. Never route pump current through a Raspberry Pi GPIO pin.

## Physical model

Use a roughly 60 × 45 cm base. Put water and plants on one side and electronics on a raised dry platform on the other. Keep the reservoir below or beside the plant area, never above the Pi. Fix the camera distance and use a plain leaf-scan background for repeatability.
