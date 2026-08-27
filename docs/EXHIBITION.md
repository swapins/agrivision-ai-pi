# 90-Second Exhibition Script

1. "This is AgriVision AI, an edge-AI smart agriculture model using Raspberry Pi 4 and a Coral Edge TPU."
2. Point to the soil sensor: "It measures soil moisture and can automatically irrigate the plant through a safe pump driver."
3. Move the sensor into the prepared dry-soil cup and show moisture falling and the bounded 5 V pump cycle.
4. Put a healthy leaf or prepared test card in the fixed scan area and press **SCAN LEAF**. Show the green LED and dashboard confidence.
5. Put a problem leaf or prepared test card in the scan area and scan again. Show the red LED, brief buzzer, and cautious warning.
6. Point to the Coral: "The neural-network inference happens locally. The image does not need to be sent to a cloud server."
7. Close: "The validated INT8 model reached 98.58% held-out test accuracy on the PlantVillage dataset. This is a school prototype, not professional field diagnosis."

## Viva Essentials

- **Edge AI:** inference close to the data source instead of sending every input to a remote cloud.
- **ADC:** Raspberry Pi GPIO cannot directly measure analog voltage; ADS1115 converts the soil sensor signal.
- **MOSFET:** lets a small GPIO control signal switch a separately powered pump.
- **Inference:** using an already trained model to predict on a new input.
- **Quantization:** lower-precision integer representation that enables efficient edge inference.
- **Edge TPU compilation:** converts a compatible INT8 TFLite model into an artifact that runs on the Coral accelerator.
- **Why uncertainty:** low-confidence predictions are not presented as facts.

## Simulation Rule

If `AGRIVISION_SIMULATION=1` is used, say clearly that the displayed sensor, camera, and inference values are simulated. Do not present them as real Raspberry Pi or Coral output.
