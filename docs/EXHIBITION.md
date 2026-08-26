# 90-second exhibition script

1. “This is AgriVision AI, an edge-AI smart agriculture model using Raspberry Pi and a Coral Edge TPU.”
2. Point to the soil sensor: “It measures soil moisture and can automatically irrigate the plant.”
3. Move the sensor into the prepared dry-soil cup and show the moisture fall and limited 5 V pump cycle.
4. Put a healthy leaf/test card in the fixed scan area and press **SCAN LEAF**. Show the green LED and confidence.
5. Put a diseased/problem leaf/test card in the scan area and scan again. Show the red LED, brief buzzer and alert.
6. Point to the Coral: “The neural-network inference happens locally. The image does not need to be sent to a cloud server.”
7. Close: “The same idea can be extended to crop monitoring, water management and early field inspection.”

## Viva essentials

- **Edge AI:** inference close to the data source instead of sending every input to a remote cloud.
- **ADC:** Raspberry Pi GPIO cannot directly measure analog voltage; ADS1115 converts the soil sensor signal.
- **MOSFET:** lets a small GPIO control signal switch a separately powered pump.
- **Inference:** using an already trained model to predict on new input.
- **Quantization:** lower-precision integer representation that enables efficient edge inference.
- **Why uncertainty:** low-confidence predictions are not presented as facts.
