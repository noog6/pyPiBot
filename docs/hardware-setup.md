# Hardware Setup

This guide captures the baseline hardware used to assemble Theo and the
mechanical wiring notes for the initial prototype. Your build can vary based on
available parts, but these are the pieces used during testing.

## Embedded Hardware Requirements

- **Embedded controller:** Raspberry Pi Zero W (or newer equivalent)
- **Audio controller:** RaspiAudio MIC v2/v3 HAT
- **Servo controller:** Waveshare PCA9685 Servo Driver HAT
- **Sensor board:** Waveshare Sense HAT B
- **Camera:** 5MP OV5647 Mini Raspberry Pi Camera
- **Servos:** Two 9g hobby servos (pan/tilt)

## Assembly Notes

Required tools: soldering iron, wire cutters, and standard electronics tools.

1. **Raspberry Pi Zero W**
   - Solder longer wire-wrapping pins onto the headerless board.
2. **Sense HAT B modifications**
   - Fold I2C pins down 90 degrees so they sit closer to the board (without
     touching the Pi).
   - Desolder the ADC sensor pins from the board.
   - Solder a battery sensor circuit (resistor bridge) so the low-voltage ADC
     can measure the higher-voltage LiPo 2S battery range (7V–8.4V).
3. **Stacking order**
   - Stack the Sense HAT B onto the newly installed longer pins of the
     Raspberry Pi Zero W.
   - Plug the RaspiAudio MIC board into the Sense HAT pins.
   - Plug the Servo Driver HAT into the RaspiAudio pins.
   - Connect the camera ribbon cable to the Raspberry Pi camera connector.

If your HATs differ, confirm I2C addresses and power requirements before
powering the stack.
