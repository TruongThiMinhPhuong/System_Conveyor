# Hardware Setup Guide

Complete guide for assembling and wiring the AI Fruit Sorting Conveyor System.

## Safety First ⚠️

- **Power off** all devices before wiring
- Check voltage ratings before connecting
- Use appropriate power supplies
- Never connect motors directly to Raspberry Pi GPIO
- Double-check polarity (+/-)

## Components Needed

### Main Components
- ✅ Raspberry Pi 4 (8GB RAM)
- ✅ Camera Module v2 (5MP 1080p)
- ✅ MicroSD Card (32GB+, Class 10)
- ✅ Power Supply for RPi (5V 3A USB-C)

### Motors & Control
- ✅ Servo Motor: MG996R
- ✅ Motor Driver: L298N Module
- ✅ Conveyor Motor: JGB37-545 with encoder
- ✅ Power Supply for Motors (6-12V, 2A+)

### Miscellaneous
- ✅ Breadboard or PCB
- ✅ Jumper wires (M-M, M-F)
- ✅ Mounting hardware
- ✅ Conveyor belt structure

## Raspberry Pi 4 Pinout Reference

```
      3.3V  1 ◉ ◉  2   5V
     GPIO2  3 ◉ ◉  4   5V
     GPIO3  5 ◉ ◉  6   GND
     GPIO4  7 ◉ ◉  8   GPIO14 (UART TX)
       GND  9 ◉ ◉ 10   GPIO15 (UART RX)
    GPIO17 11 ◉ ◉ 12   GPIO18 (PWM0) ← SERVO
    GPIO27 13 ◉ ◉ 14   GND
    GPIO22 15 ◉ ◉ 16   GPIO23
      3.3V 17 ◉ ◉ 18   GPIO24
    GPIO10 19 ◉ ◉ 20   GND
     GPIO9 21 ◉ ◉ 22   GPIO25
    GPIO11 23 ◉ ◉ 24   GPIO8
       GND 25 ◉ ◉ 26   GPIO7
     GPIO0 27 ◉ ◉ 28   GPIO1
     GPIO5 29 ◉ ◉ 30   GND
     GPIO6 31 ◉ ◉ 32   GPIO12
    GPIO13 33 ◉ ◉ 34   GND
    GPIO19 35 ◉ ◉ 36   GPIO16
    GPIO26 37 ◉ ◉ 38   GPIO20
       GND 39 ◉ ◉ 40   GPIO21
```

## Step 1: Camera Module Installation

### Installation Steps
1. **Power off** Raspberry Pi
2. Locate the **CSI camera connector** (between HDMI and audio jack)
3. Gently pull up the plastic clip
4. Insert **ribbon cable** with blue side facing audio jack
5. Push down the clip to secure
6. **Do not force** - ribbon should slide in easily

### Testing Camera
```bash
# After OS installation
libcamera-hello
# Should show camera preview
```

## Step 2: Servo Motor (MG996R) Wiring

### Servo Specifications
- Operating Voltage: 4.8-7.2V
- Stall Torque: 11 kg⋅cm (4.8V), 13 kg⋅cm (6V)
- Control Signal: PWM (50Hz)

### Wiring Connections

```
MG996R Servo Wire Colors:
├─ Brown  → Ground (GND)
├─ Red    → Power (6V external)
└─ Orange → Signal (GPIO 18)
```

**IMPORTANT**: Use **external 6V power supply** for servo (NOT from RPi)

### Connection Diagram
```
External 6V Supply
  ├─ (+) → Servo Red Wire
  └─ (-) → Common GND with RPi

Raspberry Pi
  ├─ GPIO 18 (Pin 12) → Servo Orange Wire
  └─ GND (Pin 9)      → Common GND
```

## Step 3: L298N Motor Driver Wiring

### L298N Specifications
- Motor Supply: 5-35V
- Logic Supply: 5V
- Max Current per Channel: 2A
- Onboard 5V regulator (remove jumper if using >12V)

### Pin Connections

#### Power Connections
```
L298N → Power
├─ +12V  → 12V Power Supply (+)
├─ GND   → Common Ground
└─ +5V   → No connection (using RPi 5V)
```

#### RPi GPIO → L298N Control
```
Raspberry Pi                L298N
├─ GPIO 22 (Pin 15)    →   ENA (Enable A - PWM)
├─ GPIO 23 (Pin 16)    →   IN1 (Motor A Direction)
├─ GPIO 24 (Pin 18)    →   IN2 (Motor A Direction)
└─ GND (Pin 20)        →   GND
```

#### Motor Connections
```
L298N                      Conveyor Motor
├─ OUT1                →   Motor Wire 1
└─ OUT2                →   Motor Wire 2
```

### Connection Diagram
```
[12V PSU]              [L298N Module]              [Conveyor Motor]
   |                         |                           |
   +12V ────────────────▶ +12V                          |
   GND  ────────────────▶ GND                           |
                            |                           |
                          OUT1 ────────────────────▶ Wire 1
                          OUT2 ────────────────────▶ Wire 2
                            |
                          ENA  ◀──── GPIO 22
                          IN1  ◀──── GPIO 23
                          IN2  ◀──── GPIO 24
                          GND  ◀──── RPi GND
```

## Step 4: Complete Wiring Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Raspberry Pi 4 (8GB)                     │
│                                                             │
│  [Camera CSI] ← Camera Module                              │
│                                                             │
│  GPIO 18 (Pin 12) ──────────┐                             │
│  GND (Pin 9) ────────────┐  │                             │
│  GPIO 22-25 ──────────┐  │  │                             │
│  GND (Pin 20) ─────┐  │  │  │                             │
│  5V (Pin 2,4) ─┐   │  │  │  │                             │
└────────────────┼───┼──┼──┼──┼─────────────────────────────┘
                 │   │  │  │  │
                 │   │  │  │  └─────▶ Servo Signal (Orange)
                 │   │  │  └────────▶ Common GND
                 │   │  │
                 │   │  └──────────▶ L298N Control Pins
                 │   └─────────────▶ L298N GND
                 │
                 ▼
         ┌───────────────┐
         │   Servo PSU   │
         │    (6V 2A)    │
         └───────┬───────┘
                 │
                 ├──▶ Servo VCC (Red)
                 └──▶ Common GND (Brown)

┌─────────────────────────────────────┐       ┌──────────────┐
│        L298N Motor Driver           │       │ Conveyor PSU │
│                                     │       │  (12V 2A+)   │
│  +12V ◀─────────────────────────────┼───────┤   +12V       │
│  GND  ◀─────────────────────────────┼───────┤   GND        │
│                                     │       └──────────────┘
│  OUT1 ────────────┐                │
│  OUT2 ────────────┼───────────────▶│  Conveyor Motor
│                   └────────────────▶│  (JGB37-545)
└─────────────────────────────────────┘
```

## Step 5: Power Supply Setup

### Power Requirements
1. **Raspberry Pi**: 5V 3A USB-C (official adapter recommended)
2. **Servo Motor**: 6V 2A (can use 4x AA batteries or regulated PSU)
3. **Conveyor Motor**: 12V 2A+ (depends on motor spec)

### Ground Connection ⚠️
**CRITICAL**: All power supplies must share a **common ground**

```
RPi GND ←──┬──→ Servo GND ←──┬──→ L298N GND ←──→ Motor PSU GND
           │                 │
    All connected together (Common GND)
```

## Step 6: Assembly & Mounting

### Mechanical Assembly
1. Mount Raspberry Pi on stable platform
2. Position camera for good fruit viewing angle
3. Mount servo for sorting gate mechanism
4. Install conveyor belt structure
5. Attach motor to conveyor drive
6. Secure all wiring with cable ties

### Camera Positioning Tips
- **Height**: 20-30cm above conveyor belt
- **Angle**: Perpendicular to belt for best detection
- **Lighting**: Ensure even lighting (LED strip recommended)
- **Focus**: Adjust camera focus for fruit distance

## Step 7: Testing Each Component

### Test Camera
```bash
python hardware/camera.py
```

### Test Servo
```bash
python hardware/servo_control.py
```

### Test Motor
```bash
python hardware/motor_control.py
```

### Test Complete System
```bash
python hardware/conveyor.py
```

## Troubleshooting

### Camera Not Detected
- Check ribbon cable connection
- Enable camera in raspi-config: `sudo raspi-config`
- Reboot Raspberry Pi

### Servo Not Moving
- Check power supply (needs external 6V)
- Verify GPIO 18 connection
- Check common ground connection

### Motor Not Running
- Verify L298N power connections
- Check motor wire polarity
- Test with different speed values

### GPIO Permission Errors
```bash
sudo usermod -a -G gpio $USER
# Logout and login again
```

## Safety Checklist ✅

Before powering on:
- [ ] All connections double-checked
- [ ] No bare wires touching
- [ ] Correct voltage for each component
- [ ] Common ground established
- [ ] Motors secured (won't fly off)
- [ ] Camera ribbon properly inserted
- [ ] Power supplies rated correctly

## Next Steps

Once hardware is assembled and tested:
1. ✅ Proceed to [Software Setup](software_setup.md)
2. ✅ Install Raspberry Pi OS
3. ✅ Run installation script
4. ✅ Train AI models
5. ✅ Deploy and test system
