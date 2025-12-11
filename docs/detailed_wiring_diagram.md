# Sơ Đồ Kết Nối Chi Tiết - Hệ Thống Phân Loại Hoa Quả AI

## 📊 Tổng Quan Hệ Thống

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NGUỒN ĐIỆN CHÍNH                                │
│                                                                         │
│    ╔═══════════════════════════════════════╗                          │
│    ║  Adapter 12V DC (5A)                  ║                          │
│    ║  ┌─────────┐                          ║                          │
│    ║  │ AC 220V │──▶ Transformer ──▶ DC    ║                          │
│    ║  └─────────┘       ↓                  ║                          │
│    ║              Output: 12V ⎓            ║                          │
│    ╚═══════════════╦═══════════╦═══════════╝                          │
│                    │           │                                        │
│                 +12V          GND                                       │
│                    │           │                                        │
└────────────────────┼───────────┼────────────────────────────────────────┘
                     │           │
                     │           │
    ┌────────────────┼───────────┼────────────────────┐
    │                ▼           ▼                    │
    │        ╔═══════════════════════════╗            │
    │        ║ LM2596 Buck Converter     ║            │
    │        ║                           ║            │
    │   IN+  ║  ┌─────────────────┐     ║  OUT+      │
    │  ◀─────╫──┤ 12V → 6V        │     ╠────▶ 6V    │
    │        ║  │ Điều chỉnh: ⚙️   │     ║            │
    │   IN-  ║  └─────────────────┘     ║  OUT-      │
    │  ◀─────╫──────────────────────────╠────▶ GND   │
    │        ╚═══════════════════════════╝            │
    └─────────────────────────────────────────────────┘
                     │                    │
                     │                    │
                     │                    ▼
                     │            ╔════════════════════╗
                     │            ║  Servo MG996R      ║
                     │            ║                    ║
                     │            ║  Dây ĐỎ   ◀─ 6V   ║
                     │            ║  Dây NÂU  ◀─ GND  ║
                     │            ║  Dây CAM  ◀─ GPIO ║
                     │            ╚════════════════════╝
                     │                          │
                     │                     GPIO 18 từ Pi
                     │
                     ▼
            ╔═══════════════════════════════════════╗
            ║  L298N Motor Driver                   ║
            ║  ┌──────────────────────────────┐    ║
            ║  │ H-Bridge Logic               │    ║
            ║  │                              │    ║
  +12V ────▶║  │  PWM Speed Control           │    ║
   GND ────▶║  │  Direction Control           │    ║
            ║  └──────────────────────────────┘    ║
            ║                                      ║
  GPIO ────▶║  ENA, IN1, IN2                      ║
   22-24    ║                                      ║
            ║  OUT1 ───┐                          ║
            ║  OUT2 ───┴──▶ Motor JGB37-545       ║
            ╚═══════════════════════════════════════╝
                                    │
                           ╔════════▼═══════════╗
                           ║ Động Cơ Băng      ║
                           ║ Chuyền JGB37-545  ║
                           ║                   ║
                           ║ 12V DC Motor      ║
                           ║ With Encoder      ║
                           ╚═══════════════════╝

┌─────────────────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI 4 (Nguồn Riêng)                         │
│                                                                         │
│    ╔═══════════════════════════════════════╗                          │
│    ║  USB-C Power Adapter 5V 3A            ║                          │
│    ║  ┌─────────┐                          ║                          │
│    ║  │ AC 220V │──▶ USB Adapter ──▶ 5V    ║                          │
│    ║  └─────────┘                          ║                          │
│    ╚═══════════════╦═══════════════════════╝                          │
│                    │                                                    │
│                    ▼ USB-C                                              │
│           ╔════════════════════╗                                       │
│           ║  Raspberry Pi 4    ║                                       │
│           ║  (8GB RAM)         ║                                       │
│           ║                    ║                                       │
│           ║  GPIO 18 ──────────╫──▶ Servo Signal                      │
│           ║  GPIO 22 ──────────╫──▶ L298N ENA                         │
│           ║  GPIO 23 ──────────╫──▶ L298N IN1                         │
│           ║  GPIO 24 ──────────╫──▶ L298N IN2                         │
│           ║  GND ──────────────╫──▶ COMMON GND ⚡                     │
│           ╚════════════════════╝                                       │
└─────────────────────────────────────────────────────────────────────────┘

     ⚡ COMMON GROUND - ĐIỂM NỐI CHUNG QUAN TRỌNG ⚡
═══════════════════════════════════════════════════════════════════════════
  Pi GND ──┬── Servo GND ──┬── L298N GND ──┬── Buck GND ──┬── 12V Adapter
           │               │                │              │      GND
           └───────────────┴────────────────┴──────────────┘
                    TẤT CẢ NỐI VỀ 1 ĐIỂM!
```

---

## 🔌 Chi Tiết Kết Nối Từng Module

### 1️⃣ Buck Converter LM2596 (12V → 6V)

```
     ┌─────────────────────────────────────┐
     │   LM2596 DC-DC Buck Converter       │
     │                                     │
     │   INPUT                OUTPUT      │
     │   ┌────┐              ┌────┐       │
     │   │IN+ │◀─(+12V)      │OUT+│──▶(+6V)─┐
     │   └────┘              └────┘         │
     │                                      │
     │   ┌────┐              ┌────┐         │
     │   │IN- │◀─(GND)       │OUT-│──▶(GND)─┤
     │   └────┘              └────┘         │
     │                                      │
     │        [⚙️ Adjustment Screw]         │
     │      (Vặn để điều chỉnh 6V)          │
     └─────────────────────────────────────┘
              │                      │
              │                      ├──▶ Servo VCC (Đỏ)
              │                      └──▶ Servo GND (Nâu)
              │
              └── Từ nguồn 12V chính
```

**Hướng dẫn điều chỉnh:**
1. Kết nối INPUT với 12V
2. Để OUTPUT không nối gì
3. Dùng đồng hồ đo điện áp ở OUT+ và OUT-
4. Vặn vít ⚙️ từ từ cho đến khi đồng hồ hiển thị **6.0V**
5. Sau đó mới kết nối servo

---

### 2️⃣ L298N Motor Driver

```
     ┌───────────────────────────────────────────────┐
     │        L298N H-Bridge Motor Driver            │
     │                                               │
     │  POWER INPUT          CONTROL INPUT          │
     │  ┌────────┐           ┌─────────────┐        │
     │  │ +12V   │◀─(Red)    │ ENA  ◀ GPIO22│       │
     │  │  VIN   │           │ IN1  ◀ GPIO23│       │
     │  └────────┘           │ IN2  ◀ GPIO24│       │
     │                       │ GND  ◀ Pi GND│       │
     │  ┌────────┐           └─────────────┘        │
     │  │  GND   │◀─(Black)                         │
     │  └────────┘                                   │
     │                       MOTOR OUTPUT            │
     │  [X] +5V EN           ┌────┐  ┌────┐         │
     │  (Tháo jumper này)    │OUT1│  │OUT2│         │
     │                       └──┬─┘  └──┬─┘         │
     │                          └───┬───┘           │
     │                              │               │
     └──────────────────────────────┼───────────────┘
                                    │
                            ╔═══════▼═══════╗
                            ║ Motor         ║
                            ║ JGB37-545     ║
                            ║               ║
                            ║ Dây 1 ◀ OUT1 ║
                            ║ Dây 2 ◀ OUT2 ║
                            ╚═══════════════╝
```

**Chú ý quan trọng:**
- ⚠️ **THÁO jumper** giữa +12V và +5V (nếu có)
- Không dùng regulator 5V onboard của L298N
- Motor có thể đấu ngược (chỉ đảo chiều quay)

---

### 3️⃣ Servo MG996R

```
     ╔═══════════════════════════════╗
     ║      Servo MG996R             ║
     ║                               ║
     ║   ┌──────────────────────┐    ║
     ║   │                      │    ║
     ║   │   Motor & Gears      │    ║
     ║   │                      │    ║
     ║   └──────────────────────┘    ║
     ║                               ║
     ║   Dây ra (3 dây):             ║
     ║                               ║
     ║   🟤 NÂU (Brown)   ──▶ GND    ║──▶ Common GND
     ║                               ║
     ║   🔴 ĐỎ (Red)      ──▶ VCC    ║──▶ 6V từ Buck
     ║                               ║
     ║   🟠 CAM (Orange)  ──▶ Signal ║──▶ GPIO 18 (Pi)
     ║                               ║
     ╚═══════════════════════════════╝
```

**Thông số:**
- Điện áp: 4.8V - 7.2V (tối ưu 6V)
- Dòng: 0.5A - 1.5A (khi chuyển động)
- PWM: 50Hz (chu kỳ 20ms)

---

### 4️⃣ Raspberry Pi GPIO Pinout

```
    Raspberry Pi 4 - GPIO Header (40 pins)
    
    3.3V  ● ● 5V      ← Không dùng cho motor/servo
    G2    ● ● 5V
    G3    ● ● GND     ← GND (Pin 6,9,14,20,25,30,34,39)
    G4    ● ● G14
    GND   ● ● G15
    G17   ● ● G18     ← SERVO SIGNAL (PWM) 🟠
    G27   ● ● GND
    G22   ● ● G23     ← MOTOR ENA (PWM) 🔵
    3.3V  ● ● G24     ← MOTOR IN1 🔵
    G10   ● ● GND     ← MOTOR IN2 🔵
    G9    ● ● G25
    G11   ● ● G8
    GND   ● ● G7
    ...
    
    Kết nối sử dụng:
    - Pin 12 (GPIO 18) → Servo Signal
    - Pin 15 (GPIO 22) → L298N ENA
    - Pin 16 (GPIO 23) → L298N IN1
    - Pin 18 (GPIO 24) → L298N IN2
    - Pin 20 (GND)     → Common GND
```

---

## 🔋 Sơ Đồ Dòng Điện

```
NGUỒN 12V (5A)
     │
     ├──────────────────┬──────────────────┐
     │                  │                  │
  +12V                +12V              +12V
     │                  │                  │
     ▼                  ▼                  │
 ┌────────┐      ┌──────────┐             │
 │ Buck   │      │  L298N   │             │
 │Convert │      │  Logic   │             │
 └───┬────┘      └────┬─────┘             │
     │                │                   │
   +6V              Motor              Reserve
     │              Drive                 │
     ▼                ▼                   │
 ┌────────┐      ┌────────┐              │
 │ Servo  │      │ Motor  │              │
 │ 0.5-1A │      │ 1-2A   │              │
 └────────┘      └────────┘              │
     │                │                  │
     └────────────────┴──────────────────┘
                      │
                    GND → Common Ground
```

**Tổng công suất:**
- Servo: 6V × 1A = 6W
- Motor: 12V × 2A = 24W
- Logic: 12V × 0.1A = 1.2W
- **Tổng: ~31W** → Nguồn 12V 5A (60W) là đủ dư

---

## 🛠️ Checklist Lắp Đặt

### Bước 1: Chuẩn Bị
- [ ] Breadboard hoặc PCB
- [ ] Terminal blocks (cục nối dây)
- [ ] Dây đỏ/đen 18-20 AWG
- [ ] Jumper wires đủ màu
- [ ] Đồng hồ vạn năng

### Bước 2: Điều Chỉnh Buck Converter
- [ ] Kết nối INPUT 12V
- [ ] Đo OUTPUT trước khi nối servo
- [ ] Vặn chỉnh về **6.0V chính xác**
- [ ] Ghi nhãn OUT+ và OUT-

### Bước 3: Tạo Common Ground
- [ ] Nối tất cả GND vào 1 điểm
- [ ] Kiểm tra liên thông bằng đồng hồ
- [ ] Chú ý: Pi GND + Servo GND + L298N GND + Nguồn GND

### Bước 4: Kết Nối Nguồn
- [ ] 12V vào Buck Converter IN+
- [ ] 12V vào L298N VIN
- [ ] 6V từ Buck ra Servo VCC
- [ ] Kiểm tra cực tính (+/-)

### Bước 5: Kết Nối GPIO
- [ ] GPIO 18 → Servo Signal
- [ ] GPIO 22 → L298N ENA
- [ ] GPIO 23 → L298N IN1
- [ ] GPIO 24 → L298N IN2
- [ ] Pi GND → Common GND

### Bước 6: Kết Nối Motor
- [ ] L298N OUT1 → Motor dây 1
- [ ] L298N OUT2 → Motor dây 2
- [ ] (Có thể đảo nếu quay sai chiều)

### Bước 7: Kiểm Tra Cuối
- [ ] Không có dây chạm ngắn mạch
- [ ] Tất cả GND nối chung
- [ ] Servo có nguồn 6V riêng
- [ ] L298N có nguồn 12V
- [ ] Pi có nguồn USB-C riêng

---

## 🧪 Test Từng Bước

### Test 1: Nguồn Điện
```bash
# Đo điện áp các điểm:
- Buck Output: 6.0V ± 0.1V ✓
- L298N VIN: 12V ± 0.5V ✓
- Common GND: 0V (so với bất kỳ GND nào) ✓
```

### Test 2: Servo
```bash
cd ~/System_Conveyor
source venv/bin/activate
python hardware/servo_control.py
# Servo phải chuyển động Left → Center → Right
```

### Test 3: Motor
```bash
python hardware/motor_control.py
# Motor phải quay forward và reverse
```

### Test 4: Tích Hợp
```bash
python hardware/conveyor.py
# Test toàn bộ hệ thống
```

---

## ⚠️ An Toàn Điện

### TUYỆT ĐỐI TRÁNH:
❌ Lấy nguồn servo từ Pi 5V (sẽ hỏng Pi!)
❌ Quên nối Common GND (tín hiệu không hoạt động)
❌ Đấu ngược cực (+/-) 
❌ Cấp quá 7.2V cho servo (cháy servo)
❌ Chạm tay vào mạch khi đang có điện

### NÊN LÀM:
✅ Đo điện áp trước khi kết nối
✅ Kiểm tra sơ đồ nhiều lần
✅ Test từng module riêng rẽ
✅ Dùng cầu chì bảo vệ
✅ Tắt nguồn khi thay đổi kết nối

---

## 📸 Hình Ảnh Tham Khảo

Xem các hình ảnh được tạo ở trên để có cái nhìn trực quan về:
1. **Complete Wiring Overview** - Tổng quan toàn bộ kết nối
2. **Breadboard Layout** - Bố trí trên breadboard
3. **Step by Step** - Từng bước kết nối

---

## 💡 Mẹo Hữu Ích

1. **Dùng màu dây nhất quán:**
   - Đỏ: +12V, +6V, +5V
   - Đen: GND (tất cả)
   - Vàng/Cam: Signal PWM
   - Xanh/Trắng: Control GPIO

2. **Ghi nhãn mọi thứ:**
   - Dán nhãn trên dây
   - Ghi điện áp trên terminal
   - Đánh số thứ tự kết nối

3. **Căn chỉnh dây gọn gàng:**
   - Dùng dây buộc (cable tie)
   - Tránh dây chéo nhau
   - Để khoảng cách tản nhiệt

4. **Backup plan:**
   - Có thêm jumper dự phòng
   - Có thêm buck converter backup
   - Kiểm tra nhiều lần trước bật nguồn

