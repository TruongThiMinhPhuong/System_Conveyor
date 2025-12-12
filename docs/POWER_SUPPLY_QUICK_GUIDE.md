# 🔌 Hướng Dẫn Kết Nối Nguồn Điện 

## ✅ Giải Pháp Tối Ưu 

### Danh Sách Nguồn Điện Cần Có:

| Thiết Bị | Nguồn Cần | Ghi Chú |
|----------|-----------|---------|
| **Raspberry Pi 4** | 5V 3A USB-C | Adapter chính hãng |
| **Servo + Motor** | 12V 5A Adapter | Nguồn chính cho hệ thống |

**Total: CHỈ CẦN 2 NGUỒN ĐIỆN!**

---

## 📦 Linh Kiện Cần Mua Thêm

| Linh Kiện | Thông Số | Số Lượng | Link Tham Khảo |
|-----------|----------|----------|----------------|
| **Adapter 12V DC** | 12V 5A (60W) | 1 | Shopee/Lazada |
| **Buck Converter** | LM2596  | 1 | Module hạ áp 12V→6V |
| **Terminal Block** | 2-3 cổng xoắn vít | 2-3 | Nối dây GND chung |
| **Dây Nguồn** | 18-20 AWG | 2-3m | Đỏ (+) và Đen (GND) |

**Chi phí ước tính: ~200,000 - 300,000 VNĐ**

---

## 🔧 Kết Nối Đơn Giản 4 Bước

### Bước 1: Cấp Nguồn 12V
```
Adapter 12V ─┬─→ Buck Converter IN+
             └─→ L298N (+12V)
```

### Bước 2: Hạ Áp Xuống 6V
```
Buck Converter:
  OUT+ ─→ Servo VCC (dây đỏ)
  OUT- ─→ Common GND
```
⚙️ **Điều chỉnh Buck về 6.0V trước khi nối servo!**

### Bước 3: Nối Common GND
```
Pi GND ──┬── Servo GND ──┬── L298N GND ──┬── 12V Adapter GND
         └───────────────┴────────────────┘
                  TẤT CẢ NỐI CHUNG!
```

### Bước 4: Kết Nối GPIO
```
Pi GPIO 18 → Servo Signal (dây cam)
Pi GPIO 22 → L298N ENA
Pi GPIO 23 → L298N IN1
Pi GPIO 24 → L298N IN2
```

---

## ⚡ Sơ Đồ Tóm Tắt

```
┌──────────────┐
│ Adapter 12V  │
│    (5A)      │
└──┬───────┬───┘
   │       │
  +12V    GND
   │       │
   ├───────┼─────────────┐
   │       │             │
   ▼       ▼             ▼
┌─────────────┐    ┌──────────┐
│Buck Convert │    │  L298N   │
│ 12V → 6V    │    │ (Motor)  │
└──┬──────────┘    └──────────┘
   │ 6V
   ▼
┌──────────┐
│  Servo   │
│ MG996R   │
└──────────┘
     │
   GPIO 18
     │
┌────▼───────┐
│ Rasp Pi 4  │
│ (5V USB-C) │
└────────────┘
```

---

## 🛡️ An Toàn Điện - 3 Điều QUAN TRỌNG

### ⚠️ 1. TUYỆT ĐỐI KHÔNG lấy nguồn servo từ Pi!
```
❌ SAI: Pi 5V → Servo (sẽ HỎng Pi!)
✅ ĐÚNG: Buck 6V → Servo
```

### ⚡ 2. BẮT BUỘC nối Common GND!
```
Nếu không nối GND chung:
→ Tín hiệu GPIO không hoạt động
→ Servo/Motor không nhận lệnh
```

### 🔍 3. Điều chỉnh Buck Converter TRƯỚC!
```
Bước 1: Kết nối 12V vào Buck
Bước 2: Đo điện áp OUTPUT
Bước 3: Vặn vít đến 6.0V
Bước 4: MỚI nối servo
```

---

## 🧪 Test Nhanh

### Test 1: Kiểm tra điện áp
```bash
# Dùng đồng hồ vạn năng:
Buck Output: 6.0V ± 0.1V ✓
L298N VIN: 12V ✓
Common GND: 0V giữa tất cả các điểm ✓
```

### Test 2: Test Servo
```bash
cd ~/System_Conveyor
python hardware/servo_control.py
# Phải thấy servo quay Left → Center → Right
```

### Test 3: Test Motor
```bash
python hardware/motor_control.py
# Motor phải quay 2 chiều
```

---

## 📊 Tính Toán Công Suất

| Thiết Bị | Điện Áp | Dòng Điện | Công Suất |
|----------|---------|-----------|-----------|
| Servo MG996R | 6V | 1A | 6W |
| Motor JGB37-545 | 12V | 2A | 24W |
| L298N Logic | 12V | 0.1A | 1.2W |
| **TỔNG** | - | - | **~31W** |

**Nguồn 12V 5A (60W) → Dư 50% → An toàn! ✅**

---

## 🎯 Checklist Cuối Cùng

Trước khi bật nguồn, kiểm tra:

- [ ] ✅ Buck converter đã điều chỉnh về 6.0V
- [ ] ✅ Tất cả GND đã nối chung (Pi + Servo + L298N + 12V)
- [ ] ✅ Servo có nguồn 6V RIÊNG (không từ Pi)
- [ ] ✅ L298N có nguồn 12V
- [ ] ✅ Raspberry Pi có nguồn USB-C 5V riêng
- [ ] ✅ Không có dây chạm ngắn mạch
- [ ] ✅ Đã kiểm tra cực tính (+/- đúng)

---

## 📞 Troubleshooting Nhanh

### Vấn đề: Servo không chạy
```
✓ Kiểm tra nguồn 6V có đến servo không
✓ Kiểm tra Common GND
✓ Test GPIO 18 bằng LED
```

### Vấn đề: Motor không quay
```
✓ Kiểm tra 12V vào L298N
✓ Thử đảo dây motor
✓ Test GPIO 22-24
```

### Vấn đề: Hệ thống không hoạt động
```
✓ Kiểm tra COMMON GND trước tiên!
✓ Đo điện áp từng điểm
✓ Test từng module riêng
```

---

## 📚 Tài Liệu Chi Tiết

Xem thêm:
- **[Detailed Wiring Diagram](detailed_wiring_diagram.md)** - Sơ đồ đầy đủ với ASCII art
- **[Hardware Setup](hardware_setup.md)** - Hướng dẫn lắp ráp chi tiết
- **[User Manual](user_manual.md)** - Hướng dẫn vận hành

---

## 💡 Tips Hữu Ích

1. **Dùng màu dây chuẩn:**
   - ĐỎ = +12V, +6V (Positive)
   - ĐEN = GND (Ground)
   - VÀNG/CAM = Signal
   
2. **Ghi nhãn:**
   - Dán nhãn voltage trên mỗi điểm
   - Đánh số thứ tự kết nối
   
3. **An toàn:**
   - Luôn TẮT nguồn khi thay đổi kết nối
   - Kiểm tra KỸ trước bật nguồn
   - Có chuẩn bị jumper/dây dự phòng

---

**Chúc bạn lắp đặt thành công! 🎉**

*Nếu có thắc mắc, tham khảo [detailed_wiring_diagram.md](detailed_wiring_diagram.md) hoặc [hardware_setup.md](hardware_setup.md)*
