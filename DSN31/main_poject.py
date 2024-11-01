import cv2
from ultralytics import YOLO

# โหลดโมเดล YOLO ที่เทรนมาแล้ว
model = YOLO('Train/ShirtV11.pt')

# โทนสีที่เกี่ยวข้องกับกิจกรรม
tone_classid = {
    "ทางการ": [8, 1],
    "โดดเด่น": [5, 3, 9],
    "อ่อนโยน": [7, 8, 3],
    "สุขุม": [1, 2],
    "ขอเบๆ": [1, 2, 8],
}

# กิจกรรมที่เกี่ยวข้องกับโทนสี
activity_classid = {
    "ทางการ": [8, 1],
    "ที่ท่องเที่ยวธรรมชาติ": [7, 8, 3, 2],
    "ที่ท่องเที่ยวในเมือง": [1,2,3,5,7,8,9],  # เปลี่ยนจาก tone_classid เป็นรายการ class_id
    "ปาร์ตี้": [5, 3, 9],
}

# Mapping ระหว่าง class_id กับสี
class_id_to_color = {
    1: "Black",    
    2: "Blue",  
    3: "Green",
    4: "Long",  
    5: "Red", 
    6: "Short", 
    7: "Sky", 
    8: "White", 
    9: "Yellow", 
}

# รับข้อมูลจากผู้ใช้
activity = input("ใส่สถานที่ ที่จะไป (ทางการ ที่ท่องเที่ยวธรรมชาติ ที่ท่องเที่ยวในเมือง ปาร์ตี้): ")
tone = input("ใส่โทนสีของคุณ (ทางการ โดดเด่น อ่อนโยน สุขุม ขอเบๆ): ")

# ค้นหา class_id ที่เกี่ยวข้องกับกิจกรรมและโทนสี
selected_class_ids = []

if activity in activity_classid:
    selected_class_ids.extend(activity_classid[activity])

if tone in tone_classid:
    selected_class_ids.extend(tone_classid[tone])

if not selected_class_ids:
    print("ไม่พบกิจกรรมหรือโทนสีที่ระบุ")
    exit()

# ฟังก์ชันสำหรับกรองเสื้อที่มีสีตามที่เลือก
def filter_color(frame, selected_class_ids):
    results = model(frame, conf=0.2)  # ใช้โมเดล YOLO ตรวจจับวัตถุในเฟรม

    # วนลูปผ่านวัตถุที่ตรวจจับได้
    for result in results[0].boxes:
        class_id = int(result.cls)  # หมายเลขคลาส
        bbox = result.xyxy[0].cpu().numpy()  # Bounding box

        # ตรวจสอบว่า class_id ที่ตรวจพบอยู่ใน selected_class_ids หรือไม่
        if class_id in selected_class_ids:
            detected_color = class_id_to_color.get(class_id, None)

            if detected_color:
                x1, y1, x2, y2 = map(int, bbox)

                # วาดกรอบรอบเสื้อที่ตรวจพบ
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detected_color.capitalize()} Shirt", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            print(f"ตรวจพบ class_id: {class_id}, แต่ไม่ตรงกับสีที่ต้องการ")

    return frame

# เปิดกล้อง
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ไม่สามารถเปิดกล้องได้")
    exit()

while True:
    check, frame = cap.read()
    if not check:
        print("ไม่สามารถอ่านภาพจากกล้องได้")
        break

    # ตรวจจับเสื้อและกรองเฉพาะสีที่ต้องการ
    frame = filter_color(frame, selected_class_ids)

    
    cv2.imshow("Color Filtered Shirt Detection", frame)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
