from flask import Flask, render_template, Response, request
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)

# โหลดโมเดล YOLO ที่เทรนมาแล้ว
model = YOLO("ShirtV11.pt")

# กำหนด tone_classid, activity_classid และ class_id_to_color
tone_classid = {
    "ทางการ": [8,1],
    "โดดเด่น": [5, 3, 9],
    "อ่อนโยน": [7, 8, 3],
    "สุขุม": [1, 2],
    "ขอเบๆ": [1, 2, 8],
}
activity_classid = {
    "ทางการ": [8,1],
    "ที่ท่องเที่ยวธรรมชาติ": [7, 8, 3, 2],
    "ที่ท่องเที่ยวในเมือง": [1, 2, 3, 5, 7, 8, 9],
    "ปาร์ตี้": [5, 3, 9],
}
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

# ฟังก์ชันหาค่าสีด้วย numpy ภายในกรอบที่ตรวจจับได้จากโมเดล YOLO
def filter_color(frame, selected_class_ids):
    # ตรวจจับวัตถุในเฟรมด้วยโมเดล YOLO
    results = model(frame, conf=0.5)
    
    for result in results[0].boxes:
        class_id = int(result.cls)
        bbox = result.xyxy[0].cpu().numpy()
        
        if class_id in selected_class_ids:
            x1, y1, x2, y2 = map(int, bbox)
            
            #ครอปพื้นที่ที่ตรวจจับได้
            roi = frame[y1:y2, x1:x2]
            
            #แปลง ROI เป็นโหมดสี HSV
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            #หาสีของวัตถุในกรอบ
            detected_color = detect_color_in_roi(hsv_roi)
            
            #จับเสื้อ
            if detected_color:

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{detected_color.capitalize()} Shirt", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return frame

#จับสี
def detect_color_in_roi(hsv_roi):
    color_bounds = {
        "Black": (np.array([0, 0, 0]), np.array([180, 255, 30])),
        "Blue": (np.array([90, 50, 50]), np.array([130, 255, 255])),
        "Green": (np.array([40, 50, 50]), np.array([80, 255, 255])),
        "Red": (np.array([0, 50, 50]), np.array([10, 255, 255])),
        "Sky": (np.array([90, 50, 50]), np.array([130, 255, 255])),
        "White": (np.array([0, 0, 200]), np.array([180, 20, 255])),
        "Yellow": (np.array([20, 50, 50]), np.array([30, 255, 255]))
    }

    for color_name, (lower_bound, upper_bound) in color_bounds.items():
        mask = cv2.inRange(hsv_roi, lower_bound, upper_bound)
        if cv2.countNonZero(mask) > 0:
            return color_name
    return None

#วิดีโอ
def generate_frames(selected_class_ids):
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame = filter_color(frame, selected_class_ids)
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        #ส่งเฟรมไปยังเบราว์เซอร์
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('Main.html')

@app.route('/setting')
def setting():
    return render_template('Set.html')

@app.route('/video_feed', methods=['POST'])
def video_feed():
    
    #รับค่าจากฟอร์ม
    activity = request.form.get("activity")
    tone = request.form.get("tone")

    #กำหนด selected_class_ids ตามข้อมูลที่ได้รับ
    selected_class_ids = []
    if activity in activity_classid:
        selected_class_ids.extend(activity_classid[activity])
    if tone in tone_classid:
        selected_class_ids.extend(tone_classid[tone])

    #ส่งเฟรมไปยังฟังก์ชัน generate_frames
    return Response(generate_frames(selected_class_ids),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
