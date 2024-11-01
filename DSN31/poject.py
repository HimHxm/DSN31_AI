#หัวข้อมูล
    #ไหล่ อก เอว น้ำหนัก ส่วนสูง 
    # ** โทนสีผิว สถานที่(ถ้ามี) ** 
    #รูปที่ให้เลือกหุ่น  ต้องไปบวกลบกัน
    #เพศ
from ultralytics import YOLO
import cv2
camera = cv2.VideoCapture(0)


model = YOLO("ShirtColor.pt")

 
 # ที่กรอกข้อมูล
#Chest = float(input("กรอกรอบอกของท่าน(นิ้ว) : "))
#height = float(input("กรอกส่วนสูงของท่าน(cm) : "))
#weight = float(input("กรอกน้ำหนักของท่าน :"))


while True:
    check , frame = camera.read()   
    #ส่วนข้อมูล 
    
    #ปรับเป็นกล้องมือถือ
        #framerot = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    
    #ตั้งค่าการตรวจจับ 
        
    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    Blur = cv2.GaussianBlur(frame_gray,(5,5),0)
    thesh , frame_binary = cv2.threshold(Blur,150,255,cv2.THRESH_BINARY)
    dilate=cv2.dilate(frame_binary,None,iterations=3)
    contours,hierarchy=cv2.findContours(dilate,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    contours = model(frame)




    annotated_frame = contours[0].plot()
    
    #วาดสี่เหลี่ยม
        #for contour in contours:
            #(x,y,w,h) = cv2.boundingRect(contour)
            
            #สำคัญ อาจใช้กับ size ได้
            #if Chest <=38:
                #if cv2.contourArea(contour)<7000:
                    #continue
            
                #cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
            #else :
                #continue


#ส่วนแสดงผล
    cv2.imshow("Output",annotated_frame)
    if cv2.waitKey(1) & 0xff == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()











    





























