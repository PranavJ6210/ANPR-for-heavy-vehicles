from ultralytics import YOLO
import cv2
import csv
import util
from sort.sort import *
from util import *
import easyocr
reader = easyocr.Reader(['en'])
from datetime import datetime

results = {}

mot_tracker = Sort()


coco_model = YOLO('truck.pt')
license_plate_detector = YOLO('numberplate.pt')


# cap = cv2.VideoCapture("test.mp4")
cap = cv2.VideoCapture("plate79.JPG")

vehicles = [0]
def writecsv(data_t):
    with open('D:/FinalMP1/test.csv', mode='a', newline='') as file:
        fieldnames = ['timestamp', 'license_plate_text']  
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        try:
            writer.writerow({'timestamp': data_t[0], 'license_plate_text': data_t[1]})
            print("done")
        except:
            pass
data = []
def read(img):

    detection = reader.readtext(img)
    for detect in detection:
        bbox, text, score = detect
        print(text)
        text = text.upper().replace(' ', '')
        dt=datetime.now()
        # data.append([dt, text])
        # write_csv(data)
        
        if license_complies_format(text):
            data.append([dt, format_license(text)])
            writecsv(data)
            return format_license(text), score
        
frame_nmr = -1
ret = True
# data = []
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        
        track_ids = mot_tracker.update(np.asarray(detections_))

        
        license_plates = license_plate_detector(detections)[0]
        lis = license_plates.plot()
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:

                
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)


                try:
                    license_plate_text, license_plate_text_score = read(license_plate_crop_thresh)

                except:
                    continue
        # write_csv(data)
        
        cv2.imshow("YOLOv8 Inference", lis)
        cv2.waitKey(0)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break



cap.release()
cv2.destroyAllWindows()