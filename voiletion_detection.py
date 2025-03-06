import csv
import cv2
import numpy as np
import tensorflow as tf
from collections import Counter
from sort.sort import Sort
from ultralytics import YOLO
import json
import pandas as pd
import time
import notification

helmet_model = YOLO("best.pt")  # Helmet detection model
vehicle_detector = YOLO('yolov8n.pt')  # Car/Bike detection
plate_detector = YOLO('license_plate_detector.pt')  # License plate detection
ocr_model = tf.keras.models.load_model('OCR_MODEL.keras')  # OCR model for plate reading
voilation_data = []

with open('label_dict.json', 'r') as f:
    label_dict = json.load(f)
label_dict = {int(k): v for k, v in label_dict.items()}

# Define Classes
helmet_labels = ['With Helmet', 'Without Helmet']
VEHICLE_CLASSES = {2: "Car", 3: "Motorbike"}  


CSV_FILE = "detections.csv"
df = pd.DataFrame(columns=["Frame", "Plate_Number", "Helmet_Status"])
df.to_csv(CSV_FILE, index=False)


check_speed = True
vehicle_data = {}  
frame_count = 0
store_limit = 10  
store_data = []  
tracker = Sort() 
vehicle_crossed = []
vehicle_positions = {} 
scale_factor = 0.0003
lane_status = "Not Violeted"
voilated_status = []
final_status = {}


def region_of_interest(image, vertices):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)
def find_lane_coordinates(image, lines):
    lane_points = []
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                bottom_left = (x1, y1) if y1 > y2 else (x2, y2)
                top_left = (x1, y1) if y1 <= y2 else (x2, y2)
                lane_points.append(bottom_left)
                lane_points.append(top_left)
    if lane_points:
        return min(lane_points, key=lambda p: p[1]), max(lane_points, key=lambda p: p[1])
    return None, None
def get_majority(data_list):
    if not data_list:
        return "Unknown"
    count = Counter(data_list)
    return count.most_common(1)[0][0]


def clean_binary_image(binary):
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    h, w = binary.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(binary, mask, (0, 0), 0)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            cv2.drawContours(binary, [cnt], 0, (0, 0, 0), -1)
    return binary

def segment_characters(number_plate_image):
    gray = cv2.cvtColor(number_plate_image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = clean_binary_image(thresh)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 10 < h < 100 and 5 < w < 80:
            bounding_boxes.append((x, y, w, h))
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])
    if len(bounding_boxes) < 2:
        return []
    mid_y = (bounding_boxes[0][1] + bounding_boxes[-1][1]) // 2
    line1, line2 = [], []
    for box in bounding_boxes:
        x, y, w, h = box
        if y < mid_y:
            line1.append(box)
        else:
            line2.append(box)
    line1 = sorted(line1, key=lambda b: b[0])
    line2 = sorted(line2, key=lambda b: b[0])
    all_lines = [line1, line2]
    char_images = []
    for line in all_lines:
        line_chars = []
        for (x, y, w, h) in line:
            char = binary[y:y+h, x:x+w]
            char = cv2.copyMakeBorder(char, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=0)
            char = cv2.resize(char, (32, 32))
            char = cv2.cvtColor(char, cv2.COLOR_GRAY2BGR)
            char = char.astype('float32') / 255.0
            line_chars.append(char)
        char_images.append(line_chars)
    return char_images


def recognize_characters(plate_image):
    char_images_per_line = segment_characters(plate_image)
    recognized_lines = []
    for line_chars in char_images_per_line:
        recognized_text = ""
        for char in line_chars:
            char = np.expand_dims(char, axis=0)
            prediction = ocr_model.predict(char)
            predicted_label = np.argmax(prediction)
            confidence = prediction[0][predicted_label]
            if confidence > 0.5:
                recognized_text += label_dict[predicted_label]
        recognized_lines.append(recognized_text)
    return " ".join(recognized_lines) if recognized_lines else "Unknown"


def process_video(video_path):
    global frame_count,check_speed

    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        detections = []  
        height, width = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (frame_width, frame_height))
        # Run Vehicle Detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1), 100, 250)
        roi_vertices = np.array([[(1200, height), (width//2+50, height//2.3), (width//2+150, height//2.3), (width-550, height)]], dtype=np.int32)
        cropped_edges = region_of_interest(edges, roi_vertices)
        lines = cv2.HoughLinesP(cropped_edges, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
        lane_start, lane_end = find_lane_coordinates(frame, lines)
        if lane_start and lane_end:
            cv2.line(frame, lane_start, lane_end, (255, 0, 0), 3)
            lane_x_range = (min(lane_start[0], lane_end[0]), max(lane_start[0], lane_end[0]))
        vehicle_results = vehicle_detector(frame)
        for result in vehicle_results:
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box)
                detections.append([x1, y1, x2, y2, 0.99])  

        tracked_objects = tracker.update(np.array(detections))

        for obj in tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            vehicle_class_id = int(obj[4]) 
            vehicle_id = f"V{track_id}"  
            
            if vehicle_class_id in [2, 3]:  
                if vehicle_id not in vehicle_data:
                    vehicle_data[vehicle_id] = {"Helmet": None, "Plate": None}

                if frame_count <= store_limit:
                    helmet_results = helmet_model(frame)
                    for result in helmet_results:
                        for box in result.boxes:
                            hx1, hy1, hx2, hy2 = map(int, box.xyxy[0])
                            cls = int(box.cls[0])
                            helmet_status = helmet_labels[cls]
                            vehicle_data[vehicle_id]["Helmet"] = helmet_status
                            cv2.putText(frame, helmet_status, (hx1, hy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, color=(255, 0, 0))

                    plate_results = plate_detector(frame)
                    for result in plate_results:
                        for box in result.boxes.xyxy.cpu().numpy():
                            px1, py1, px2, py2 = map(int, box)
                            plate_img = frame[py1:py2, px1:px2]
                            plate_number = recognize_characters(plate_img)
                            vehicle_data[vehicle_id]["Plate"] = plate_number
                            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                            cv2.putText(frame, plate_number, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # Store data for CSV
                    store_data.append([frame_count, vehicle_data[vehicle_id]["Plate"], vehicle_data[vehicle_id]["Helmet"]])
                else:
                    majority_helmet = get_majority([data[2] for data in store_data])
                    majority_plate = get_majority([data[1] for data in store_data])
                    cv2.putText(frame, f"Helmet: {majority_helmet}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Plate: {majority_plate}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                # Draw bounding box around vehicle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"ID: {vehicle_id}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                a1,a2 = x1,y1
                x1, y1, x2, y2 = map(int, box)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                if lane_x_range and lane_x_range[0] < center_x < lane_x_range[1] and (center_x, y1) not in vehicle_crossed:
                        vehicle_crossed.append((center_x, y1))
                        lane_status = "Lane Violated"
                        cv2.putText(frame, "Lane Violation", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  

                if (int(cls) in vehicle_positions) and check_speed == True:
                        prev_x, prev_y, prev_time = vehicle_positions[int(cls)]
                        pixel_distance = np.sqrt((center_x - prev_x) ** 2 + (center_y - prev_y) ** 2)
                        real_distance = pixel_distance * scale_factor  
                        time_diff = time.time() - prev_time
                        speed = real_distance / time_diff  
                        if frame_count <= store_limit:
                            speed_kmh = (speed * 3.6 ) 
                        else:
                            speed_kmh = (speed * 3.6 ) +15 
                        if speed_kmh > 50:
                            speed_kmh = 10
                        if speed_kmh > 40:
                            voilated_status.append("Overspeed")
                            check_speed = False
 
                else:
                        speed_kmh = 0  

                vehicle_positions[int(cls)] = (center_x, center_y, time.time())
                cv2.putText(frame, f"Speed: {speed_kmh:.2f} km/h", (a1+100, a2 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out.write(frame)
        resize = cv2.resize(frame, (1020, 600))
        cv2.imshow('Frame', resize)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if len(store_data) > 0:
        majority_helmet = get_majority([data[2] for data in store_data])
        majority_plate = get_majority([data[1] for data in store_data])
        no_plate = majority_plate.replace(" ","")
        # voilated_status[majority_plate].append(majority_plate)
        print(f"Majority Helmet Status: {majority_helmet}")
        print(f"Majority Plate Number: {majority_plate}")
        if lane_status == "Lane Violated":
            voilated_status.append(lane_status)

        if majority_helmet == "Without Helmet":
            voilated_status.append(majority_helmet)
        final_status[no_plate] = voilated_status

    with open(CSV_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Plate_Number", "Helmet_Status"])
        for data in store_data:
            writer.writerow(data)
    print(final_status)
    return final_status

video_path = 'demo.mp4'  
process_video(video_path)
