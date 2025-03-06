import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import voiletion_detection
import manage_databsase
import notification
def upload_video():
    global video_path
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi;*.mov")])
    if video_path:
        status_label.config(text=f"Video uploaded: {video_path.split('/')[-1]}", fg="green")

def process_video():
    if video_path:
        status_label.config(text=f"Processing video...", fg="blue")
        final_result = voiletion_detection.process_video(video_path)
        print("Final Result = ",final_result)
        numbers_list = list(final_result.keys())
        voilation_list = final_result.get(numbers_list[0])
        voilated_rules = " ".join(voilation_list)
        # number_plate = numbers_list.replace(" ","")
        manage_databsase.insert_voilation(numbers_list[0],voilated_rules)
        full_name,email = manage_databsase.get_user_by_vehicle(numbers_list[0])
        notification.send_violation_email(email, full_name, numbers_list, voilated_rules)
        status_label.config(text=f"Rule violated by vehicle no: {numbers_list[0]}", fg="red")
    else:
        status_label.config(text="Please upload a video first!", fg="red")


root = tk.Tk()
root.title("Traffic Rule Violation System")
root.geometry("800x600")
root.configure(bg="#f0f0f0")

title_label = Label(root, text="HELMET, SPEED, LANE AND NUMBER PLATE DETECTION WITH NOTIFICATION SYSTEM USING YOLO ALGORITHM",
                    font=("Arial", 14, "bold"), wraplength=780, justify="center", bg="#f0f0f0", fg="#333")
title_label.pack(pady=10)


try:
    img = Image.open("banner.png")  
    img = img.resize((800,200))
    img = ImageTk.PhotoImage(img)
    img_label = Label(root, image=img, bg="#f0f0f0")
    img_label.pack(pady=10)
except Exception as e:
    print("Error loading banner image:", e)


video_path = ""
upload_btn = Button(root, text="Upload Video", command=upload_video, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white", padx=20, pady=10)
upload_btn.pack(pady=10)

process_btn = Button(root, text="Process Video", command=process_video, font=("Arial", 12, "bold"), bg="#FF5722", fg="white", padx=20, pady=10)
process_btn.pack(pady=10)


status_label = Label(root, text="Status: Waiting for video upload", font=("Arial", 12), bg="#f0f0f0", fg="black")
status_label.pack(side="bottom", fill="x", pady=10)

root.mainloop()
