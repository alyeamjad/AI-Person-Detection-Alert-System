import cv2
import threading
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import datetime
import time
from playsound import playsound
import os
import csv
import smtplib
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
import io

class YoloApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv11n Person Detection in ROI")
        self.cap = None
        self.frame = None
        self.rois = []  # Changed to support multiple ROIs
        self.running = False
        self.detection_enabled = False
        self.model = YOLO("yolo11n.pt")
        self.class_names = self.model.names
        self.conf_threshold = 0.5
        self.save_video = True
        self.video_writer = None

        # Email Configuration
        self.email_config = {
            'smtp_server': 'smtp.gmail.com',
            'smtp_port': 587,
            'sender_email': '',
            'sender_password': '',
            'recipient_email': ''
        }

        # Alert Frame Capture
        self.alert_frames = {}  # Dictionary to store frames for each ROI
        self.frame_count = 4    # Number of frames to capture per alert

        # Person Stay Detection
        self.person_start_times = {}  # Dictionary to track multiple ROIs
        self.alert_played = {}
        self.audio_path = os.path.join(os.getcwd(), "voice.mp3")
        self.wait_time = 10  # Default wait time in seconds

        # Detection Logging
        self.log_file = "detection_log.csv"
        self.setup_logging()

        # GUI Elements
        self.canvas = Label(self.root)
        self.canvas.pack()

        self.btn_frame = Frame(self.root)
        self.btn_frame.pack(pady=10)

        Button(self.btn_frame, text="Start Camera", command=self.start_camera, width=15).grid(row=0, column=0, padx=5)
        Button(self.btn_frame, text="Add ROI", command=self.add_roi, width=15).grid(row=0, column=1, padx=5)
        Button(self.btn_frame, text="Clear ROIs", command=self.clear_rois, width=15).grid(row=0, column=2, padx=5)
        Button(self.btn_frame, text="Start Detection", command=self.start_detection, width=15).grid(row=0, column=3, padx=5)
        Button(self.btn_frame, text="Stop", command=self.stop_camera, width=15).grid(row=0, column=4, padx=5)

        # Email Configuration Frame
        self.email_frame = Frame(self.root)
        self.email_frame.pack(pady=5)
        
        Label(self.email_frame, text="Email Configuration:").grid(row=0, column=0, columnspan=2, pady=5)
        
        Label(self.email_frame, text="Sender Email:").grid(row=1, column=0, padx=5)
        self.sender_email_entry = Entry(self.email_frame, width=30)
        self.sender_email_entry.grid(row=1, column=1, padx=5)
        
        Label(self.email_frame, text="App Password:").grid(row=2, column=0, padx=5)
        self.sender_password_entry = Entry(self.email_frame, width=30, show="*")
        self.sender_password_entry.grid(row=2, column=1, padx=5)
        
        Label(self.email_frame, text="Recipient Email:").grid(row=3, column=0, padx=5)
        self.recipient_email_entry = Entry(self.email_frame, width=30)
        self.recipient_email_entry.grid(row=3, column=1, padx=5)
        
        Button(self.email_frame, text="Save Email Config", command=self.save_email_config, width=20).grid(row=4, column=0, columnspan=2, pady=5)

        Label(self.btn_frame, text="Wait Time (s):").grid(row=1, column=0, padx=5, pady=5)
        self.wait_entry = Entry(self.btn_frame, width=5)
        self.wait_entry.insert(0, str(self.wait_time))
        self.wait_entry.grid(row=1, column=1)

        self.status_label = Label(self.root, text="Status: Not started", font=("Arial", 12), fg="blue")
        self.status_label.pack(pady=5)

        # ROI Listbox
        self.roi_frame = Frame(self.root)
        self.roi_frame.pack(pady=5)
        Label(self.roi_frame, text="Active ROIs:").pack()
        self.roi_listbox = Listbox(self.roi_frame, width=40, height=5)
        self.roi_listbox.pack()

    def setup_logging(self):
        if not os.path.exists(self.log_file):
            with open(self.log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'ROI', 'Event', 'Confidence'])

    def log_detection(self, roi_index, event, confidence=None):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, f"ROI {roi_index + 1}", event, confidence])

    def add_roi(self):
        if self.frame is None:
            messagebox.showwarning("Warning", "Camera not started!")
            return

        self.running = False
        self.cap.release()

        roi = cv2.selectROI("Draw ROI", self.frame, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow("Draw ROI")

        if roi[2] > 0 and roi[3] > 0:
            self.rois.append(roi)
            self.roi_listbox.insert(END, f"ROI {len(self.rois)}: {roi}")
            messagebox.showinfo("ROI Added", f"Region {len(self.rois)}: {roi}")
        else:
            messagebox.showerror("Error", "Invalid ROI selected")

        self.start_camera()

    def clear_rois(self):
        self.rois = []
        self.roi_listbox.delete(0, END)
        self.person_start_times = {}
        self.alert_played = {}
        self.alert_frames = {}
        messagebox.showinfo("ROIs Cleared", "All ROIs have been cleared")

    def save_email_config(self):
        self.email_config['sender_email'] = self.sender_email_entry.get()
        self.email_config['sender_password'] = self.sender_password_entry.get()
        self.email_config['recipient_email'] = self.recipient_email_entry.get()
        messagebox.showinfo("Success", "Email configuration saved!")

    def send_alert_email(self, roi_index, frames):
        if not all(self.email_config.values()):
            messagebox.showwarning("Warning", "Please configure email settings first!")
            return

        try:
            # Create message
            msg = MIMEMultipart()
            msg['Subject'] = f'Alert: Person Detected in ROI {roi_index + 1}'
            msg['From'] = self.email_config['sender_email']
            msg['To'] = self.email_config['recipient_email']

            # Add text
            text = f"Person detected in ROI {roi_index + 1} at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            text += f"Attached are {len(frames)} frames captured during the alert."
            msg.attach(MIMEText(text))

            # Add images
            for i, frame in enumerate(frames):
                _, img_encoded = cv2.imencode('.jpg', frame)
                img_bytes = img_encoded.tobytes()
                img = MIMEImage(img_bytes)
                img.add_header('Content-Disposition', 'attachment', filename=f'alert_frame_{i+1}.jpg')
                msg.attach(img)

            # Send email
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.send_message(msg)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to send email: {str(e)}")

    def update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            display_frame = frame.copy()
            self.frame = frame.copy()

            any_person_detected = False
            status_messages = []

            for i, roi in enumerate(self.rois):
                x, y, w, h = roi
                roi_frame = frame[y:y+h, x:x+w]
                results = self.model(roi_frame, verbose=False)[0]
                annotated = roi_frame.copy()

                person_detected = False
                if results.boxes is not None:
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])

                        if cls_id == 0 and conf >= self.conf_threshold:
                            person_detected = True
                            any_person_detected = True
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            label = f"{self.class_names[cls_id]} {conf:.2f}"
                            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6, (0, 255, 0), 2)

                display_frame[y:y+h, x:x+w] = annotated

                if person_detected:
                    if i not in self.person_start_times:
                        self.person_start_times[i] = time.time()
                        self.log_detection(i, "Person Entered", conf)
                        # Initialize frame capture for this ROI
                        self.alert_frames[i] = []
                    elif len(self.alert_frames[i]) < self.frame_count:
                        # Continue capturing frames
                        self.alert_frames[i].append(display_frame.copy())

                    elapsed = time.time() - self.person_start_times[i]
                    remaining = self.wait_time - elapsed

                    if remaining > 0:
                        status_messages.append(f"ROI {i+1}: Alert in {int(remaining)}s")
                    else:
                        status_messages.append(f"ROI {i+1}: Alert!")
                        if i not in self.alert_played or not self.alert_played[i]:
                            threading.Thread(target=self.play_audio, daemon=True).start()
                            self.alert_played[i] = True
                            self.log_detection(i, "Alert Triggered", conf)
                            # Send email with captured frames
                            if len(self.alert_frames[i]) > 0:
                                threading.Thread(target=self.send_alert_email, 
                                              args=(i, self.alert_frames[i]), 
                                              daemon=True).start()
                else:
                    if i in self.person_start_times:
                        self.log_detection(i, "Person Left")
                    self.person_start_times.pop(i, None)
                    self.alert_played.pop(i, None)
                    self.alert_frames.pop(i, None)

            if not any_person_detected:
                self.status_label.config(text="Waiting for person...", fg="blue")
            else:
                self.status_label.config(text=" | ".join(status_messages), fg="orange")

            # Convert to Tk format
            rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            self.canvas.imgtk = imgtk
            self.canvas.configure(image=imgtk)
            self.root.update_idletasks()

            if self.save_video and self.video_writer:
                self.video_writer.write(display_frame)

            time.sleep(0.03)

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.running = True
        self.status_label.config(text="Status: Camera started", fg="green")
        self.start_detection()

    def stop_camera(self):
        self.running = False
        self.cap.release()
        self.status_label.config(text="Status: Camera stopped", fg="red")

    def start_detection(self):
        self.detection_enabled = True
        self.status_label.config(text="Status: Detection started", fg="green")
        threading.Thread(target=self.update_frame).start()

    def play_audio(self):
        playsound(self.audio_path)

if __name__ == "__main__":
    root = Tk()
    app = YoloApp(root)
    root.mainloop()