#  AI Person Detection & Alert System

##  Overview
The **AI Person Detection & Alert System** is a real-time security application built with **Python**, **OpenCV**, and **YOLOv11**.  
It detects people in **custom-defined Regions of Interest (ROIs)** from a live camera feed,  
triggers alerts if a person stays too long, plays an audio warning, and sends an **email notification** with captured frames.

---

##  Features
-  **Custom ROIs** – Select multiple detection zones.
-  **AI-Powered Detection** – Uses YOLOv11 for accurate real-time person detection.
-  **Stay Duration Alerts** – Triggers alert if a person remains too long in a zone.
-  **Audio Alerts** – Plays a warning sound.
-  **Email Notifications** – Sends detection images to a configured email.
-  **Logging** – Saves detection events to CSV.
-  **Video Saving** – Records detection footage

---

##  Clone the Repository

```
git clone https://github.com/alyeamjad/AI-Person-Detection-Alert-System.git
cd AI-Person-Detection-Alert-System
```
### Create a Conda Environment
```
conda create -n person_alert python=3.10 -y
conda activate person_alert
```
### Install Requirements 
```
pip install -r requirements.txt

```
### Run 
```
python main.py

```



