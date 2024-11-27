import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image, ImageTk
import pygame
import tkinter as tk
from tkinter import Label
import time
from scipy.spatial import distance as dist
import dlib
from plyer import notification
from plyer import vibrator

# Initialize Pygame mixer for alarm sound
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("C:/Users/raish/code/ddd_sys/beep2.mp3")
beep_sound = pygame.mixer.Sound("C:/Users/raish/code/ddd_sys/drowsy_beep.mp3")

# Load pre-trained model (for detecting open/closed eyes)
class EyeStateModel(nn.Module):
    def __init__(self):
        super(EyeStateModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)  
        self.fc2 = nn.Linear(128, 2) 
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 6 * 6)  
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = EyeStateModel()
model.load_state_dict(torch.load('C:/Users/raish/code/ddd_sys/test_train_detect.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# OpenCV Haar cascades for face and eye detection
face_cascade = cv2.CascadeClassifier('C:/Users/raish/code/ddd_sys/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:/Users/raish/code/ddd_sys/haarcascade_eye.xml')

# Initialize dlib for landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/Users/raish/code/ddd_sys/shape_predictor_68_face_landmarks.dat")

# Initialize variables for EAR detection
frame_counter = 0
sleep_counter = 0
drowsy_counter = 0
active_counter = 0
ear_threshold = 0.2  # Threshold for EAR (to consider eyes as closed)
no_eyes_counter = 0
status = ""
color = (0, 0, 0)

# Initialize Pygame for sound alarm
# Cool down period in seconds
COOL_DOWN_PERIOD = 5  # 5 seconds
last_beep_time = 0  # Stores the time of the last beep

def play_alarm():
    global last_beep_time
    current_time = time.time()    
    if current_time - last_beep_time >= COOL_DOWN_PERIOD:
        alarm_sound.play()  # Play the beep
        last_beep_time = current_time  # Update the last beep time

def play_beep():
    beep_sound.play()
def send_notification(title, message):
    notification.notify(
        title=title,
        message=message,
        app_name='Drowsiness Detection',
        timeout=5  # Notification stays for 5 seconds
    )
notification_sent = False
def send_notification(title, message):
    """Send notification only if it's not already sent."""
    global notification_sent
    if not notification_sent:
        notification.notify(
            title=title,
            message=message,
            app_name='Drowsiness Detection',
            timeout=5
        )
        notification_sent = True  # Mark notification as sent
def trigger_vibration(duration=2):
    try:
        vibrator.vibrate(time=duration)
    except NotImplementedError:
        print("Vibration is not supported on this platform.")
# Improved beverage recommendations based on the detected drowsiness level

def recommend_beverage(status):
    """Recommend beverages based on the drowsiness level."""
    current_time = time.localtime().tm_hour  # Get the current hour of the day

    if status == "Drowsy":
        if current_time < 12:  # Morning
            return "Moderate: Try a light coffee or green tea to refresh!"
        elif current_time < 18:  # Afternoon
            return "Moderate: How about some iced tea or a small coffee?"
        else:  # Evening
            return "Moderate: Avoid heavy caffeine! Maybe try herbal tea."
    
    elif status == "Sleeping":
        if current_time < 12:  # Morning
            return "Severe: Have an espresso shot or a strong black coffee!"
        elif current_time < 18:  # Afternoon
            return "Severe: You need a double shot of espresso or an energy drink!"
        else:  # Evening
            return "Severe: Drink some water or consider light stretching!"
    
    elif status == "Active :)":
        if current_time < 12:  # Morning
            return "Stay Active: Keep hydrated with some water or juice!"
        elif current_time < 18:  # Afternoon
            return "Stay Active: Maybe have some water or a refreshing smoothie!"
        else:  # Evening
            return "Stay Active: Stick with water to maintain hydration!"

    elif status == "No Eyes":
        return "No Eyes Detected: Please ensure your face is visible."
    
    return "Unknown state. Stay hydrated!"



# Function to calculate the Eye Aspect Ratio (EAR)
def calculate_ear(eye_points):
    A = dist.euclidean(eye_points[1], eye_points[5])  # Vertical distance between two eye landmarks
    B = dist.euclidean(eye_points[2], eye_points[4])  # Horizontal distance between two eye landmarks
    C = dist.euclidean(eye_points[0], eye_points[3])  # Horizontal distance between the outer eye landmarks
    ear = (A + B) / (2.0 * C)  # EAR formula
    return ear

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert image to grayscale
    transforms.Resize((48, 48)),  # Resize image to 48x48
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
])

# Function to update status and recommendation
def update_status():
    global sleep_counter, notification_sent, drowsy_counter, active_counter, no_eyes_counter, status, color, frame_counter

    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:  # No faces detected
        status = "No Eyes"
        color = (0, 0, 255)  # Red for no eyes
        recommendation_label.config(text="No Eyes Detected")
        no_eyes_counter += 1
    else:
        no_eyes_counter = 0  # Reset no eyes counter if eyes are detected
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

            # Detect facial landmarks using dlib
            face_rect = dlib.rectangle(x, y, x + w, y + h)
            landmarks = predictor(gray, face_rect)

            # Get eye landmarks for EAR calculation
            left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]  # Left eye landmarks
            right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]  # Right eye landmarks

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)

            # Debugging EAR values
            ear = (left_ear + right_ear) / 2
            print(f"Left EAR: {left_ear}, Right EAR: {right_ear}, Average EAR: {ear}")

            # Determine eye state based on EAR
            if ear < ear_threshold:  # If EAR is very low, eyes are closed
                sleep_counter += 1
                active_counter = 0
                if sleep_counter > 6 and sleep_counter < 24:  
                    status = "Drowsy"
                    color = (0, 0, 255)  #Blue for drowsy
                    play_beep() # play beep sound
                    send_notification("Drowsiness Alert", "Please stay alert. Eyes closed!")
                    beverage_recommendation = recommend_beverage(status)
                    recommendation_label.config(text=beverage_recommendation)
                    print("Drowsiness detected!")
                    trigger_vibration(duration=0.2)
                if sleep_counter > 24:  
                    status = "Sleeping"
                    color = (255, 0, 0)  # Red for sleeping
                    play_alarm()  # Play alarm sound
                    beverage_recommendation = recommend_beverage(status)
                    recommendation_label.config(text=beverage_recommendation)
                    print("Sleeping detected!")
            elif ear >= ear_threshold:  # If EAR is above threshold, eyes are open
                sleep_counter = 0
                active_counter += 1
                if active_counter > 11:
                    status = "Active :)"
                    color = (0, 255, 0)  # Green for active
                    beverage_recommendation = recommend_beverage(status)
                    recommendation_label.config(text=beverage_recommendation)
                    print("Active detected!")
                    if notification_sent:
                        notification_sent = False

    # Update status on GUI
    status_label.config(text=status, fg=f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')
    update_image(frame)

    # Set delay before checking again
    root.after(50, update_status)

# Function to update video frame
def update_image(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (800, 600))
    img = Image.fromarray(frame)
    img = ImageTk.PhotoImage(image=img)
    video_label.imgtk = img
    video_label.config(image=img)

# Main function
def main():
    global cap, status_label, recommendation_label, video_label, root
    root = tk.Tk()
    root.title("Drowsiness Detection App")
    cap = cv2.VideoCapture(0)
    status_label = tk.Label(root, text="", font=("Helvetica", 24))
    status_label.pack()
    recommendation_label = tk.Label(root, text="", font=("Helvetica", 16))
    recommendation_label.pack()
    video_label = tk.Label(root)
    video_label.pack()
    update_status()
    root.mainloop()

if __name__ == "__main__":
    main()

