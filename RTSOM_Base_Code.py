import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Calibration value: adjust based on actual object
pixels_per_cm = 59.999999999

# Create main window
window = tk.Tk()
window.title("Real-Time Object Measurement") 
window.geometry("960x700")

# Display label
label = ttk.Label(window)
label.pack(padx=10, pady=10)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def get_contours(img, img_contour, ppcm):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            x, y, w, h = cv2.boundingRect(cnt)
            width_cm = round(w / ppcm, 1)
            height_cm = round(h / ppcm, 1)
            cv2.rectangle(img_contour, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.putText(img_contour, f"{width_cm} cm x {height_cm} cm", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            break  # only measure one object

def process_frame():
    ret, frame = cap.read()
    if not ret:
        print("Webcam not detected")
        return

    frame = cv2.resize(frame, (960, 640))
    contour_img = frame.copy()

    blur = cv2.GaussianBlur(frame, (7, 7), 1)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dilated = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

    get_contours(dilated, contour_img, pixels_per_cm)

    img_rgb = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
    label.img_tk = img_tk
    label.config(image=img_tk)

    window.after(30, process_frame)

# Quit function for key/button
def exit_program(event=None):
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

# Bind 'q' key and add a Quit button
window.bind('<q>', exit_program)
ttk.Button(window, text="Quit", command=exit_program).pack(pady=10)

# Start the loop
window.after(0, process_frame)
window.mainloop()
