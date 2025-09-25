import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Tkinter window setup
window = tk.Tk()
window.title("Real-Time Object Measurement")
window.geometry("960x720")

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Calibration: Use a known object (e.g., credit card = 8.5 cm width)
pixels_per_cm = 38  # Adjust after observing the print output below
threshold1 = 50
threshold2 = 150
min_area = 5000

def getContours(img, imgContour, pixels_per_cm):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected = False

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > min_area:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)

            # Check aspect ratio to filter out noise
            aspect_ratio = w / float(h)
            if 0.5 < aspect_ratio < 2.5:
                width_cm = w / pixels_per_cm
                height_cm = h / pixels_per_cm

                # Calibration help — print width in pixels
                print(f"[Calibration] Width in pixels: {w} → pixels_per_cm = {w / 8.5:.2f}")

                # Only draw if object is roughly centered
                img_center_x = imgContour.shape[1] // 2
                obj_center_x = x + w // 2
                if abs(obj_center_x - img_center_x) < imgContour.shape[1] // 3:
                    cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(imgContour, f"{width_cm:.1f} cm x {height_cm:.1f} cm",
                                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    detected = True

    if not detected:
        cv2.putText(imgContour, "No valid object detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

def process_image():
    success, img = cap.read()
    if not success:
        print("Failed to capture frame")
        return

    img = cv2.resize(img, (900, 600))
    imgContour = img.copy()

    imgGray = cv2.cvtColor(cv2.GaussianBlur(img, (7, 7), 1), cv2.COLOR_BGR2GRAY)
    imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
    imgDil = cv2.dilate(imgCanny, np.ones((5, 5), np.uint8), iterations=1)

    getContours(imgDil, imgContour, pixels_per_cm)

    imgRGB = cv2.cvtColor(imgContour, cv2.COLOR_BGR2RGB)
    imgTk = ImageTk.PhotoImage(image=Image.fromarray(imgRGB))
    label.imgTk = imgTk
    label.config(image=imgTk)

    window.after(33, process_image)

# GUI label for video display
label = ttk.Label(window)
label.pack(padx=10, pady=10, fill="both", expand=True)

# Quit on 'q'
def exit_program(event):
    cap.release()
    cv2.destroyAllWindows()
    window.quit()

window.bind('<q>', exit_program)

# Start
window.after(0, process_image)
window.mainloop()

# Cleanup
cap.release()
cv2.destroyAllWindows()
