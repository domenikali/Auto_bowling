import cv2
import numpy as np

def compute_center():
    image = cv2.imread("bowling_pins.jpg")

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the red color range in HSV
    # You may need to adjust these values depending on the shade of red in your image
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 | mask2

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append((cX, cY))

    center_x = 0
    center_y = 0

    for point in points:
        center_x += point[0]
        center_y += point[1]

    center_x = center_x // len(points)
    center_y = center_y // len(points)

    for point in points:
        cv2.circle(image, point, 5, (0, 255, 0), -1)  # Green dots on detected points
    if len(points) >= 2:
        cv2.circle(image, (center_x, center_y), 5, (255, 0, 0), -1)  # Blue dot at the center

    cv2.imshow("Detected Points and Center", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

def get_image():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)
    filename = 'bowling_pins.jpg'
    cv2.imwrite(filename, frame)
    print(f"Picture saved as {filename}")
    cv2.destroyAllWindows()

def main():
    get_image()
    compute_center()

if __name__ =="__main__":
    main()


