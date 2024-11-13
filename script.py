import cv2
import numpy as np
import math

def calibrate():
    image = cv2.imread("calibration.jpg")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([30, 50, 50])
    upper_green = np.array([90, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    points = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            points.append((cX, cY))
    if len(points) != 2:
        print("Error: Could not detect the calibration points.")
        exit()
    
    return points

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
    return [center_x, center_y]
    

def get_image(name):
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)
    
    cv2.imwrite(name, frame)
    print(f"Picture saved as {name}")
    cv2.destroyAllWindows()

def main():
    #get_image("calibration.jpg")
    calibration_points =calibrate()
    print("calibration is done, place pins\n")
    cv2.waitKey(0)
    #get_image("bowling_pins.jpg")
    distance_factor = 24/5 # mesured on the board
    bowling_pins_center=compute_center()
    calibration_distance = ((calibration_points[0][0] - calibration_points[1][0])**2 + (calibration_points[0][1] - calibration_points[1][1])**2)**0.5
    image = cv2.imread("bowling_pins.jpg")
    
    #calculate angle of the bottom line
    delta_x = calibration_points[1][0] - calibration_points[0][0]
    delta_y = calibration_points[1][1] - calibration_points[0][1]
    bottom_line_angle = np.arctan2(delta_y, delta_x)

    # Calculate the middle point of the bottom line
    bottom_line_center_point = ((calibration_points[0][0] + calibration_points[1][0]) // 2, (calibration_points[0][1] + calibration_points[1][1]) // 2)

    # Calculate the ball point using the angle knowing the distance from the center of the bottom line
    length = calibration_distance*distance_factor  # Length of the line
    ball_point = (int(bottom_line_center_point[0] + length * np.sin(-bottom_line_angle)), int(bottom_line_center_point[1] + length * np.cos(-bottom_line_angle)))
    # Calculate the vector from calibration_point[0] to calibration_point[1]
    line_vector = np.array([calibration_points[1][0] - calibration_points[0][0], calibration_points[1][1] - calibration_points[0][1]])
    center_vector = np.array([bowling_pins_center[0] - calibration_points[0][0], bowling_pins_center[1] - calibration_points[0][1]])
    # Project the center_vector onto the line_vector
    line_length_squared = line_vector.dot(line_vector)
    projection_length = center_vector.dot(line_vector) / line_length_squared
    projection_vector = projection_length * line_vector
    # Calculate the coordinates of the projected point
    projected_point = (calibration_points[0][0] + projection_vector[0], calibration_points[0][1] + projection_vector[1])
    projected_point = (int(projected_point[0]), int(projected_point[1]))

    bottom_line_center_point=((calibration_points[0][0]+calibration_points[1][0])//2, (calibration_points[0][1]+calibration_points[1][1])//2)

    cv2.circle(image, bowling_pins_center, 5, (0, 255, 0), -1)  # Green dots on center of the pins
    cv2.line(image, bottom_line_center_point, ball_point, (0, 0, 255), 2) # Red line from the center of the bottom line to the ball
    cv2.line(image, calibration_points[0], calibration_points[1], (0, 0, 255), 2) # Red line between the calibration points
    cv2.line(image,calibration_points[0],ball_point,(0,0,255),2) # Red line from the first calibration point to the ball
    cv2.line(image,calibration_points[1],ball_point,(0,0,255),2)# Red line from the second calibration point to the ball
    cv2.line(image,bowling_pins_center,ball_point,(0,255,0),3)# Green line on the trajectory
    cv2.circle(image, projected_point, 5, (255, 255, 0), -1)  # Red dot at the projected point
    cv2.line(image, ball_point, projected_point, (255, 0, 0), 1)#blue line for the trajectory till the projected point

    #calculate the angle of fireng 
    fireng_angle =np.degrees(np.arctan(math.dist(projected_point,bottom_line_center_point)/math.dist(bottom_line_center_point,ball_point)))
    cv2.putText(image, f"Fireng angle: {fireng_angle}", (7, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

    cv2.imshow("Detected Points and Center", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    

if __name__ =="__main__":
    main()


