# Bowling Project

This project aims to automate the aiming and firing system of a mini bowling alley.

---

## Idea and Development

This project is primarily developed in Python and C++ for Arduino.

The Python script is responsible for capturing an image using a webcam pointed at the bowling pins, identifying their positions, calculating the center of the pins, and determining the angle at which the bowling ball needs to be fired.

C++ is used to control the movement of the bowling ball pipe and activate the solenoid to launch the ball for a strike.

---

## Approach

I initially painted a green dot on the two bottom vertices of the bowling pin triangle to align the image every time I set up the webcam. I measured the board to find the triangle with vertices at the two green points and the bowling ball. This is essential to correctly compute the angle, even if the camera moves slightly.
The computer then maps each pixel of the bottom side of the triangle to the angle at which the bowling ball needs to be fired. Later, I painted each bowling pin with a red dot to easily recognize them.
For every shot, the script computes the angle by projecting the center point onto the bottom line of the triangle, then calculates the angle at which the ball needs to be fired based on the initial values.
