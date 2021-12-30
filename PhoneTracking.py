import cv2
from random import randrange

trained_phone_data = cv2.CascadeClassifier("haarcascadeandphonev1.xml")

# To capture image from webcam
webcam = cv2.VideoCapture(0)

# Iteratre forever frames
while True:

    successful_frame_read, frame = webcam.read()

    # Change picture to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    phone_coordinates = trained_phone_data.detectMultiScale(grayscaled_img)

    # Draw a rectangle around the face
    for (x, y, w, h) in phone_coordinates:
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (randrange(256), randrange(256), randrange(256)),
            7,
        )

    cv2.imshow("Phone use detector", frame)
    key = cv2.waitKey(1)
    # the '1' is the auto wait before the program hits a key so the webcame image can move to the next frame-otherwise it gets stuck until we manually press a key

    if key == 81 or key == 113:
        break

# Release the Videocapture Object
webcam.release()


print("Code completed!")
