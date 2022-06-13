# Face Detection in OpenCV
# Import OpenCV and Keyboard (as K) Modules
import cv2
import keyboard as k

# Gets Webcam
webcam = cv2.VideoCapture(0)
# Gets Face Data from a Library
trainedFaceData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
while True:

    # Returns only as Tuple, so must give 2 values.
    successful_frame_read, frame = webcam.read()
    # Mirrors the Video
    frame = cv2.flip(frame, 1)

    # Converts the Image to Grayscale for Face Processing
    grayscale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Gets the Coordinates of the Face in the Image from the Face Data and Grayscale Image
    faceCoords = trainedFaceData.detectMultiScale(grayscale_img)

    # Uses FaceCoords to create a Rectangle around the Face
    for (x, y, w, h) in faceCoords:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0))

    # Shows the Webcam
    # param String is for "App" Name
    # param Frame the Frame to show
    cv2.imshow("Face Detection", frame)
    # Updates Frame every Millisecond
    cv2.waitKey(1)

    # Checks for Keypress of ("x") to Break For Loop, and End App Running.
    if k.is_pressed("x"):
        break


# If For Loop is Broken
webcam.release()
print("Done")
