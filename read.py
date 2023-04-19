import cv2 as cv
import numpy as np
import time

# Load the pre-trained deep learning face detection model
modelFile = "C:\\Users\Louis_HD\\Desktop\\dev\\test\\models\\res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "C:\\Users\\Louis_HD\\Desktop\\dev\\test\\models\\deploy.prototxt"
net = cv.dnn.readNetFromCaffe(configFile, modelFile)


net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


# Initialize the video capture object
#capture = cv.VideoCapture(0)

capture = cv.VideoCapture("C:\\Users\\Louis_HD\\Desktop\\dev\\test\\Photos\\clip2.mp4")


# Initialize the FPS counter
fps_start_time = time.time()
fps_counter = 0
fps = 0

while True:

    # Capture a frame from the video stream
    ret, frame = capture.read()
    
    # Perform face detection on the frame
    blob = cv.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()

    # Draw bounding boxes around the detected faces
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Calculate the FPS
    fps_counter += 1
    if (time.time() - fps_start_time) > 1:
        fps = fps_counter / (time.time() - fps_start_time)
        fps_counter = 0
        fps_start_time = time.time()

    # Display the FPS in the top left corner of the screen
    cv.putText(frame, f"FPS: {round(fps, 2)}", (10, 50), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the output frame
    cv.imshow('Detected Faces', frame)

    # Exit the loop if the 'd' key is pressed
    if cv.waitKey(1) & 0xFF == ord('d'):
        break

# Release the video capture object and close all windows
capture.release()
cv.destroyAllWindows()
