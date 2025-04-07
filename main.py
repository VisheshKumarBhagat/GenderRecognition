import cv2
from face_recognition import faceRecognitionPipeline

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()

    if ret == False: break
    # Write the frame to the output file
    # out.write(frame)

    pred_img, pred_dict = faceRecognitionPipeline(frame, path=False)

    # Display the captured frame
    cv2.imshow('Camera', pred_img)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
        break

# Release the capture and writer objects
cam.release()
# out.release()
cv2.destroyAllWindows()
