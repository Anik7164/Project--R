import cv2
from ultralytics import YOLO

# Load the lightweight YOLO model for edge inference (downloads automatically)
model = YOLO('yolo11n.pt') 

# Open the video feed (replace with 0 to use your laptop webcam directly)
video_path = "sample_match.mp4"
cap = cv2.VideoCapture(video_path)

print("Starting real-time tracking. Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        # Run YOLO tracking on the current frame
        # classes=[0, 32] specifically filters for 'person' (players) and 'sports ball'
        results = model.track(frame, persist=True, classes=[0, 32])
        
        # Visualize the bounding boxes and tracking IDs on the frame
        annotated_frame = results[0].plot()
        
        # Display the live annotated frame
        cv2.imshow("KothaCast - Live Vision Tracking", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # End of video stream
        break

# Clean up memory
cap.release()
cv2.destroyAllWindows()