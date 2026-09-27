import cv2
import math
from ultralytics import YOLO

# Load the model
model = YOLO('yolo11n.pt') 
cap = cv2.VideoCapture("sample_match.mp4")

prev_ball_pos = None
event_cooldown = 0

print("Tracking active. Looking for rapid ball movement to trigger commentary...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Run tracking (verbose=False keeps the terminal clean)
    results = model.track(frame, persist=True, classes=[0, 32], verbose=False)
    annotated_frame = results[0].plot()
    
    # Extract bounding box data
    boxes = results[0].boxes
    for box in boxes:
        # Class 32 is 'sports ball' in the COCO dataset
        if int(box.cls[0]) == 32: 
            x, y, w, h = box.xywh[0].tolist()
            current_ball_pos = (x, y)
            
            # If we have a previous position, calculate the speed (pixel distance)
            if prev_ball_pos and event_cooldown == 0:
                dist = math.sqrt((x - prev_ball_pos[0])**2 + (y - prev_ball_pos[1])**2)
                
                # If the ball moves more than 50 pixels between frames, trigger an event
                if dist > 50: 
                    print(f"\n⚽ EVENT TRIGGERED: Fast ball movement! (Velocity: {int(dist)} pixels/frame)")
                    print("--> Sending to LLM for Commentary Generation...")
                    
                    # Cooldown for 30 frames (~1 second) so we don't spam the LLM
                    event_cooldown = 30 
            
            prev_ball_pos = current_ball_pos
            
    if event_cooldown > 0:
        event_cooldown -= 1
        
    cv2.imshow("KothaCast - Live Action Heuristics", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()