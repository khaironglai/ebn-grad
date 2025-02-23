import cv2
import torch

# Load the YOLOv10 model (assuming it's a PyTorch model)
# Replace 'yolov10_model_path' with the path to your YOLOv10 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/Users/tanjiunkoon/Downloads/Others/Finalized yolov10/app/current_best.pt')  # Update this for YOLOv10

# Set the model to evaluation mode
model.eval()

# Load the video file
video_path = "/Users/tanjiunkoon/Downloads/Others/Finalized yolov10/app/IMG_3785.mov"  # Replace with the path to your video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the output video
output_path = 'output_video.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Loop over video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Preprocess the frame for YOLO (if required by YOLOv10)
    # Convert the frame (BGR to RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert the frame to a torch tensor and normalize it
    img_tensor = torch.from_numpy(img_rgb).float()
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0  # Convert from HWC to CHW format and normalize to [0, 1]

    # Add a batch dimension
    img_tensor = img_tensor.unsqueeze(0)

    # Perform object detection on the frame
    with torch.no_grad():
        results = model(img_tensor)

    # Convert the results to NumPy format and extract bounding boxes, labels, etc.
    boxes = results.xyxy[0].cpu().numpy()  # xyxy format bounding boxes
    labels = results.names  # Class labels

    # Draw the bounding boxes on the frame
    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        label = labels[int(cls_id)]
        confidence = f"{conf:.2f}"

        # Draw a rectangle around the object
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Add a label with the class and confidence
        cv2.putText(frame, f'{label} {confidence}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Write the frame with detections to the output video
    out.write(frame)

    # Optional: Display the frame with detections (can slow down processing for large videos)
    # cv2.imshow('YOLOv10 Object Detection', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Output video saved as {output_path}")