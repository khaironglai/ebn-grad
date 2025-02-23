import cv2
import torch
import torchvision.ops as ops
import supervision as sv
import logging
from datetime import datetime
import os

from ..models import models

logger = logging.getLogger(__name__)

def gen_frames():
    global latest_detected_class, latest_confidence_score

    if not models.model_loaded:
        logger.warning("Model not loaded. Cannot generate frames.")
        return

    cap = cv2.VideoCapture(0)
    logger.info("Webcam video capture started.")

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        logger.error("Error: Could not open webcam.")
        return

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            logger.error("Failed to read frame from webcam.")
            break

        try:
            results = models.selected_model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            logger.debug("Frame processed successfully.")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            break

        boxes = detections.xyxy
        scores = detections.confidence

        # Convert boxes and scores to PyTorch tensors
        boxes = torch.tensor(boxes, dtype=torch.float32)
        scores = torch.tensor(scores, dtype=torch.float32)

        # Apply Non-maximum suppression
        indices = ops.nms(boxes, scores, iou_threshold=0.5)

        boxes = boxes[indices]
        class_ids = torch.tensor(detections.class_id)[indices]
        confidences = scores[indices]

        detected_class = None
        highest_confidence = 0

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            if confidence.item() >= 0.1:
                class_name = models.category_dict.get(class_id.item(), "Unknown")
                if confidence.item() > highest_confidence:
                    highest_confidence = confidence.item()
                    detected_class = class_name
                cv2.rectangle(
                    frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (255, 255, 255),  # White color for the bounding box
                    2,  # Thickness of the bounding box
                )

                cv2.putText(
                    frame,
                    f"{class_name}: {confidence:.2f}",
                    (int(box[0]), int(box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 255, 255),
                    2,
                )

            if detected_class is not None and frame_count % 30 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"app/static/captured_images/{detected_class}_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                models.recent_images.append(filename)
                logger.info(f"Captured image: {filename}")

        # Update the accumulated counts using the cooldown mechanism
        if detected_class in models.accumulated_counts:
            now = datetime.now()
            if (
                models.last_count_time[detected_class] is None
                or now - models.last_count_time[detected_class] > models.cooldown_period
            ):
                models.accumulated_counts[detected_class] += 1
                models.last_count_time[detected_class] = now
                logger.info(f"Accumulated count for {detected_class}: {models.accumulated_counts[detected_class]}")

        frame_count += 1

        latest_detected_class = detected_class
        latest_confidence_score = highest_confidence

        # Encode the frame
        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    logger.info("Webcam video capture ended.")

def process_video_background(temp_video_path, task_id, task_status, task_results):
    cap = cv2.VideoCapture(temp_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Generate a unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_video_filename = f"processed_video_{timestamp}_{task_id}.mp4"
    out_video_path = f"app/static/processed_videos/{out_video_filename}"

    out = cv2.VideoWriter(out_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        results = models.selected_model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        boxes = detections.xyxy
        class_ids = detections.class_id
        confidences = detections.confidence

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            if confidence >= 0.1:
                class_name = models.category_dict.get(class_id, "Unknown")
                cv2.rectangle(
                    frame,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"{class_name}: {confidence:.2f}",
                    (int(box[0]), int(box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )
        # Write the processed frame to the output video
        out.write(frame)

    cap.release()
    out.release()
    os.remove(temp_video_path)  # Clean up the temporary video file

    # Save the result HTML snippet
    result_html = f"""
    <div class="result-video">
        <h2>Processed Video</h2>
        <video width="640" height="480" controls>
            <source src="/static/processed_videos/{out_video_filename}" type="video/mp4">
            Your browser does not support the video tag.
        </video>
    </div>
    """
    task_status[task_id] = "Completed"
    task_results[task_id] = result_html