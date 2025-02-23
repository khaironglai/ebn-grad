import logging
from fastapi import UploadFile
import cv2
import numpy as np
import base64
from datetime import datetime
import os
import supervision as sv

from ..models import models

logger = logging.getLogger(__name__)

async def process_uploaded_image(file: UploadFile):
    global latest_detected_class, latest_confidence_score, accumulated_counts, last_count_time, recent_images

    logger.info(f"Processing uploaded image: {file.filename}")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if not models.model_loaded or models.selected_model is None:
            logger.error("Model not loaded. Cannot process image.")
            return """
                <div class="result-image">
                    <h2>Processed Image</h2>
                    <p>Model not loaded. Please select a model first.</p>
                </div>
            """

        # Process the image using the selected model
        results = models.selected_model(img)[0]
        detections = sv.Detections.from_ultralytics(results)

        boxes = detections.xyxy
        class_ids = detections.class_id
        confidences = detections.confidence

        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            if confidence >= 0.1:
                class_name = models.category_dict.get(class_id, "Unknown")
                cv2.rectangle(
                    img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    img,
                    f"{class_name}: {confidence:.2f}",
                    (int(box[0]), int(box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (255, 0, 0),
                    2,
                )

                # Update detections
                models.latest_detected_class = class_name
                models.latest_confidence_score = confidence

                # Update accumulated counts with cooldown
                if class_name in models.accumulated_counts:
                    now = datetime.now()
                    if (
                        models.last_count_time[class_name] is None
                        or now - models.last_count_time[class_name] > models.cooldown_period
                    ):
                        models.accumulated_counts[class_name] += 1
                        models.last_count_time[class_name] = now
                        logger.info(f"Accumulated count for {class_name}: {models.accumulated_counts[class_name]}")

        # Encode the image to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode('utf-8')

        logger.info("Image processed and encoded successfully.")

        # Return an HTML snippet displaying the image
        return f"""
        <div class="result-image">
            <h2>Processed Image</h2>
            <img src="data:image/jpeg;base64,{img_str}" alt="Processed Image">
        </div>
        """

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return """
            <div class="result-image">
                <h2>Processed Image</h2>
                <p>An error occurred during processing.</p>
            </div>
        """

def get_description():
    if models.latest_detected_class is None:
        return {
            "grade": "No detection",
            "description": "No birdnest detected.",
            "confidence": 0.0,
            "accumulatedCounts": models.accumulated_counts,
        }
    return {
        "grade": models.latest_detected_class,
        "description": models.birdnest_descriptions.get(models.latest_detected_class, "No description available."),
        "confidence": models.latest_confidence_score,
        "accumulatedCounts": models.accumulated_counts,
    }

def get_captured_images():
    images = list(models.recent_images)  # Convert deque to list
    images = [os.path.basename(image) for image in images]
    return images