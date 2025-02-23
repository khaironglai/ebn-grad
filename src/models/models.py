from ultralytics import YOLOv10
import logging
from collections import deque
from datetime import timedelta

from ..utils.device_setup import setup_device

logger = logging.getLogger(__name__)

device = setup_device()
selected_model = None
model_loaded = False

# Shared variables
category_dict = {
    0: "Grade A Birdnest",
    1: "Grade B Birdnest",
    2: "Grade C Birdnest",
    3: "Unknown",
}

birdnest_descriptions = {
    'Grade A Birdnest': "Grade A edible bird nests are of the highest quality, featuring a clean, off-white color, minimal impurities, and thick, tightly woven strands. These nests often have a perfect cup shape with very few feathers, making them the most sought-after.",
    'Grade B Birdnest': "Grade B edible bird nests are considered mid-tier in quality, featuring an off-white, light yellow, or slightly greyish color. These nests often contain minor impurities like feathers and dirt, although they are generally clean after washing. The structure of Grade B nests may exhibit slight inconsistencies or imperfections, with the strands not being as thick or tightly woven as those found in higher-grade nests.",
    'Grade C Birdnest': "Grade C edible bird nests are of the lowest quality among the three grades. These nests may have a yellowish or greyish color and contain more impurities and feathers. The strands are often thinner and less tightly woven, and the nests may have irregular shapes and more visible imperfections.",
    "Unknown": "other objects",
}

latest_detected_class = None
latest_confidence_score = 0.0
recent_images = deque(maxlen=5)
accumulated_counts = {
    "Grade A Birdnest": 0,
    "Grade B Birdnest": 0,
    "Grade C Birdnest": 0,
}

# Cooldown mechanism
last_count_time = {
    "Grade A Birdnest": None,
    "Grade B Birdnest": None,
    "Grade C Birdnest": None,
}
cooldown_period = timedelta(seconds=10)  # Adjust as needed

def load_model(model_name: str = "yolov10", model_path: str = ""):
    global selected_model, model_loaded
    if not model_path:
        model_path = "src/models/current_best.pt"  # Set default to 'current_best.pt'
    try:
        logger.info(f"Attempting to load model '{model_name}' from path: {model_path}")
        if model_name in ["yolov8", "yolov9", "yolov10"]:
            selected_model = YOLOv10(model_path).to(device)
            model_loaded = True
            logger.info(f"{model_name} loaded successfully from {model_path}.")
        else:
            raise ValueError("Unsupported model selected.")
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}' from '{model_path}': {e}")
        raise