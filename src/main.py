import logging
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import threading
import webbrowser
import uvicorn

from .routers import home, upload_picture, upload_video, contact_us
from .utils.device_setup import setup_device

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Setup device (MPS or CPU)
device = setup_device()

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

# Setup templates
templates = Jinja2Templates(directory="src/templates")

# Include routers
app.include_router(home.router)
app.include_router(upload_picture.router)
app.include_router(upload_video.router)
app.include_router(contact_us.router)

def open_browser():
    webbrowser.open_new("http://127.0.0.1:8000/")

@app.on_event("shutdown")
def shutdown_event():
    logger.info("Shutting down application. Releasing resources.")
    # If you have any resources to release, do it here
    # Example: if you manage a global VideoCapture object, release it
    # cap.release()
    
if __name__ == "__main__":
    threading.Timer(1, open_browser).start()
    logger.info("Starting Uvicorn server...")
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)