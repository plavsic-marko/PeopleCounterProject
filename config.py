import os
from dotenv import load_dotenv

# Učitavanje .env fajla
load_dotenv()

# YOLO Model Config
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8m.pt")

# Putanja do videa
VIDEO_PATH = os.getenv("VIDEO_PATH", "videos/sample.mp4")

# Gde se čuva CSV izveštaj
CSV_OUTPUT_PATH = os.getenv("CSV_OUTPUT_PATH", "output/counts.csv")

# Da li koristimo GPU
USE_GPU = os.getenv("USE_GPU", "True") == "True"
