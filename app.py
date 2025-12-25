from fastapi import FastAPI, UploadFile, File
import shutil
import json
import os
from analyze import analyze_video

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Bird Counting API is running"}

@app.post("/analyze_video")
async def analyze_video_api(video: UploadFile = File(...)):

    # Create folders if not exist
    os.makedirs("sample_videos", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Save uploaded video
    input_path = f"sample_videos/{video.filename}"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Analyze video
    result = analyze_video(input_path)

    # Save JSON output
    with open("outputs/result.json", "w") as f:
        json.dump(result, f, indent=4)

    return result
