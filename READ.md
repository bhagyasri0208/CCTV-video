**Bird Counting & Weight Estimation**

**Overview**:
This project processes poultry CCTV footage to detect, track,
and estimate bird count and weight using computer vision.
I used YOLOv8 for detection and ByteTrack for tracking.
Bird count is computed per frame.
Weight is estimated using bounding-box area as a visual proxy.
FastAPI processes videos and returns structured JSON output.

**Approach**:
- YOLOv8 for bird detection
- ByteTrack for stable tracking IDs
- Unique tracking IDs used for counting
- Bounding box area used as a weight proxy

**API**:
POST /analyze_video
Input: Poultry video file
Output: JSON with bird count over time and weight index

**Setup:**
1. Create virtual environment
2. pip install -r requirements.txt
3. uvicorn app:app --reload
4. Open http://127.0.0.1:8000/docs

**Outputs:**
- Annotated video saved in outputs
- JSON response from API
