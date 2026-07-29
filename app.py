import os
import uuid
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import base64
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

load_dotenv()

from src.attention import AttentionMonitor
from src.utils import process_video_file

app = FastAPI(title="Attention Monitor API")

# Allow CORS for React frontend
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model
MODEL_PATH = os.getenv("MODEL_PATH", "models/phone-classification.pt")
monitor = AttentionMonitor(model_path=MODEL_PATH)

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "Welcome to Attention Monitor API"}

@app.get("/api/livekit-token")
def get_livekit_token(room: str = "attention-room", participant_name: str = "user"):
    # Requires LIVEKIT_API_KEY and LIVEKIT_API_SECRET in .env
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")
    
    if not api_key or not api_secret or api_key == "your_api_key":
        return JSONResponse(status_code=500, content={"error": "LiveKit credentials not configured in .env"})
        
    grant = VideoGrants(room_join=True, room=room)
    token = AccessToken(api_key, api_secret) \
        .with_identity(participant_name) \
        .with_name(participant_name) \
        .with_grants(grant)
        
    return {"token": token.to_jwt()}

@app.post("/api/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        annotated_img = monitor.process_frame(img)
        
        # Encode back to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return {"image": f"data:image/jpeg;base64,{img_base64}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/api/process-video")
async def process_video(file: UploadFile = File(...)):
    try:
        # Save uploaded video to a temporary file
        temp_input_filename = f"temp_{uuid.uuid4().hex}.mp4"
        temp_input_path = os.path.join(OUTPUT_DIR, temp_input_filename)
        
        with open(temp_input_path, "wb") as f:
            f.write(await file.read())
            
        # Process video
        output_path = process_video_file(temp_input_path, OUTPUT_DIR, monitor)
        
        # Clean up input temp file
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
            
        # Return the output file
        return FileResponse(output_path, media_type="video/mp4", filename="processed_video.mp4")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.websocket("/api/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive base64 image string from client
            data = await websocket.receive_text()
            
            # Remove header if present (e.g., "data:image/jpeg;base64,")
            if "," in data:
                data = data.split(",")[1]
                
            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                annotated_img = monitor.process_frame(img)
                _, buffer = cv2.imencode('.jpg', annotated_img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_text(f"data:image/jpeg;base64,{img_base64}")
            else:
                await websocket.send_text("error: could not decode image")
                
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass
