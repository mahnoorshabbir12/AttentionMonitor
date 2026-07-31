"""
LiveKit Video Processing Agent
Connects directly to a LiveKit room, subscribes to the user's camera,
runs YOLO inference on each frame, and publishes the annotated video back.
"""
import asyncio
import os
import cv2
import numpy as np
import logging
from dotenv import load_dotenv

load_dotenv()

from livekit import rtc, api
from src.inference.detection_pipeline import AttentionMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("livekit-agent")

LIVEKIT_URL = os.getenv("LIVEKIT_URL")
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
ROOM_NAME = "attention-room"
AGENT_IDENTITY = "ai-agent"


async def generate_agent_token() -> str:
    """Generate an access token for the agent to join the room."""
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET) \
        .with_identity(AGENT_IDENTITY) \
        .with_name("AI Monitor") \
        .with_grants(api.VideoGrants(
            room_join=True,
            room=ROOM_NAME,
            can_publish=True,
            can_subscribe=True,
        ))
    return token.to_jwt()


async def process_track(room: rtc.Room, track: rtc.RemoteVideoTrack, monitor: AttentionMonitor):
    """Subscribe to a remote video track, process frames with YOLO, and publish back."""
    logger.info(f"Processing video track: {track.sid}")

    # Create a video source to publish annotated frames back
    video_source = rtc.VideoSource(width=640, height=480)
    local_track = rtc.LocalVideoTrack.create_video_track("ai-annotated", video_source)

    publish_options = rtc.TrackPublishOptions()
    publish_options.source = rtc.TrackSource.SOURCE_CAMERA
    await room.local_participant.publish_track(local_track, publish_options)
    logger.info("Published AI-annotated track to room.")

    # State for inference task
    inference_task = None

    # Stream incoming frames
    video_stream = rtc.VideoStream(track)
    async for event in video_stream:
        try:
            frame = event.frame

            # Convert to RGBA numpy array
            argb_frame = frame.convert(rtc.VideoBufferType.RGBA)
            arr = np.frombuffer(argb_frame.data, dtype=np.uint8)
            img_rgba = arr.reshape((argb_frame.height, argb_frame.width, 4))

            # RGBA -> BGR for OpenCV / YOLO
            img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)

            # Frame dropping logic: Run inference only if not already running
            if inference_task is None or inference_task.done():
                # Start new inference in background
                inference_task = asyncio.create_task(
                    asyncio.to_thread(monitor.process_frame, img_bgr, True)
                )
                # Display current tracked boxes instantly (using the locked state from previous frames)
                annotated = monitor.process_frame(img_bgr, run_inference=False)
            else:
                # Inference is busy. Drop this frame's inference, but render current tracks and advance miss counts
                annotated = monitor.process_frame(img_bgr, run_inference=False)

            # BGR -> RGBA for LiveKit
            out_rgba = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGBA)

            out_frame = rtc.VideoFrame(
                width=out_rgba.shape[1],
                height=out_rgba.shape[0],
                type=rtc.VideoBufferType.RGBA,
                data=out_rgba.tobytes(),
            )
            video_source.capture_frame(out_frame)

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            continue


async def main():
    if not all([LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET]):
        logger.error("Missing LiveKit credentials in .env file!")
        return

    # Load model once
    model_path = os.getenv("MODEL_PATH", "models/exp.onnx")
    monitor = AttentionMonitor(model_path)
    logger.info(f"YOLO model loaded: {model_path}")

    token = await generate_agent_token()

    room = rtc.Room()

    # Set up track subscription handler
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            logger.info(f"Video track received from: {participant.identity}")
            asyncio.ensure_future(process_track(room, track, monitor))

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant joined: {participant.identity}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant left: {participant.identity}")

    # Connect to the room
    logger.info(f"Connecting to room '{ROOM_NAME}' at {LIVEKIT_URL}...")
    await room.connect(LIVEKIT_URL, token)
    logger.info(f"Connected! Agent is live in room '{ROOM_NAME}'.")
    logger.info("Waiting for participants to join...")

    # Keep alive until interrupted
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await room.disconnect()
        logger.info("Agent disconnected.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user.")
