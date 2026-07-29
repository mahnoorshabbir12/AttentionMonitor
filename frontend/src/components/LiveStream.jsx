import React, { useState, useRef } from 'react';

const LiveStream = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const isProcessingRef = useRef(false);
  const animFrameRef = useRef(null);

  const startStream = async () => {
    setError(null);
    try {
      // 1. Get webcam stream
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }

      // 2. Connect to WebSocket
      const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
      wsRef.current = new WebSocket(`${wsUrl}/api/ws/stream`);

      wsRef.current.onopen = () => {
        setIsStreaming(true);
        isProcessingRef.current = false;
        sendFrame();  // Start the send loop
      };

      wsRef.current.onmessage = (event) => {
        isProcessingRef.current = false;  // Ready for next frame
        if (imgRef.current && event.data.startsWith('data:image')) {
          imgRef.current.src = event.data;
        }
      };

      wsRef.current.onerror = () => setError('WebSocket connection failed. Is the backend running?');
      wsRef.current.onclose = () => setIsStreaming(false);

    } catch (err) {
      setError(`Camera error: ${err.message}`);
    }
  };

  const sendFrame = () => {
    // Use requestAnimationFrame for smooth scheduling
    animFrameRef.current = requestAnimationFrame(() => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

      // Only send if backend has finished processing the last frame (backpressure)
      if (!isProcessingRef.current && videoRef.current && canvasRef.current) {
        const video = videoRef.current;
        const vw = video.videoWidth;
        const vh = video.videoHeight;

        if (vw > 0 && vh > 0) {
          isProcessingRef.current = true;

          // Resize canvas to match video (preserving aspect ratio)
          if (canvasRef.current.width !== vw || canvasRef.current.height !== vh) {
            canvasRef.current.width = vw;
            canvasRef.current.height = vh;
          }

          const ctx = canvasRef.current.getContext('2d');
          ctx.drawImage(video, 0, 0, vw, vh);

          // Send as JPEG at decent quality
          const data = canvasRef.current.toDataURL('image/jpeg', 0.75);
          wsRef.current.send(data);
        }
      }

      sendFrame(); // Schedule next frame
    });
  };

  const stopStream = () => {
    if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
    if (wsRef.current) wsRef.current.close();
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop());
      streamRef.current = null;
    }
    setIsStreaming(false);
  };

  return (
    <div className="card">
      <h2>Live Camera Monitoring</h2>
      <p>Real-time phone detection using your webcam.</p>

      <div style={{
        position: 'relative',
        width: '100%',
        aspectRatio: '4/3',
        background: '#000',
        borderRadius: '12px',
        overflow: 'hidden',
        marginTop: '2rem',
        border: '1px solid var(--border-color)',
        boxShadow: 'var(--shadow-lg)',
      }}>
        {/* Hidden video element to capture webcam */}
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          style={{ display: 'none' }}
        />
        {/* Hidden canvas for capturing frames */}
        <canvas ref={canvasRef} style={{ display: 'none' }} />

        {/* Display the AI-annotated result */}
        {isStreaming ? (
          <img
            ref={imgRef}
            alt="Live Stream"
            src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs="
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        ) : (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'var(--text-secondary)' }}>
            Camera Offline
          </div>
        )}
      </div>

      <div style={{ marginTop: '2rem', display: 'flex', justifyContent: 'center', gap: '1rem' }}>
        {!isStreaming ? (
          <button className="btn-primary" onClick={startStream}>
            Start Camera
          </button>
        ) : (
          <button className="btn-primary" onClick={stopStream} style={{ background: 'var(--danger-color)' }}>
            Stop Camera
          </button>
        )}
      </div>

      {error && <div className="error-msg">{error}</div>}
    </div>
  );
};

export default LiveStream;
