import React, { useState, useEffect, useRef } from 'react';

const LiveStream = () => {
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState(null);
  
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const wsRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);

  const startStream = async () => {
    setError(null);
    try {
      // 1. Get webcam stream
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }

      // 2. Connect to WebSocket
      const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
      wsRef.current = new WebSocket(`${wsUrl}/api/ws/stream`);
      
      wsRef.current.onopen = () => {
        setIsStreaming(true);
        
        // 3. Start sending frames
        intervalRef.current = setInterval(() => {
          if (videoRef.current && canvasRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            const context = canvasRef.current.getContext('2d');
            context.drawImage(videoRef.current, 0, 0, 640, 480);
            
            // Get base64 image data
            const imageData = canvasRef.current.toDataURL('image/jpeg', 0.5);
            wsRef.current.send(imageData);
          }
        }, 100); // 10 FPS
      };

      wsRef.current.onmessage = (event) => {
        if (imgRef.current && event.data.startsWith('data:image')) {
          imgRef.current.src = event.data;
        }
      };

      wsRef.current.onerror = (err) => {
        setError('WebSocket connection error.');
        stopStream();
      };
      
      wsRef.current.onclose = () => {
        stopStream();
      };

    } catch (err) {
      setError('Could not access webcam: ' + err.message);
    }
  };

  const stopStream = () => {
    setIsStreaming(false);
    
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close();
    }
    
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    
    if (imgRef.current) {
      imgRef.current.src = "";
    }
  };

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      stopStream();
    };
  }, []);

  return (
    <div className="card">
      <h2>Live Camera Stream</h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
        Use your webcam to detect phone usage in real-time.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
        <div style={{ position: 'relative', width: '640px', maxWidth: '100%', height: '480px', backgroundColor: '#000', borderRadius: '8px', overflow: 'hidden', marginBottom: '1.5rem', boxShadow: 'var(--shadow-md)' }}>
          {/* Hidden video element to capture webcam */}
          <video ref={videoRef} style={{ display: 'none' }} width="640" height="480" muted></video>
          {/* Hidden canvas to extract frames */}
          <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }}></canvas>
          
          {/* Display element for annotated frames */}
          {isStreaming ? (
            <img ref={imgRef} alt="Live Stream" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
          ) : (
            <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: '#fff', opacity: 0.5 }}>
              Camera Offline
            </div>
          )}
        </div>
        
        {error && <p className="error-msg" style={{ marginBottom: '1rem' }}>{error}</p>}
        
        <button 
          className="btn-primary" 
          onClick={isStreaming ? stopStream : startStream}
          style={{ backgroundColor: isStreaming ? '#e74c3c' : 'var(--primary-blue)' }}
        >
          {isStreaming ? 'Stop Stream' : 'Start Camera'}
        </button>
      </div>
    </div>
  );
};

export default LiveStream;
