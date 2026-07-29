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
  const isProcessingRef = useRef(false);

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
        isProcessingRef.current = false;
        
        // 3. Start sending frames with backpressure
        intervalRef.current = setInterval(() => {
          if (isProcessingRef.current) return; // Skip if backend is still processing the last frame
          
          if (videoRef.current && canvasRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            
            const videoWidth = videoRef.current.videoWidth;
            const videoHeight = videoRef.current.videoHeight;
            
            // Wait until video has actually started playing and has dimensions
            if (videoWidth > 0 && videoHeight > 0) {
              isProcessingRef.current = true;
              
              // Ensure canvas matches video dimensions exactly to prevent squishing (which hurts YOLO accuracy)
              if (canvasRef.current.width !== videoWidth) {
                canvasRef.current.width = videoWidth;
                canvasRef.current.height = videoHeight;
              }
              
              const context = canvasRef.current.getContext('2d');
              context.drawImage(videoRef.current, 0, 0, videoWidth, videoHeight);
              
              // Get base64 image data (Higher quality 0.9 preserves phone details for the model)
              const imageData = canvasRef.current.toDataURL('image/jpeg', 0.9);
              wsRef.current.send(imageData);
            }
          }
        }, 100); // Max 10 FPS
      };

      wsRef.current.onmessage = (event) => {
        isProcessingRef.current = false; // Ready for next frame
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
            <img 
              ref={imgRef} 
              alt="Live Stream" 
              src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" 
              style={{ width: '100%', height: '100%', objectFit: 'contain' }} 
            />
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
