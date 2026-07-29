import React, { useState } from 'react';
import {
  LiveKitRoom,
  useTracks,
  VideoTrack,
} from '@livekit/components-react';
import '@livekit/components-styles';
import { Track } from 'livekit-client';

const LiveStream = () => {
  const [token, setToken] = useState("");
  const [error, setError] = useState(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const serverUrl = import.meta.env.VITE_LIVEKIT_URL;

  const startStream = async () => {
    try {
      // Clear old token to force reconnect
      setToken("");
      setError(null);
      
      const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api/livekit-token`);
      const data = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Failed to fetch token');
      }
      
      setToken(data.token);
      setIsStreaming(true);
    } catch (err) {
      setError(err.message);
    }
  };

  const stopStream = () => {
    setToken("");
    setIsStreaming(false);
  };

  return (
    <div className="card">
      <h2>Live Camera Monitoring</h2>
      <p>Real-time AI analysis with ultra-low latency WebRTC (LiveKit).</p>

      <div style={{ 
        position: 'relative', 
        width: '100%', 
        aspectRatio: '4/3', 
        background: 'var(--bg-color)', 
        borderRadius: '12px', 
        overflow: 'hidden', 
        marginTop: '2rem', 
        border: '1px solid var(--border-color)',
        boxShadow: 'var(--shadow-lg)'
      }}>
        {isStreaming && token ? (
          <LiveKitRoom
            video={true} // Automatically ask for user's camera permissions and publish it
            audio={false}
            token={token}
            serverUrl={serverUrl}
            style={{ height: '100%' }}
            onDisconnected={stopStream}
          >
            <AIStreamViewer />
          </LiveKitRoom>
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

// Component to render ONLY the remote AI annotated track
function AIStreamViewer() {
  const trackRefs = useTracks([Track.Source.Camera]);
  
  // Filter for remote tracks (which will be the AI agent)
  const remoteTracks = trackRefs.filter(t => !t.participant.isLocal);

  return (
    <div style={{ width: '100%', height: '100%', background: 'var(--bg-color)' }}>
      {remoteTracks.length > 0 ? (
        <VideoTrack trackRef={remoteTracks[0]} style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
      ) : (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'var(--text-secondary)', flexDirection: 'column' }}>
          <div className="loader"></div>
          <p style={{ marginTop: '1rem' }}>Waiting for AI Agent to join...</p>
        </div>
      )}
    </div>
  );
}

export default LiveStream;
