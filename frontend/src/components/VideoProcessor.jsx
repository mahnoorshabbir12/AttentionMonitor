import React, { useState, useRef } from 'react';

const VideoProcessor = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [resultVideoUrl, setResultVideoUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = (file) => {
    if (file && file.type.startsWith('video/')) {
      setSelectedFile(file);
      setResultVideoUrl(null);
      setError(null);
    }
  };

  const handleFileChange = (e) => {
    handleFile(e.target.files[0]);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const processVideo = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/process-video`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process video');
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setResultVideoUrl(url);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Video Analysis</h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
        Upload a video to detect phone usage throughout the clip. Processing may take some time.
      </p>

      {!selectedFile && !resultVideoUrl ? (
        <div 
          className={`upload-zone ${isDragging ? 'dragging' : ''}`}
          onClick={handleUploadClick}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          style={{ borderColor: isDragging ? 'var(--primary-blue)' : 'var(--border-color)', background: isDragging ? 'rgba(74, 144, 226, 0.05)' : 'var(--bg-color)' }}
        >
          <input
            type="file"
            accept="video/*"
            style={{ display: 'none' }}
            ref={fileInputRef}
            onChange={handleFileChange}
          />
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--primary-blue)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
            <line x1="7" y1="2" x2="7" y2="22"></line>
            <line x1="17" y1="2" x2="17" y2="22"></line>
            <line x1="2" y1="12" x2="22" y2="12"></line>
            <line x1="2" y1="7" x2="7" y2="7"></line>
            <line x1="2" y1="17" x2="7" y2="17"></line>
            <line x1="17" y1="17" x2="22" y2="17"></line>
            <line x1="17" y1="7" x2="22" y2="7"></line>
          </svg>
          <p>Click or drag video to upload</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          {resultVideoUrl ? (
            <video src={resultVideoUrl} controls className="result-video" style={{ maxHeight: '400px', marginBottom: '1rem' }} />
          ) : (
            <div style={{ marginBottom: '1rem', color: 'var(--primary-blue)', fontWeight: '500' }}>
              Selected File: {selectedFile.name}
            </div>
          )}
          
          <div style={{ display: 'flex', gap: '1rem' }}>
            <button className="btn-primary" style={{ background: 'var(--text-secondary)' }} onClick={() => { setSelectedFile(null); setResultVideoUrl(null); }}>
              {resultVideoUrl ? 'Upload Another' : 'Clear'}
            </button>
            {!resultVideoUrl && (
              <button className="btn-primary" onClick={processVideo} disabled={loading}>
                {loading ? 'Processing...' : 'Analyze Video'}
              </button>
            )}
          </div>
        </div>
      )}

      {loading && (
        <div style={{ textAlign: 'center' }}>
          <div className="loader"></div>
          <p style={{ color: 'var(--text-secondary)' }}>Processing video frames... Please wait.</p>
        </div>
      )}
      {error && <p className="error-msg">{error}</p>}
    </div>
  );
};

export default VideoProcessor;
