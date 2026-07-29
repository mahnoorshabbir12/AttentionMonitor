import React, { useState, useRef } from 'react';

const ImageProcessor = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResultImage(null);
      setError(null);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  const processImage = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
      const response = await fetch(`${apiUrl}/api/process-image`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to process image');
      }

      const data = await response.json();
      setResultImage(data.image);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2>Photo Analysis</h2>
      <p style={{ color: 'var(--text-secondary)', marginBottom: '1.5rem' }}>
        Upload a photo to detect if a phone is being used.
      </p>

      {!preview ? (
        <div className="upload-zone" onClick={handleUploadClick}>
          <input
            type="file"
            accept="image/*"
            style={{ display: 'none' }}
            ref={fileInputRef}
            onChange={handleFileChange}
          />
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="var(--primary-blue)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
          <p>Click or drag image to upload</p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
          <img
            src={resultImage || preview}
            alt="Preview"
            className="result-image"
            style={{ maxHeight: '400px', marginBottom: '1rem' }}
          />
          
          <div style={{ display: 'flex', gap: '1rem' }}>
            <button className="btn-primary" style={{ background: 'var(--text-secondary)' }} onClick={() => { setPreview(null); setSelectedFile(null); setResultImage(null); }}>
              Clear
            </button>
            <button className="btn-primary" onClick={processImage} disabled={loading || resultImage}>
              {loading ? 'Processing...' : 'Analyze Photo'}
            </button>
          </div>
        </div>
      )}

      {loading && <div className="loader"></div>}
      {error && <p className="error-msg">{error}</p>}
    </div>
  );
};

export default ImageProcessor;
