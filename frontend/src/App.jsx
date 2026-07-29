import React, { useState, useEffect } from 'react';
import ImageProcessor from './components/ImageProcessor';
import VideoProcessor from './components/VideoProcessor';
import LiveStream from './components/LiveStream';

function App() {
  const [activeTab, setActiveTab] = useState('photo');
  const [theme, setTheme] = useState('light');

  // Load theme preference from localStorage on mount
  useEffect(() => {
    const savedTheme = localStorage.getItem('theme') || 'light';
    setTheme(savedTheme);
    document.documentElement.setAttribute('data-theme', savedTheme);
  }, []);

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
    document.documentElement.setAttribute('data-theme', newTheme);
  };

  return (
    <div className="app-container">
      <header>
        <h1>Attention Monitor</h1>
        <button className="theme-toggle" onClick={toggleTheme}>
          {theme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode'}
        </button>
      </header>

      <main>
        <div className="tabs">
          <button 
            className={`tab-btn ${activeTab === 'photo' ? 'active' : ''}`}
            onClick={() => setActiveTab('photo')}
          >
            Photo
          </button>
          <button 
            className={`tab-btn ${activeTab === 'video' ? 'active' : ''}`}
            onClick={() => setActiveTab('video')}
          >
            Video
          </button>
          <button 
            className={`tab-btn ${activeTab === 'live' ? 'active' : ''}`}
            onClick={() => setActiveTab('live')}
          >
            Live Camera
          </button>
        </div>

        <div className="tab-content">
          {activeTab === 'photo' && <ImageProcessor />}
          {activeTab === 'video' && <VideoProcessor />}
          {activeTab === 'live' && <LiveStream />}
        </div>
      </main>
    </div>
  );
}

export default App;
