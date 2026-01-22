import React from 'react';

const AudioVisualizer = ({ spectrogramData, riskLevel }) => {
  if (!spectrogramData) return null;

  // Color coding the border based on risk
  const borderColor = 
    riskLevel === 'High' ? '#dc3545' : 
    riskLevel === 'Medium' ? '#ffc107' : 
    '#2ecc71';

  return (
    <div className="position-relative rounded overflow-hidden shadow-sm" 
         style={{ border: `2px solid ${borderColor}`, background: '#000' }}>
      
      {/* 1. The Spectrogram Image */}
      <img 
        src={spectrogramData} 
        alt="Mel Spectrogram Analysis" 
        style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }} 
      />

      {/* 2. Medical Overlay (Grid & Labels) */}
      <div className="position-absolute top-0 start-0 w-100 h-100 p-2 d-flex flex-column justify-content-between pointer-events-none">
        <div className="d-flex justify-content-between text-white small opacity-75" style={{fontSize: '0.7rem', fontFamily: 'monospace'}}>
          <span>16kHz</span>
          <span>MEL-SPECTROGRAM</span>
        </div>
        <div className="d-flex justify-content-between text-white small opacity-75" style={{fontSize: '0.7rem', fontFamily: 'monospace'}}>
          <span>0.0s</span>
          <span>5.0s</span>
        </div>
      </div>

      {/* 3. Scanline Effect */}
      <div 
        className="position-absolute w-100 bg-white opacity-25"
        style={{
            height: '2px',
            top: '0',
            left: '0',
            animation: 'scanline 2s linear infinite',
            boxShadow: '0 0 10px 2px rgba(255, 255, 255, 0.5)'
        }}
      />
      
      <style>{`
        @keyframes scanline {
            0% { top: 0%; }
            100% { top: 100%; }
        }
      `}</style>
    </div>
  );
};

export default AudioVisualizer;