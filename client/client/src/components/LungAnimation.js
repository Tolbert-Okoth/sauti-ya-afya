/* client/src/components/LungAnimation.js */
import React from 'react';

const LungAnimation = ({ isRecording, volume }) => {
  // Volume scaling logic
  const scale = 1 + (volume / 255) * 1.5; 

  return (
    <div className="d-flex justify-content-center align-items-center my-4" style={{ height: '160px' }}>
      {/* Outer Glow Ring */}
      <div className="position-absolute rounded-circle"
           style={{
               width: '120px', 
               height: '120px', 
               border: '1px solid rgba(255,255,255,0.2)',
               transform: `scale(${isRecording ? 1.2 : 1})`,
               transition: 'transform 0.5s ease'
           }}
      ></div>

      <div 
        className="d-flex align-items-center justify-content-center text-white fw-bold shadow-lg"
        style={{
          width: '100px',
          height: '100px',
          borderRadius: '50%',
          // Dynamic Gradient
          background: isRecording 
            ? 'radial-gradient(circle at 30% 30%, #ff6b6b, #c0392b)' 
            : 'radial-gradient(circle at 30% 30%, #95a5a6, #7f8c8d)',
          transform: `scale(${isRecording ? scale : 1})`,
          transition: 'transform 0.05s ease-out, background 0.3s',
          // Holographic Glow
          boxShadow: isRecording 
            ? '0 0 30px rgba(220, 53, 69, 0.8), inset 0 0 20px rgba(255,255,255,0.5)' 
            : '0 4px 15px rgba(0,0,0,0.2)',
          zIndex: 2
        }}
      >
        {isRecording ? (
            <div className="d-flex flex-column align-items-center">
                <div className="spinner-grow spinner-grow-sm mb-1" role="status"></div>
                <small style={{fontSize: '0.6rem'}}>REC</small>
            </div>
        ) : "Ready"}
      </div>
    </div>
  );
};

export default LungAnimation;