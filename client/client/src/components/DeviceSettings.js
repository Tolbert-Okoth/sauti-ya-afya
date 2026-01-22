/* client/src/components/DeviceSettings.js */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaMicrophoneAlt, FaSlidersH, FaLock } from 'react-icons/fa';

const DeviceSettings = () => {
  const navigate = useNavigate();
  const [mics, setMics] = useState([]);
  const [selectedMic, setSelectedMic] = useState('');
  const [testLevel, setTestLevel] = useState(0);

  // Custom Glass Input Style
  const glassInputStyle = {
      background: 'rgba(255,255,255,0.4)',
      border: '1px solid rgba(255,255,255,0.3)',
      color: '#2d3436',
      backdropFilter: 'blur(5px)'
  };

  useEffect(() => {
    setMics([
        { label: 'Default - Built-in Microphone', id: 'default' },
        { label: 'Headset Microphone (Wired)', id: 'headset' }
    ]);
  }, []);

  const toggleTest = () => {
    const interval = setInterval(() => {
        setTestLevel(Math.random() * 100);
    }, 100);
    setTimeout(() => {
        clearInterval(interval);
        setTestLevel(0);
    }, 3000);
  };

  return (
    <div className="container p-0" style={{ maxWidth: '600px' }}>
      <button className="btn btn-link text-dark-brown text-decoration-none mb-3 p-0 fw-bold" onClick={() => navigate(-1)}>
        <FaArrowLeft /> Back to Settings
      </button>
      
      <div className="d-flex align-items-center mb-4">
          <div className="bg-white rounded-circle p-2 text-dark me-3 shadow-sm">
             <FaMicrophoneAlt />
          </div>
          <h4 className="fw-bold text-dark-brown mb-0">Device & Audio</h4>
      </div>

      <div className="glass-card p-4 mb-4">
            <label className="form-label fw-bold text-dark-brown small text-uppercase">Microphone Input</label>
            <select 
                className="form-select mb-4 shadow-sm" 
                style={glassInputStyle}
                value={selectedMic} 
                onChange={(e) => setSelectedMic(e.target.value)}
            >
                {mics.map(m => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select>

            <button className="btn btn-outline-dark w-100 mb-3 border-2" onClick={toggleTest}>
                <FaMicrophoneAlt className="me-2"/> Test Microphone (3s)
            </button>
            
            {/* Glass Visualizer Bar */}
            <div className="progress" style={{height: '12px', background: 'rgba(0,0,0,0.1)', borderRadius: '6px'}}>
                <div 
                    className={`progress-bar rounded-pill transition-all ${testLevel > 80 ? 'bg-danger' : 'bg-success'}`} 
                    role="progressbar" 
                    style={{width: `${testLevel}%`, transition: 'width 0.1s ease'}}
                ></div>
            </div>
      </div>

      <div className="glass-card p-4">
            <div className="d-flex align-items-center mb-3">
                <FaSlidersH className="me-2 text-muted"/>
                <h6 className="fw-bold text-dark-brown mb-0">Recording Parameters</h6>
            </div>
            
            <div className="mb-4">
                <label className="small text-muted fw-bold">NOISE SENSITIVITY</label>
                <select className="form-select" style={glassInputStyle}>
                    <option>Low (Noisy Environment)</option>
                    <option selected>Medium (Standard)</option>
                    <option>High (Quiet Room)</option>
                </select>
            </div>

            <div className="mb-2">
                <label className="small text-muted fw-bold d-flex align-items-center">
                    MAX DURATION <FaLock className="ms-2 opacity-50" size={10}/>
                </label>
                <input 
                    type="text" 
                    className="form-control text-muted fst-italic" 
                    value="30 seconds" 
                    disabled 
                    style={{...glassInputStyle, background: 'rgba(0,0,0,0.05)', cursor: 'not-allowed'}} 
                />
                <small className="text-muted ms-1" style={{fontSize:'0.7rem'}}>Locked by Admin Policy</small>
            </div>
      </div>
      
      <div className="alert mt-3 small shadow-sm border-0 d-flex align-items-center" 
           style={{background: 'rgba(255,243,205,0.7)', color: '#856404', backdropFilter: 'blur(5px)'}}>
        <span className="me-2 fs-5">⚠</span> Ensure you are in a quiet environment before screening patients.
      </div>
    </div>
  );
};

export default DeviceSettings;