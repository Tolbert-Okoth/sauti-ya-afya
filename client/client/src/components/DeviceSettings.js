/* client/src/components/DeviceSettings.js */
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaMicrophoneAlt, FaSlidersH, FaLock, FaExclamationTriangle } from 'react-icons/fa';
import { useTranslation } from '../hooks/useTranslation';

const DeviceSettings = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();
  
  const [mics, setMics] = useState([]);
  const [selectedMic, setSelectedMic] = useState('');
  const [testLevel, setTestLevel] = useState(0);
  const [isListening, setIsListening] = useState(false);
  const [permissionError, setPermissionError] = useState(false);

  // Refs to maintain audio context without re-renders
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);
  const rafIdRef = useRef(null);

  // Custom Glass Input Style
  const glassInputStyle = {
      background: 'rgba(255,255,255,0.4)',
      border: '1px solid rgba(255,255,255,0.3)',
      color: '#2d3436',
      backdropFilter: 'blur(5px)'
  };

  // 1. Fetch Real Hardware Devices
  useEffect(() => {
    const getDevices = async () => {
      try {
        // Request permission briefly to unlock device labels
        await navigator.mediaDevices.getUserMedia({ audio: true });
        
        const devices = await navigator.mediaDevices.enumerateDevices();
        const audioInputs = devices
          .filter(device => device.kind === 'audioinput')
          .map(dev => ({ 
            label: dev.label || `Microphone ${dev.deviceId.slice(0,5)}...`, 
            id: dev.deviceId 
          }));
          
        setMics(audioInputs);
        if (audioInputs.length > 0) setSelectedMic(audioInputs[0].id);
        setPermissionError(false);
      } catch (err) {
        console.error("Mic permission error:", err);
        setPermissionError(true);
      }
    };
    getDevices();

    // Cleanup on unmount
    return () => stopListening();
  }, []);

  // 2. Real-Time Audio Processing Logic
  const startListening = async () => {
    if (isListening) {
        stopListening();
        return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: { deviceId: selectedMic ? { exact: selectedMic } : undefined } 
      });

      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      const bufferLength = analyserRef.current.frequencyBinCount;
      dataArrayRef.current = new Uint8Array(bufferLength);

      setIsListening(true);
      draw(); // Start the animation loop
    } catch (err) {
      console.error("Error accessing mic:", err);
      setPermissionError(true);
    }
  };

  const stopListening = () => {
    if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
    if (audioContextRef.current) audioContextRef.current.close();
    setIsListening(false);
    setTestLevel(0);
  };

  // 3. The Visualizer Loop (60fps)
  const draw = () => {
    if (!analyserRef.current) return;

    analyserRef.current.getByteFrequencyData(dataArrayRef.current);

    // Calculate Average Volume (RMS approximation)
    let sum = 0;
    const len = dataArrayRef.current.length;
    for (let i = 0; i < len; i++) {
        sum += dataArrayRef.current[i];
    }
    const average = sum / len;

    // Normalize signal for UI (Standard speech is usually 30-70 range)
    // Multiplied by 1.5 to make it responsive
    const visualLevel = Math.min((average / 128) * 100 * 1.5, 100);
    
    setTestLevel(visualLevel);
    rafIdRef.current = requestAnimationFrame(draw);
  };

  return (
    <div className="container p-0" style={{ maxWidth: '600px' }}>
      <button className="btn btn-link text-dark-brown text-decoration-none mb-3 p-0 fw-bold" onClick={() => navigate(-1)}>
        <FaArrowLeft /> {t('back')}
      </button>
      
      <div className="d-flex align-items-center mb-4">
          <div className="bg-white rounded-circle p-2 text-dark me-3 shadow-sm">
             <FaMicrophoneAlt />
          </div>
          <h4 className="fw-bold text-dark-brown mb-0">{t('menu_device')}</h4>
      </div>

      {/* Permission Error Alert */}
      {permissionError && (
        <div className="alert alert-danger d-flex align-items-center mb-4 shadow-sm">
            <FaExclamationTriangle className="me-2"/>
            <div>
                <strong>Access Denied:</strong> Please allow microphone access in your browser settings.
            </div>
        </div>
      )}

      <div className="glass-card p-4 mb-4">
            <label className="form-label fw-bold text-dark-brown small text-uppercase">Microphone Input</label>
            <select 
                className="form-select mb-4 shadow-sm" 
                style={glassInputStyle}
                value={selectedMic} 
                onChange={(e) => {
                    setSelectedMic(e.target.value);
                    stopListening(); // Reset if mic changes
                }}
            >
                {mics.length > 0 ? (
                    mics.map(m => <option key={m.id} value={m.id}>{m.label}</option>)
                ) : (
                    <option>Loading devices...</option>
                )}
            </select>

            <button 
                className={`btn w-100 mb-3 border-2 ${isListening ? 'btn-danger' : 'btn-outline-dark'}`} 
                onClick={startListening}
            >
                <FaMicrophoneAlt className="me-2"/> 
                {isListening ? 'Stop Test' : 'Test Microphone'}
            </button>
            
            {/* Real Audio Visualizer Bar */}
            <div className="d-flex justify-content-between small text-muted mb-1">
                <span>Silence</span>
                <span>Loud</span>
            </div>
            <div className="progress" style={{height: '12px', background: 'rgba(0,0,0,0.1)', borderRadius: '6px'}}>
                <div 
                    className={`progress-bar rounded-pill transition-all ${testLevel > 60 ? 'bg-danger' : 'bg-success'}`} 
                    role="progressbar" 
                    style={{width: `${testLevel}%`, transition: 'width 0.05s linear'}}
                ></div>
            </div>
            
            {/* Real-time Environmental Feedback */}
            <div className="text-center mt-2" style={{height: '20px'}}>
                {isListening && (
                    <small className={`fw-bold ${testLevel > 60 ? 'text-danger' : 'text-success'}`}>
                        {testLevel > 60 ? 'Environment Too Noisy!' : 'Good Audio Levels'}
                    </small>
                )}
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