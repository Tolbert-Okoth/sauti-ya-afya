/* client/src/components/SmartRecorder.js */
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { auth } from '../firebase'; 
import config from '../config'; 
import LungAnimation from './LungAnimation';
import AudioVisualizer from './AudioVisualizer'; 
// import { useTranslation } from '../hooks/useTranslation'; // Uncomment if using
import { 
  FaMicrophone, FaStop, FaNotesMedical, FaCheckCircle, 
  FaStethoscope, FaPhoneAlt, FaCalendarAlt, FaSignOutAlt 
} from 'react-icons/fa';

const SmartRecorder = ({ onLogout }) => {
  // const { t } = useTranslation();
  
  // Recorder State
  const [isRecording, setIsRecording] = useState(false);
  const [volume, setVolume] = useState(0);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState(''); 
  
  // Admin Config & Counties
  const [systemConfig, setSystemConfig] = useState({ confidence_threshold: 0.75 });
  const [counties, setCounties] = useState([]); 

  // Patient Form State
  const [patientData, setPatientData] = useState({
    name: '',
    dob: '', 
    location: '', 
    phone: '', 
    symptoms: '' 
  });

  const [errors, setErrors] = useState({});
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const animationRef = useRef(null);

  const glassInputStyle = {
      background: 'rgba(255, 255, 255, 0.7)', 
      border: '1px solid rgba(255, 255, 255, 0.5)',
      color: '#2d3436',
      borderRadius: '8px',
      padding: '10px 12px',
      outline: 'none'
  };

  useEffect(() => {
    // 🚀 NEW: WAKE UP SERVER IMMEDIATELY
    // This reduces "perceived latency" by starting the cold start early
    const wakeUpServer = async () => {
        try {
            console.log("🔥 Warming up AI Engine...");
            await axios.get('https://sauti-ya-afya-1.onrender.com/');
            console.log("✅ AI Engine is Awake & Ready!");
        } catch (e) {
            console.log("⚠️ Server waking up...");
        }
    };
    wakeUpServer(); 

    const initData = async () => {
        try {
            if (auth.currentUser) {
                const token = await auth.currentUser.getIdToken();
                const configRes = await axios.get(`${config.API_BASE_URL}/system-config`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                setSystemConfig(configRes.data);

                const countiesRes = await axios.get(`${config.API_BASE_URL}/counties`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                setCounties(countiesRes.data);
            }
        } catch (err) {
            console.error("Failed to load init data", err);
        }
    };
    initData();
  }, []);

  const handleInputChange = (e) => {
    setPatientData({ ...patientData, [e.target.name]: e.target.value });
    if (errors[e.target.name]) setErrors({ ...errors, [e.target.name]: null });
  };

  const validateForm = () => {
      let newErrors = {};
      let isValid = true;
      const phoneRegex = /^(?:254|\+254|0)?(7|1)\d{8}$/;
      
      if (!patientData.name.trim()) { newErrors.name = "Patient name is required"; isValid = false; }
      if (!patientData.phone) {
          newErrors.phone = "Phone number is required";
          isValid = false;
      } else if (!phoneRegex.test(patientData.phone.replace(/\s+/g, ''))) {
          newErrors.phone = "Invalid Kenyan number (e.g., 0712345678)";
          isValid = false;
      }
      if (!patientData.dob) {
          newErrors.dob = "Date of Birth is required";
          isValid = false;
      } else {
          const selectedDate = new Date(patientData.dob);
          const today = new Date();
          if (selectedDate > today) {
              newErrors.dob = "Date cannot be in the future";
              isValid = false;
          }
      }
      if (!patientData.location) { newErrors.location = "Please select a county"; isValid = false; }
      setErrors(newErrors);
      return isValid;
  };

  const calculateAge = (dob) => {
      if(!dob) return "0y";
      const birthDate = new Date(dob);
      const difference = Date.now() - birthDate.getTime();
      const ageDate = new Date(difference); 
      const years = Math.abs(ageDate.getUTCFullYear() - 1970);
      return `${years}y`;
  };

  const startRecording = async () => {
    if (!validateForm()) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const analyser = audioContext.createAnalyser();
      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);
      analyser.fftSize = 256;
      const dataArray = new Uint8Array(analyser.frequencyBinCount);

      const updateVolume = () => {
        analyser.getByteFrequencyData(dataArray);
        const avg = dataArray.reduce((a, b) => a + b) / dataArray.length;
        setVolume(avg);
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
          animationRef.current = requestAnimationFrame(updateVolume);
        }
      };

      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };
      mediaRecorderRef.current.onstop = handleStop;
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setAnalysis(null);
      setSaveStatus('');
      updateVolume();
    } catch (err) {
      console.error(err);
      alert("Microphone access denied. Please allow permissions.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
    setIsRecording(false);
    setVolume(0);
    cancelAnimationFrame(animationRef.current);
  };

  const handleStop = async () => {
    setLoading(true);
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    const calculatedAge = calculateAge(patientData.dob);

    try {
      const pythonFormData = new FormData();
      pythonFormData.append('file', audioBlob, 'recording.webm');
      pythonFormData.append('age', calculatedAge);
      pythonFormData.append('symptoms', patientData.symptoms || "None reported");
      pythonFormData.append('threshold', systemConfig.confidence_threshold);

      // 🛑 DEBUG: FORCE DIRECT CONNECTION (Bypass Vercel 10s Timeout)
      console.log("🚀 SENDING DIRECTLY TO RENDER: https://sauti-ya-afya-1.onrender.com/analyze");
      
      // ✅ DIRECT URL - Allows long timeouts for Cold Starts
      const aiRes = await axios.post('https://sauti-ya-afya-1.onrender.com/analyze', pythonFormData);
      
      console.log("✅ RESPONSE RECEIVED:", aiRes.data);
      const aiResult = aiRes.data;
      setAnalysis(aiResult);
      setSaveStatus('saving');

      const token = await auth.currentUser.getIdToken();
      
      const nodeFormData = new FormData();
      nodeFormData.append('file', audioBlob, 'recording.webm');
      nodeFormData.append('name', patientData.name);
      nodeFormData.append('age', calculatedAge);
      nodeFormData.append('dob', patientData.dob);
      nodeFormData.append('location', patientData.location);
      nodeFormData.append('phone', patientData.phone);
      nodeFormData.append('symptoms', patientData.symptoms || "");
      nodeFormData.append('diagnosis', aiResult.preliminary_assessment);
      nodeFormData.append('risk_level', aiResult.risk_level_output);
      nodeFormData.append('biomarkers', JSON.stringify(aiResult.biomarkers));
      
      // ✅ USE THE REAL IMAGE FROM SERVER
      // If server sends nothing (rare), fallback to empty string (Node backend handles empty strings fine, just not corrupted data)
      nodeFormData.append('spectrogram', aiResult.visualizer?.spectrogram_image || "");

      await axios.post(`${config.API_BASE_URL}/patients`, nodeFormData, {
        headers: { 
            Authorization: `Bearer ${token}`,
            'Content-Type': 'multipart/form-data'
        }
      });
      setSaveStatus('saved');

    } catch (err) {
      console.error("❌ UPLOAD ERROR:", err);
      // 🛑 SHOW ERROR TO USER
      alert(`Analysis Failed: ${err.message}`);
      setSaveStatus('error');
    } finally {
      setLoading(false);
    }
  };

  const getRiskColor = (risk) => {
    if (risk === "High") return "text-danger";
    if (risk === "Medium") return "text-warning";
    return "text-success";
  };

  return (
    <div className="container-fluid p-0 d-flex justify-content-center">
      <div className="glass-card w-100" style={{ maxWidth: '1100px', minHeight: '85vh' }}>
        <div className="p-3 p-md-5">
          
          {/* Header */}
          <div className="d-flex align-items-center justify-content-between mb-5">
              <div className="d-flex align-items-center">
                  <div className="bg-white rounded-circle p-3 text-accent me-3 shadow-sm">
                      <FaNotesMedical size={24} />
                  </div>
                  <div>
                      <h4 className="fw-bold text-dark-brown mb-0">New Screening</h4>
                      <small className="text-muted d-block">Phase 6: Cloud Deployment</small>
                  </div>
              </div>
              {onLogout && (
                <button onClick={onLogout} className="btn btn-outline-danger d-flex align-items-center rounded-pill px-4 shadow-sm">
                  <FaSignOutAlt className="me-2"/> Logout
                </button>
              )}
          </div>
          
          {/* Input Form - Desktop Grid */}
          <div className="row g-4 text-start">
              <div className="col-12 col-md-6">
                  <label className="form-label small fw-bold text-muted text-uppercase">Patient Name</label>
                  <input 
                      name="name" 
                      className={`form-control ${errors.name ? 'is-invalid' : ''}`} 
                      style={glassInputStyle} 
                      onChange={handleInputChange} 
                      placeholder="Enter full name"
                  />
                  {errors.name && <div className="invalid-feedback">{errors.name}</div>}
              </div>

              <div className="col-12 col-md-6">
                  <label className="form-label small fw-bold text-muted text-uppercase">Date of Birth</label>
                  <div className="input-group">
                      <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.4)'}}><FaCalendarAlt className="text-muted"/></span>
                      <input 
                          name="dob" 
                          type="date" 
                          className={`form-control ${errors.dob ? 'is-invalid' : ''}`} 
                          style={glassInputStyle} 
                          onChange={handleInputChange} 
                          max={new Date().toISOString().split("T")[0]} 
                      />
                      {errors.dob && <div className="invalid-feedback d-block">{errors.dob}</div>}
                  </div>
              </div>

              <div className="col-12 col-md-6">
                  <label className="form-label small fw-bold text-muted text-uppercase">Phone Number (KE)</label>
                  <div className="input-group">
                      <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.4)'}}><FaPhoneAlt className="text-muted"/></span>
                      <input 
                          name="phone" 
                          type="tel"
                          className={`form-control ${errors.phone ? 'is-invalid' : ''}`} 
                          style={glassInputStyle} 
                          onChange={handleInputChange} 
                          placeholder="e.g. 0712 345 678" 
                      />
                      {errors.phone && <div className="invalid-feedback d-block">{errors.phone}</div>}
                  </div>
              </div>

              <div className="col-12 col-md-6">
                  <label className="form-label small fw-bold text-muted text-uppercase">Location</label>
                  <select 
                      name="location" 
                      className={`form-select ${errors.location ? 'is-invalid' : ''}`} 
                      style={glassInputStyle} 
                      onChange={handleInputChange} 
                      value={patientData.location}
                  >
                      {/* 🛑 FIX 2: Force color on Options to prevent invisible text */}
                      <option value="" style={{color: 'black'}}>-- Select County --</option>
                      {counties.map((c) => (
                        <option key={c.id} value={c.name} style={{color: 'black'}}>
                            {/* 🛑 FIX 3: Safe Access to code */}
                            {(c.code || '000').toString().padStart(3,'0')} - {c.name}
                        </option>
                      ))}
                  </select>
                  {errors.location && <div className="invalid-feedback">{errors.location}</div>}
              </div>

              <div className="col-12">
                  <label className="form-label small fw-bold text-muted text-uppercase">Observed Symptoms</label>
                  <textarea 
                      name="symptoms" 
                      className="form-control" 
                      style={glassInputStyle} 
                      onChange={handleInputChange} 
                      rows="3" 
                      placeholder="Coughing, wheezing, fever..."
                  ></textarea>
              </div>
          </div>
          
          <hr className="my-5" style={{borderColor: 'rgba(255,255,255,0.3)'}} />

          {/* Recorder UI */}
          <div className="text-center">
              <h5 className="fw-bold mb-4 text-dark-brown">Lung Sound Capture</h5>
              <div className="glass-card d-inline-block p-5 mb-4 shadow-sm" style={{background: 'rgba(255,255,255,0.4)'}}>
                  <LungAnimation isRecording={isRecording} volume={volume} />
              </div>
              <div className="d-grid gap-2 col-12 col-md-6 mx-auto">
              {!isRecording ? (
                  <button className="btn btn-outline-danger btn-lg rounded-pill shadow-sm py-3" style={{borderWidth: '2px'}} onClick={startRecording}>
                  <FaMicrophone className="me-2"/> Start Recording
                  </button>
              ) : (
                  <button className="btn btn-danger btn-lg rounded-pill shadow-sm animate-pulse py-3" onClick={stopRecording}>
                  <FaStop className="me-2"/> Stop Recording
                  </button>
              )}
              </div>
          </div>

          {/* RESULTS AREA */}
          {loading && <div className="mt-5 text-center text-dark-brown animate-pulse">🧠 Running Quantized Neural Network...</div>}
          
          {analysis && (
            <div className={`glass-card mt-5 text-start border-start border-5 animate-slide-in shadow-lg ${analysis.risk_level_output === 'High' ? 'border-danger' : analysis.risk_level_output === 'Medium' ? 'border-warning' : 'border-success'}`}>
              
              <div className="d-flex justify-content-between align-items-center mb-4 p-4 border-bottom">
                  <h4 className="mb-0 fw-bold text-dark-brown"><FaStethoscope className="me-2"/>Diagnostic Result</h4>
                  {saveStatus === 'saved' && <span className="badge bg-success shadow-sm p-2"><FaCheckCircle className="me-1"/> Saved to Database</span>}
              </div>
              
              <div className="row g-5 p-4 pt-0">
                  {/* Text Results */}
                  <div className="col-12 col-lg-6 border-end-lg">
                      <h1 className={`mb-2 fw-bold display-6 ${getRiskColor(analysis.risk_level_output)}`}>
                          {analysis.preliminary_assessment}
                      </h1>
                      <p className="text-muted mb-4 fs-5">Risk Level: <strong>{analysis.risk_level_output.toUpperCase()}</strong></p>

                      <div className="p-4 rounded-3 shadow-sm mb-3" style={{background: 'rgba(255,255,255,0.7)'}}>
                          <label className="small fw-bold text-muted text-uppercase mb-3 d-block">AI Confidence Scores</label>
                          
                          <div className="mb-3">
                              <div className="d-flex justify-content-between mb-1">
                                  <span>Pneumonia Probability</span>
                                  <span className="fw-bold">{((analysis.biomarkers?.prob_pneumonia || 0) * 100).toFixed(1)}%</span>
                              </div>
                              <div className="progress" style={{height: '10px'}}>
                                  <div className="progress-bar bg-danger" style={{width: `${(analysis.biomarkers?.prob_pneumonia || 0) * 100}%`}}></div>
                              </div>
                          </div>

                          <div className="mb-1">
                              <div className="d-flex justify-content-between mb-1">
                                  <span>Asthma Probability</span>
                                  <span className="fw-bold">{((analysis.biomarkers?.prob_asthma || 0) * 100).toFixed(1)}%</span>
                              </div>
                              <div className="progress" style={{height: '10px'}}>
                                  <div className="progress-bar bg-warning" style={{width: `${(analysis.biomarkers?.prob_asthma || 0) * 100}%`}}></div>
                              </div>
                          </div>
                      </div>
                  </div>

                  {/* Visualizer */}
                  <div className="col-12 col-lg-6">
                      <div className="h-100 d-flex flex-column">
                          <label className="small fw-bold text-muted text-uppercase mb-3">Audio Spectrogram Analysis</label>
                          <div className="flex-grow-1 rounded-3 overflow-hidden border shadow-sm">
                            <AudioVisualizer 
                                spectrogramData={analysis.visualizer?.spectrogram_image} 
                                riskLevel={analysis.risk_level_output}
                            />
                          </div>
                      </div>
                  </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default SmartRecorder;