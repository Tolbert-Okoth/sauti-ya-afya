/* client/src/components/SmartRecorder.js */
import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { auth } from '../firebase'; 
import config from '../config'; 
import LungAnimation from './LungAnimation';
import { 
  FaMicrophone, FaStop, FaNotesMedical, FaCheckCircle, 
  FaCalendarAlt, FaPhoneAlt, FaSignOutAlt,
  FaServer, FaCircle, FaInfoCircle, FaChartBar, FaStethoscope
} from 'react-icons/fa';

// Chest spots recorded per child (back of the chest, as in the SPRSound training data)
const CHEST_SPOTS = ['Left upper back', 'Right upper back', 'Left lower back', 'Right lower back'];
const RECORD_SECONDS = 15;
const MIN_RECORD_SECONDS = 5;
// IMCI signs; the phrases are what the engine's symptom parser recognises
const IMCI_SIGNS = [
  { key: 'unable to drink or breastfeed', label: 'Unable to drink / breastfeed' },
  { key: 'vomits everything', label: 'Vomits everything' },
  { key: 'convulsions', label: 'Convulsions' },
  { key: 'lethargic', label: 'Lethargic or unconscious' },
  { key: 'chest indrawing', label: 'Chest indrawing' },
  { key: 'stridor', label: 'Stridor in a calm child' },
  { key: 'blue lips', label: 'Blue lips / tongue' },
];
// Browser voice processing (noise suppression, echo cancellation, auto gain) removes breath sounds and crackles
const RAW_AUDIO = { audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } };

const SmartRecorder = ({ onLogout }) => {
  
  // Recorder State
  const [isRecording, setIsRecording] = useState(false);
  const [volume, setVolume] = useState(0);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState(''); 
  
  // Server Status State ('sleeping' | 'waking' | 'ready')
  const [serverStatus, setServerStatus] = useState('sleeping');

  // Admin Config & Counties
  // Admin config is still loaded; per-child analysis uses the engine's validated thresholds instead of the slider
  const [, setSystemConfig] = useState({ confidence_threshold: 0.75 });
  const [counties, setCounties] = useState([]); 

  // Patient Form State
  const [patientData, setPatientData] = useState({
    name: '',
    dob: '', 
    location: '', 
    phone: '', 
    symptoms: '' 
  });

  // 👶 Several chest spots per child + IMCI signs
  const [spotBlobs, setSpotBlobs] = useState(Array(CHEST_SPOTS.length).fill(null));
  const [activeSpot, setActiveSpot] = useState(null);
  const [secondsLeft, setSecondsLeft] = useState(0);
  const [breathsPerMinute, setBreathsPerMinute] = useState('');
  const [dangerSigns, setDangerSigns] = useState([]);
  const stopTimerRef = useRef(null);

  const [errors, setErrors] = useState({});
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const animationRef = useRef(null);

  // 🔹 NEW DARK GLASS INPUT STYLE
  const glassInputStyle = {
      background: 'rgba(0, 0, 0, 0.2)', 
      border: '1px solid rgba(255, 255, 255, 0.2)',
      color: '#fff',
      borderRadius: '12px',
      padding: '11px',
      backdropFilter: 'blur(6px)'
  };

  useEffect(() => {
    const wakeUpServer = async () => {
        try {
            setServerStatus('waking');
            const minWait = new Promise(resolve => setTimeout(resolve, 1500));
            const ping = axios.get('https://sauti-ya-afya-1.onrender.com/');
            await Promise.all([ping, minWait]);
            setServerStatus('ready');
        } catch (e) {
            console.warn("Server waking up...", e);
            setServerStatus('ready'); 
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

  const getClinicalExplanation = (result) => {
      if (!result || !result.biomarkers) return "No data available.";
      if (result.explanation) return result.explanation;
      
      const bio = result.biomarkers;
      const diag = result.preliminary_assessment.replace(' Pattern', '');
      const p_pneumonia = (bio.prob_pneumonia * 100).toFixed(0);
      const p_asthma = (bio.prob_asthma * 100).toFixed(0);
      const p_normal = (bio.prob_normal * 100).toFixed(0);

      if (diag === "Pneumonia") {
          return `The AI detected acoustic signatures consistent with lung consolidation or crackles (${p_pneumonia}% confidence). These patterns strongly outweigh the characteristics of normal breath sounds (${p_normal}%) or wheezing.`;
      } else if (diag === "Asthma") {
          return `The analysis identified high-frequency continuous sounds typical of wheezing (${p_asthma}% confidence). This pattern is distinct from the clear airflow found in normal recordings.`;
      } else if (diag === "Normal") {
          return `The recording exhibits clear, unobstructed airflow patterns (${p_normal}% match). No significant adventitious sounds were detected above the risk threshold.`;
      } else {
          return "The audio pattern is inconclusive. Please ensure the recording is clear of background noise and try again.";
      }
  };

  const calculateAgeMonths = (dob) => {
      const birth = new Date(dob);
      const now = new Date();
      return Math.max(0, (now.getFullYear() - birth.getFullYear()) * 12 + (now.getMonth() - birth.getMonth())
          - (now.getDate() < birth.getDate() ? 1 : 0));
  };

  const startRecording = async (spotIndex) => {
    if (!validateForm()) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia(RAW_AUDIO);
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
      const startedAt = Date.now();
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };
      mediaRecorderRef.current.onstop = () => {
        audioContext.close();
        if ((Date.now() - startedAt) / 1000 < MIN_RECORD_SECONDS) {
          alert(`Recording too short. Keep the phone still on the chest for at least ${MIN_RECORD_SECONDS} seconds.`);
          return;
        }
        const blob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        setSpotBlobs(prev => prev.map((b, i) => (i === spotIndex ? blob : b)));
      };
      mediaRecorderRef.current.start();
      setIsRecording(true);
      setActiveSpot(spotIndex);
      setAnalysis(null);
      setSaveStatus('');
      updateVolume();

      setSecondsLeft(RECORD_SECONDS);
      stopTimerRef.current = setInterval(() => {
        setSecondsLeft(prev => {
          if (prev <= 1) { stopRecording(); return 0; }
          return prev - 1;
        });
      }, 1000);
    } catch (err) {
      console.error(err);
      alert("Microphone access denied. Please allow permissions.");
    }
  };

  const stopRecording = () => {
    clearInterval(stopTimerRef.current);
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
    }
    setIsRecording(false);
    setActiveSpot(null);
    setVolume(0);
    cancelAnimationFrame(animationRef.current);
  };

  useEffect(() => () => clearInterval(stopTimerRef.current), []);

  const toggleDangerSign = (key) => {
    setDangerSigns(prev => (prev.includes(key) ? prev.filter(k => k !== key) : [...prev, key]));
  };

  const recordedSpots = spotBlobs.filter(Boolean).length;

  const analyseChild = async () => {
    if (!validateForm()) return;
    if (recordedSpots === 0) { alert("Record at least one chest spot first."); return; }
    if (recordedSpots < 2 && !window.confirm("Only one spot recorded. At least 2 spots give a more reliable result. Analyse anyway?")) return;
    const bpm = breathsPerMinute === '' ? null : Number(breathsPerMinute);
    if (bpm !== null && (!Number.isFinite(bpm) || bpm < 5 || bpm > 150)) {
      alert("Breaths per minute should be between 5 and 150 (count for a full 60 seconds)."); return;
    }

    setLoading(true);
    setAnalysis(null);
    setSaveStatus('');
    const calculatedAge = calculateAge(patientData.dob);
    // Typed symptoms + ticked IMCI signs, in phrases the engine's parser understands
    const fullSymptoms = [patientData.symptoms, ...dangerSigns].filter(Boolean).join(', ');

    try {
      const pythonFormData = new FormData();
      spotBlobs.forEach((blob, i) => { if (blob) pythonFormData.append('files', blob, `spot${i + 1}.webm`); });
      pythonFormData.append('symptoms', fullSymptoms);
      pythonFormData.append('age_months', calculateAgeMonths(patientData.dob));
      if (bpm !== null) pythonFormData.append('breaths_per_minute', bpm);

      const submitRes = await axios.post('https://sauti-ya-afya-1.onrender.com/analyze_child', pythonFormData);
      const jobId = submitRes.data.job_id;

      let aiResult = null;
      let attempts = 0;
      const maxAttempts = 90;

      while (attempts < maxAttempts) {
          attempts++;
          await new Promise(resolve => setTimeout(resolve, 2000));
          const statusRes = await axios.get(`https://sauti-ya-afya-1.onrender.com/status/${jobId}`);

          if (statusRes.data.status === 'completed') {
              aiResult = statusRes.data.result;
              break;
          } else if (statusRes.data.status === 'failed') {
              throw new Error(statusRes.data.error || "AI Analysis Failed");
          }
      }

      if (!aiResult) throw new Error("Server timed out processing the request.");

      setAnalysis(aiResult);
      setSaveStatus('saving');

      const token = await auth.currentUser.getIdToken();
      const nodeFormData = new FormData();
      nodeFormData.append('file', spotBlobs.find(Boolean), 'recording.webm');   // first recorded spot is archived
      nodeFormData.append('name', patientData.name);
      nodeFormData.append('age', calculatedAge);
      nodeFormData.append('dob', patientData.dob);
      nodeFormData.append('location', patientData.location);
      nodeFormData.append('phone', patientData.phone);
      nodeFormData.append('symptoms', fullSymptoms);
      nodeFormData.append('diagnosis', aiResult.preliminary_assessment);
      nodeFormData.append('risk_level', aiResult.risk_level_output);
      nodeFormData.append('biomarkers', JSON.stringify(aiResult.biomarkers));
      nodeFormData.append('spectrogram', "");

      await axios.post(`${config.API_BASE_URL}/patients`, nodeFormData, {
        headers: {
            Authorization: `Bearer ${token}`,
            'Content-Type': 'multipart/form-data'
        }
      });
      setSaveStatus('saved');

    } catch (err) {
      console.error("Analysis Error:", err);
      alert(`Error: ${err.message}`);
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
      
      {/* 🟢 DARK MODE GHOST CARD */}
      <div 
        className="glass-card w-100 shadow-lg" 
        style={{ 
            maxWidth: '1100px', 
            minHeight: '85vh', 
            backdropFilter: 'none', 
            background: 'rgba(255, 255, 255, 0.05)', 
            border: '1px solid rgba(255, 255, 255, 0.1)'
        }}
      >
        <div className="p-3 p-md-5">
          
          {/* Header */}
          <div className="d-flex align-items-center justify-content-between mb-5">
              <div className="d-flex align-items-center">
                  <div className="bg-primary rounded-circle p-3 text-white me-3 shadow-sm">
                      <FaNotesMedical size={24} />
                  </div>
                  <div>
                      <h4 className="fw-bold text-white mb-0">New Screening</h4>
                      <div className="d-flex align-items-center mt-1">
                        <small className="text-muted me-2">AI Engine Status:</small>
                        {serverStatus === 'waking' && (
                            <span className="badge bg-warning text-dark animate-pulse d-flex align-items-center">
                                <FaCircle size={8} className="me-1"/> Waking Up...
                            </span>
                        )}
                        {serverStatus === 'ready' && (
                            <span className="badge bg-success d-flex align-items-center">
                                <FaCheckCircle size={8} className="me-1"/> Ready
                            </span>
                        )}
                        {serverStatus === 'sleeping' && (
                            <span className="badge bg-secondary d-flex align-items-center">
                                <FaServer size={8} className="me-1"/> Connecting...
                            </span>
                        )}
                      </div>
                  </div>
              </div>
              {onLogout && (
                <button onClick={onLogout} className="btn btn-outline-danger d-flex align-items-center rounded-pill px-4 shadow-sm text-white border-light">
                  <FaSignOutAlt className="me-2"/> Logout
                </button>
              )}
          </div>
          
          {/* Input Form */}
          <div className="row g-4 text-start">
              <div className="col-12 col-md-6">
                  <label className="form-label small fw-bold text-muted text-uppercase">Patient Name</label>
                  <input name="name" className={`form-control ${errors.name ? 'is-invalid' : ''}`} style={glassInputStyle} onChange={handleInputChange} placeholder="Enter full name"/>
                  {errors.name && <div className="invalid-feedback">{errors.name}</div>}
              </div>
              <div className="col-12 col-md-6">
                  <label className="form-label small fw-bold text-muted text-uppercase">Date of Birth</label>
                  <div className="input-group">
                      <span className="input-group-text border-0" style={{...glassInputStyle, borderRight: 'none', borderTopRightRadius: 0, borderBottomRightRadius: 0}}><FaCalendarAlt className="text-muted"/></span>
                      <input name="dob" type="date" className={`form-control ${errors.dob ? 'is-invalid' : ''}`} style={{...glassInputStyle, borderLeft: 'none', borderTopLeftRadius: 0, borderBottomLeftRadius: 0}} onChange={handleInputChange} max={new Date().toISOString().split("T")[0]} />
                      {errors.dob && <div className="invalid-feedback d-block">{errors.dob}</div>}
                  </div>
              </div>
              <div className="col-12 col-md-6">
                  <label className="form-label small fw-bold text-muted text-uppercase">Phone Number (KE)</label>
                  <div className="input-group">
                      <span className="input-group-text border-0" style={{...glassInputStyle, borderRight: 'none', borderTopRightRadius: 0, borderBottomRightRadius: 0}}><FaPhoneAlt className="text-muted"/></span>
                      <input name="phone" type="tel" className={`form-control ${errors.phone ? 'is-invalid' : ''}`} style={{...glassInputStyle, borderLeft: 'none', borderTopLeftRadius: 0, borderBottomLeftRadius: 0}} onChange={handleInputChange} placeholder="e.g. 0712 345 678" />
                      {errors.phone && <div className="invalid-feedback d-block">{errors.phone}</div>}
                  </div>
              </div>
              <div className="col-12 col-md-6">
                  <label className="form-label small fw-bold text-muted text-uppercase">Location</label>
                  <select name="location" className={`form-select ${errors.location ? 'is-invalid' : ''}`} style={glassInputStyle} onChange={handleInputChange} value={patientData.location}>
                      <option value="" style={{color: '#333'}}>-- Select County --</option>
                      {counties.map((c) => (<option key={c.id} value={c.name} style={{color: '#333'}}>{(c.code || '000').toString().padStart(3,'0')} - {c.name}</option>))}
                  </select>
                  {errors.location && <div className="invalid-feedback">{errors.location}</div>}
              </div>
              <div className="col-12">
                  <label className="form-label small fw-bold text-muted text-uppercase">Observed Symptoms</label>
                  <textarea name="symptoms" className="form-control" style={glassInputStyle} onChange={handleInputChange} rows="3" placeholder="Coughing, wheezing, fever..."></textarea>
              </div>
          </div>
          
          <hr className="my-5" style={{borderColor: 'rgba(255,255,255,0.1)'}} />

          {/* IMCI checks */}
          <div className="row g-4 text-start mb-5">
              <div className="col-12 col-md-4">
                  <label className="form-label small fw-bold text-muted text-uppercase">Breaths per minute</label>
                  <input type="number" min="5" max="150" className="form-control" style={glassInputStyle}
                         value={breathsPerMinute} onChange={(e) => setBreathsPerMinute(e.target.value)}
                         placeholder="Count for a full 60 seconds" />
                  <small className="text-muted">Child calm, chest uncovered. Needed for the IMCI fast-breathing check.</small>
              </div>
              <div className="col-12 col-md-8">
                  <label className="form-label small fw-bold text-muted text-uppercase">IMCI signs present</label>
                  <div className="d-flex flex-wrap gap-2">
                      {IMCI_SIGNS.map(sign => (
                          <button key={sign.key} type="button" onClick={() => toggleDangerSign(sign.key)}
                                  className={`btn btn-sm rounded-pill ${dangerSigns.includes(sign.key) ? 'btn-danger' : 'btn-outline-light'}`}>
                              {sign.label}
                          </button>
                      ))}
                  </div>
              </div>
          </div>

          {/* Recorder UI: several chest spots per child */}
          <div className="text-center">
              <h5 className="fw-bold mb-2 text-white">Lung Sound Capture</h5>
              <p className="text-muted mb-4">
                  Press the phone microphone firmly on the bare skin, keep it still and the room quiet.
                  Record at least 2 spots ({RECORD_SECONDS} s each, stops automatically).
              </p>
              <div className="glass-card d-inline-block p-5 mb-4 shadow-sm" style={{background: 'rgba(255,255,255,0.05)'}}>
                  <LungAnimation isRecording={isRecording} volume={volume} />
                  {isRecording && <div className="text-white mt-3 fw-bold">{CHEST_SPOTS[activeSpot]}: {secondsLeft}s</div>}
              </div>
              <div className="row g-3 col-12 col-md-10 mx-auto mb-4">
                  {CHEST_SPOTS.map((spot, i) => (
                      <div key={spot} className="col-6 col-md-3 d-grid">
                          {activeSpot === i ? (
                              <button className="btn btn-danger rounded-pill animate-pulse py-3" onClick={stopRecording}>
                                  <FaStop className="me-2"/> Stop
                              </button>
                          ) : (
                              <button className={`btn rounded-pill py-3 ${spotBlobs[i] ? 'btn-success' : 'btn-outline-danger text-white border-danger'}`}
                                      onClick={() => startRecording(i)} disabled={isRecording || loading || serverStatus !== 'ready'}>
                                  {spotBlobs[i] ? <FaCheckCircle className="me-2"/> : <FaMicrophone className="me-2"/>}
                                  {spot}{spotBlobs[i] ? ' (redo)' : ''}
                              </button>
                          )}
                      </div>
                  ))}
              </div>
              <div className="d-grid gap-2 col-12 col-md-6 mx-auto">
                  <button className="btn btn-primary btn-lg rounded-pill shadow-sm py-3" onClick={analyseChild}
                          disabled={isRecording || loading || recordedSpots === 0 || serverStatus !== 'ready'}>
                      {serverStatus !== 'ready' ? '⏳ Waiting for AI...' : <><FaStethoscope className="me-2"/> Analyse child ({recordedSpots} spot{recordedSpots === 1 ? '' : 's'})</>}
                  </button>
              </div>
          </div>

          {/* RESULTS AREA */}
          {loading && <div className="mt-5 text-center text-white animate-pulse">🧠 Running Async Analysis (Please Wait)...</div>}
          
          {analysis && (
            <div className={`glass-card mt-5 text-start border-start border-5 animate-slide-in shadow-lg ${analysis.risk_level_output === 'High' ? 'border-danger' : analysis.risk_level_output === 'Medium' ? 'border-warning' : 'border-success'}`}>
              
              <div className="d-flex justify-content-between align-items-center mb-4 p-4 border-bottom" style={{borderColor: 'rgba(255,255,255,0.1)'}}>
                  <h4 className="mb-0 fw-bold text-white"><FaStethoscope className="me-2"/>Diagnostic Result</h4>
                  {saveStatus === 'saved' && <span className="badge bg-success shadow-sm p-2"><FaCheckCircle className="me-1"/> Saved to Database</span>}
              </div>
              
              <div className="row g-5 p-4 pt-0">
                  <div className="col-12 col-lg-6 border-end-lg" style={{borderColor: 'rgba(255,255,255,0.1)'}}>
                      <h1 className={`mb-2 fw-bold display-6 ${getRiskColor(analysis.risk_level_output)}`}>
                          {analysis.preliminary_assessment}
                      </h1>
                      {analysis.biomarkers?.sound_finding && (
                          <p className="text-white mb-1">Heard: <strong>{analysis.biomarkers.sound_finding}</strong></p>
                      )}
                      <p className="text-muted mb-4 fs-5">Risk Level: <strong>{analysis.risk_level_output.toUpperCase()}</strong></p>

                      <div className="p-4 rounded-3 shadow-sm mb-3" style={{background: 'rgba(0,0,0,0.2)'}}>
                          <label className="small fw-bold text-muted text-uppercase mb-3 d-block"><FaChartBar className="me-2"/>AI Sound Scores (best spot)</label>
                          
                          <div className="mb-3">
                              <div className="d-flex justify-content-between mb-1 text-white">
                                  <span>Crackles (pneumonia sign)</span>
                                  <span className="fw-bold">{((analysis.biomarkers?.prob_pneumonia || 0) * 100).toFixed(1)}%</span>
                              </div>
                              <div className="progress" style={{height: '10px', background: 'rgba(255,255,255,0.1)'}}>
                                  <div className="progress-bar bg-danger" style={{width: `${(analysis.biomarkers?.prob_pneumonia || 0) * 100}%`}}></div>
                              </div>
                          </div>
                          
                          <div className="mb-3">
                              <div className="d-flex justify-content-between mb-1 text-white">
                                  <span>Wheeze (asthma sign)</span>
                                  <span className="fw-bold">{((analysis.biomarkers?.prob_asthma || 0) * 100).toFixed(1)}%</span>
                              </div>
                              <div className="progress" style={{height: '10px', background: 'rgba(255,255,255,0.1)'}}>
                                  <div className="progress-bar bg-warning" style={{width: `${(analysis.biomarkers?.prob_asthma || 0) * 100}%`}}></div>
                              </div>
                          </div>

                          <div className="mb-1">
                              <div className="d-flex justify-content-between mb-1 text-white">
                                  <span>Normal / Healthy</span>
                                  <span className="fw-bold">{((analysis.biomarkers?.prob_normal || 0) * 100).toFixed(1)}%</span>
                              </div>
                              <div className="progress" style={{height: '10px', background: 'rgba(255,255,255,0.1)'}}>
                                  <div className="progress-bar bg-success" style={{width: `${(analysis.biomarkers?.prob_normal || 0) * 100}%`}}></div>
                              </div>
                          </div>
                      </div>
                  </div>

                  <div className="col-12 col-lg-6">
                      <div className="h-100 d-flex flex-column">
                          <label className="small fw-bold text-muted text-uppercase mb-3"><FaInfoCircle className="me-2"/>Clinical Interpretation</label>
                          
                          {/* 🟢 Dark Blue Glass Clinical Box */}
                          <div className="flex-grow-1 rounded-3 p-4 border border-info shadow-sm" style={{backgroundColor: 'rgba(9, 132, 227, 0.15)'}}>
                              <h5 className="fw-bold text-white mb-3">Analysis Summary</h5>
                              <p className="text-white mb-0" style={{lineHeight: '1.6', opacity: 0.9}}>
                                  {getClinicalExplanation(analysis)}
                              </p>
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