/* client/src/components/DoctorDashboard.js */
import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { auth } from '../firebase';
import config from '../config'; 
import OutbreakMap from './OutbreakMap';
import AudioVisualizer from './AudioVisualizer'; 
import { 
  FaWhatsapp, FaCheck, FaSync, FaUserMd, FaCircle, 
  FaSpinner, FaExclamationTriangle, FaNotesMedical, FaChartBar, FaInfoCircle
} from 'react-icons/fa';

const DoctorDashboard = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [isUpdating, setIsUpdating] = useState(false);

  // 1. FETCH DATA (With Cache Busting)
  const fetchPatients = useCallback(async (isBackground = false) => {
    try {
      if (isBackground) setIsUpdating(true);
      
      const token = await auth.currentUser.getIdToken();
      const res = await axios.get(`${config.API_BASE_URL}/patients?t=${Date.now()}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      // AUTO-SORT: High Risk first
      const riskOrder = { 'High': 3, 'Medium': 2, 'Low': 1 };
      const sortedData = res.data.sort((a, b) => {
          return (riskOrder[b.risk_level] || 0) - (riskOrder[a.risk_level] || 0);
      });

      setPatients(sortedData);
    } catch (err) {
      console.error("Error fetching patients:", err);
    } finally {
      if (!isBackground) setLoading(false); 
      if (isBackground) setTimeout(() => setIsUpdating(false), 800); 
    }
  }, []);

  // 2. POLLING LOOP
  useEffect(() => {
    fetchPatients(false);
    const intervalId = setInterval(() => { fetchPatients(true); }, 5000);
    return () => clearInterval(intervalId);
  }, [fetchPatients]);

  // 3. RESOLVE CASE
  const resolveCase = async (id) => {
    if(window.confirm("Mark case as resolved and remove from triage list?")) {
        try {
            const token = await auth.currentUser.getIdToken();
            await axios.delete(`${config.API_BASE_URL}/patients/${id}`, {
                headers: { Authorization: `Bearer ${token}` }
            });
            setPatients(prev => prev.filter(p => p.id !== id));
            setSelectedPatient(null);
        } catch (err) {
            alert("Failed to resolve case.");
        }
    }
  };

  if (loading) return <div className="text-center mt-5 text-dark-brown animate-pulse">Loading Triage Data...</div>;

  return (
    <div className="container-fluid p-0">
        {/* Header */}
        <header className="d-flex justify-content-between align-items-center mb-4">
          <div>
            <h4 className="fw-bold text-dark-brown mb-0">Epidemiology Surveillance</h4>
            <p className="text-dark-brown opacity-75 small">Live data from 47 Counties</p>
          </div>
          
          <div className="glass-card px-4 py-2 d-flex align-items-center transition-all">
             <div className={`d-flex align-items-center fw-bold small ${isUpdating ? 'text-accent' : 'text-success'}`}>
                {isUpdating ? (
                    <><FaSync size={12} className="me-2 fa-spin" /> Syncing...</>
                ) : (
                    <><FaCircle size={10} className="me-2 animate-pulse" /> System Online</>
                )}
             </div>
          </div>
        </header>

        <div className="row g-4">
            {/* LEFT COLUMN: Map & Stats */}
            <div className="col-lg-7 order-lg-2">
                <div className="glass-card p-2" style={{height: '400px', overflow:'hidden'}}>
                    <OutbreakMap patients={patients} />
                </div>
                
                <div className="row g-3 mt-1">
                    <div className="col-4">
                        <div className="glass-card text-center py-3">
                            <h2 className="fw-bold text-danger mb-0">{patients.filter(p => p.risk_level === 'High').length}</h2>
                            <small className="text-dark-brown opacity-75">Critical</small>
                        </div>
                    </div>
                    <div className="col-4">
                        <div className="glass-card text-center py-3">
                            <h2 className="fw-bold text-warning mb-0">{patients.filter(p => p.risk_level === 'Medium').length}</h2>
                            <small className="text-dark-brown opacity-75">Monitor</small>
                        </div>
                    </div>
                    <div className="col-4">
                        <div className="glass-card text-center py-3">
                            <h2 className="fw-bold text-accent mb-0">{patients.filter(p => p.risk_level === 'Low').length}</h2>
                            <small className="text-dark-brown opacity-75">Stable</small>
                        </div>
                    </div>
                </div>
            </div>

            {/* RIGHT COLUMN: Patient Queue */}
            <div className="col-lg-5 order-lg-1">
                <div className="d-flex justify-content-between align-items-center mb-3">
                    <h5 className="fw-bold text-dark-brown mb-0">Triage Queue</h5>
                    <span className="badge bg-white text-dark shadow-sm">{patients.length} Pending</span>
                </div>

                <div className="overflow-auto pe-2" style={{maxHeight: '600px'}}>
                {patients.map(p => (
                    <div key={p.id} className="glass-card mb-3 p-3 position-relative hover-shadow"
                        style={{ 
                            cursor: 'pointer', 
                            borderLeft: `5px solid ${p.risk_level === 'High' ? '#dc3545' : p.risk_level === 'Medium' ? '#ffc107' : '#2ecc71'}` 
                        }}
                        onClick={() => setSelectedPatient(p)}
                    >
                        <div className="d-flex align-items-center">
                            <div className="rounded-circle d-flex align-items-center justify-content-center me-3" 
                                 style={{
                                    width:'45px', height:'45px', 
                                    background: 'rgba(255,255,255,0.4)',
                                    color: p.risk_level === 'High' ? '#dc3545' : '#2d3436'
                                }}>
                                <FaUserMd size={20} />
                            </div>
                            <div className="flex-grow-1">
                                <h6 className="mb-0 fw-bold text-dark-brown">{p.name} <span className="small fw-normal text-muted">({p.age})</span></h6>
                                <div className="small text-dark-brown opacity-75">
                                    {p.location} • {new Date(p.created_at).toLocaleTimeString([], {hour:'2-digit', minute:'2-digit'})}
                                </div>
                            </div>
                            <div className="text-end">
                                <span className={`badge ${p.risk_level === 'High' ? 'bg-danger' : p.risk_level === 'Medium' ? 'bg-warning text-dark' : 'bg-success'} mb-1`}>
                                    {p.risk_level}
                                </span>
                            </div>
                        </div>
                    </div>
                ))}
                </div>
            </div>
        </div>

        {/* MODAL: Case Review */}
        {selectedPatient && (
            <CaseReviewModal 
                patient={selectedPatient} 
                onClose={() => setSelectedPatient(null)} 
                onResolve={resolveCase} 
            />
        )}
    </div>
  );
};

// --- SUB-COMPONENT: CaseReviewModal ---
const CaseReviewModal = ({ patient, onClose, onResolve }) => {
    const [audioBlobUrl, setAudioBlobUrl] = useState(null);
    const [audioLoading, setAudioLoading] = useState(false);
    const [audioError, setAudioError] = useState(null);

    // 🛠️ FETCH REAL AUDIO BLOB
    useEffect(() => {
        if (patient && patient.recording_url) {
            const fetchAudio = async () => {
                setAudioLoading(true);
                setAudioError(null);
                setAudioBlobUrl(null);

                try {
                    const fileUrl = patient.recording_url.startsWith('http') 
                        ? patient.recording_url 
                        : `${config.SERVER_URL}/uploads/${patient.recording_url}`;

                    const response = await fetch(fileUrl);
                    if (!response.ok) throw new Error("File missing");
                    const blob = await response.blob();
                    const objectUrl = URL.createObjectURL(blob);
                    setAudioBlobUrl(objectUrl);
                } catch (err) {
                    console.error("Audio Load Error:", err);
                    setAudioError("Could not load audio file.");
                } finally {
                    setAudioLoading(false);
                }
            };
            fetchAudio();
        }
    }, [patient]);

    // 🧠 CLINICAL INTERPRETATION LOGIC
    const getClinicalExplanation = () => {
        const bio = patient.biomarkers || {};
        const diag = patient.diagnosis ? patient.diagnosis.replace(' Pattern', '') : 'Unknown';
        const p_pneumonia = ((bio.prob_pneumonia || 0) * 100).toFixed(0);
        const p_asthma = ((bio.prob_asthma || 0) * 100).toFixed(0);
        const p_normal = ((bio.prob_normal || 0) * 100).toFixed(0);

        if (diag === "Pneumonia") {
            return `AI detected acoustic signatures consistent with lung consolidation or crackles (${p_pneumonia}%). Overweighs normal breath sounds (${p_normal}%).`;
        } else if (diag === "Asthma") {
            return `Analysis identified high-frequency continuous sounds typical of wheezing (${p_asthma}%). Distinct from clear airflow.`;
        } else if (diag === "Normal") {
            return `Clear, unobstructed airflow patterns detected (${p_normal}%). No significant adventitious sounds found.`;
        } else {
            return "Audio pattern is inconclusive. Clinical correlation recommended.";
        }
    };

    return (
        <div className="modal show d-block" style={{ backgroundColor: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(5px)' }}>
            <div className="modal-dialog modal-xl modal-dialog-centered">
                <div className="modal-content glass-card border-0 overflow-hidden" style={{background: 'rgba(255,255,255,0.95)'}}>
                    <div className="modal-header border-bottom border-light">
                        <h5 className="modal-title fw-bold text-dark-brown"><FaNotesMedical className="me-2"/>Clinical Case Review</h5>
                        <button type="button" className="btn-close" onClick={onClose}></button>
                    </div>
                    <div className="modal-body p-4">
                        <div className="row">
                            {/* LEFT SIDE: Patient Data & Explanation */}
                            <div className="col-md-5 border-end border-light">
                                <h4 className="fw-bold text-dark-brown">{patient.name}</h4>
                                <p className="text-muted mb-4">{patient.age} • {patient.location}</p>
                                
                                <div className="mb-4">
                                   <label className="small fw-bold text-muted text-uppercase mb-2"><FaChartBar className="me-1"/> AI Confidence</label>
                                   <div className="bg-light p-3 rounded">
                                       {/* Pneumonia */}
                                       <div className="d-flex justify-content-between small mb-1">
                                           <span>Pneumonia</span>
                                           <span className="fw-bold">{((patient.biomarkers?.prob_pneumonia || 0) * 100).toFixed(1)}%</span>
                                       </div>
                                       <div className="progress mb-2" style={{height: '6px'}}>
                                           <div className="progress-bar bg-danger" style={{width: `${(patient.biomarkers?.prob_pneumonia || 0) * 100}%`}}></div>
                                       </div>

                                       {/* Asthma */}
                                       <div className="d-flex justify-content-between small mb-1">
                                           <span>Asthma</span>
                                           <span className="fw-bold">{((patient.biomarkers?.prob_asthma || 0) * 100).toFixed(1)}%</span>
                                       </div>
                                       <div className="progress mb-2" style={{height: '6px'}}>
                                           <div className="progress-bar bg-warning" style={{width: `${(patient.biomarkers?.prob_asthma || 0) * 100}%`}}></div>
                                       </div>

                                       {/* 🟢 Normal */}
                                       <div className="d-flex justify-content-between small mb-1">
                                           <span>Normal</span>
                                           <span className="fw-bold">{((patient.biomarkers?.prob_normal || 0) * 100).toFixed(1)}%</span>
                                       </div>
                                       <div className="progress" style={{height: '6px'}}>
                                           <div className="progress-bar bg-success" style={{width: `${(patient.biomarkers?.prob_normal || 0) * 100}%`}}></div>
                                       </div>
                                   </div>
                                </div>

                                <div className="mb-3">
                                   <label className="small fw-bold text-muted text-uppercase mb-2"><FaInfoCircle className="me-1"/> Clinical Interpretation</label>
                                   <div className="p-3 bg-alice-blue border border-info rounded text-dark small">
                                       {getClinicalExplanation()}
                                   </div>
                                </div>

                                <div className="mb-3">
                                    <label className="small fw-bold text-muted text-uppercase">Symptoms</label>
                                    <p className="bg-light p-2 rounded small text-dark fst-italic">"{patient.symptoms || "None"}"</p>
                                </div>
                            </div>

                            {/* RIGHT SIDE: Visualizer & Audio */}
                            <div className="col-md-7">
                                <div className="bg-dark rounded mb-3 overflow-hidden position-relative">
                                    {/* Spectrogram Visualizer */}
                                    <div style={{height: '250px'}}>
                                        {patient.spectrogram ? (
                                            <AudioVisualizer 
                                                spectrogramData={patient.spectrogram} 
                                                riskLevel={patient.risk_level}
                                            />
                                        ) : (
                                            <div className="d-flex flex-column align-items-center justify-content-center h-100 text-white-50">
                                                <FaNotesMedical size={40} className="mb-2 opacity-25"/>
                                                <span className="small">Verdict-Only Mode (Image Skipped)</span>
                                            </div>
                                        )}
                                    </div>
                                </div>
                                
                                {/* 🎵 REAL AUDIO PLAYER */}
                                <div className="mb-3">
                                    {patient.recording_url ? (
                                        <div className="text-center p-3 bg-light rounded">
                                            {audioLoading && <div className="small text-primary fw-bold"><FaSpinner className="spin me-2"/> Loading Audio...</div>}
                                            {audioError && <div className="small text-danger fw-bold"><FaExclamationTriangle/> {audioError}</div>}
                                            
                                            {audioBlobUrl && (
                                                <audio 
                                                    controls 
                                                    style={{width: '100%'}} 
                                                    src={audioBlobUrl}
                                                >
                                                    Your browser does not support the audio element.
                                                </audio>
                                            )}
                                        </div>
                                    ) : (
                                        <div className="alert alert-warning py-2 text-center small">No Recording Found</div>
                                    )}
                                </div>
                                
                                <div className="d-grid gap-2 d-md-flex justify-content-md-end">
                                    <a href={`https://wa.me/254${patient.phone?.substring(1)}`} target="_blank" rel="noreferrer" className="btn btn-success text-white">
                                        <FaWhatsapp className="me-2"/> Contact Patient
                                    </a>
                                    <button className="btn btn-primary" onClick={() => onResolve(patient.id)}>
                                        <FaCheck className="me-2"/> Resolve Case
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default DoctorDashboard;