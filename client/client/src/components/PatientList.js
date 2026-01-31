/* client/src/components/PatientList.js */
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { auth } from '../firebase';
import config from '../config'; 
import { 
  FaSearch, FaFilter, FaUserCircle, FaMapMarkerAlt, FaClock, 
  FaTimes, FaWhatsapp, FaCheck, FaSpinner, FaExclamationTriangle,
  FaStethoscope, FaNotesMedical, FaChartBar, FaArchive
} from 'react-icons/fa';

const PatientList = () => {
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterRisk, setFilterRisk] = useState('All'); 
  const [selectedPatient, setSelectedPatient] = useState(null);

  // 🎵 AUDIO STATE
  const [audioBlobUrl, setAudioBlobUrl] = useState(null);
  const [audioLoading, setAudioLoading] = useState(false);
  const [audioError, setAudioError] = useState(null);

  // 🔹 DARK GLASS INPUT STYLE
  const glassInputStyle = {
      background: 'rgba(0, 0, 0, 0.2)', 
      border: '1px solid rgba(255, 255, 255, 0.2)',
      color: '#fff',
      backdropFilter: 'blur(5px)'
  };

  // 1. FETCH PATIENTS
  useEffect(() => {
    const fetchMyPatients = async () => {
      try {
        const token = await auth.currentUser.getIdToken();
        const res = await axios.get(`${config.API_BASE_URL}/patients`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setPatients(res.data);
      } catch (err) {
        console.error("Error fetching patient list:", err);
      } finally {
        setLoading(false);
      }
    };
    fetchMyPatients();
  }, []);

  // 2. FETCH AUDIO (With Memory Cleanup)
  useEffect(() => {
    let activeObjectUrl = null;

    if (selectedPatient && selectedPatient.recording_url) {
        const fetchAudio = async () => {
            setAudioLoading(true);
            setAudioError(null);
            
            try {
                // Determine URL (Local vs Prod)
                const fileUrl = selectedPatient.recording_url.startsWith('http')
                    ? selectedPatient.recording_url
                    : `${config.SERVER_URL}/uploads/${selectedPatient.recording_url}`;

                const response = await fetch(fileUrl);
                if (!response.ok) throw new Error(`Server status: ${response.status}`);

                const blob = await response.blob();
                activeObjectUrl = URL.createObjectURL(blob);
                setAudioBlobUrl(activeObjectUrl);
            } catch (err) {
                console.error("Audio Load Error:", err);
                setAudioError("Failed to load audio file.");
            } finally {
                setAudioLoading(false);
            }
        };
        fetchAudio();
    }

    // CLEANUP: Revoke URL to free memory when patient changes or modal closes
    return () => {
        if (activeObjectUrl) URL.revokeObjectURL(activeObjectUrl);
        setAudioBlobUrl(null);
    };
  }, [selectedPatient]);

  const closeModal = () => {
      setSelectedPatient(null);
  };

  // 3. ROBUST SEARCH & FILTER (The Fix)
  const filteredPatients = patients.filter(p => {
    const term = searchTerm.toLowerCase().trim();
    // Check Name AND Location safely (prevents crashes if fields are missing)
    const matchesSearch = (p.name?.toLowerCase() || '').includes(term) || 
                          (p.location?.toLowerCase() || '').includes(term);
    const matchesRisk = filterRisk === 'All' || p.risk_level === filterRisk;
    return matchesSearch && matchesRisk;
  });

  // 4. ACTION BUTTON LOGIC
  const handleWhatsApp = () => {
      if (!selectedPatient?.caregiver_phone) {
          alert("No phone number registered for this patient.");
          return;
      }
      // Format phone: Remove non-digits, ensure country code (defaulting to Kenya +254)
      let phone = selectedPatient.caregiver_phone.replace(/\D/g, ''); 
      if (phone.startsWith('0')) phone = '254' + phone.substring(1); 

      const message = `SautiYaAfya Result: We have completed the respiratory screening for ${selectedPatient.name}. Please visit the clinic for results.`;
      window.open(`https://wa.me/${phone}?text=${encodeURIComponent(message)}`, '_blank');
  };

  const handleArchive = () => {
      if(!window.confirm("Are you sure you want to resolve and archive this case?")) return;
      
      // Optimistic UI update: Remove from list immediately
      setPatients(prev => prev.filter(p => p.id !== selectedPatient.id));
      setSelectedPatient(null);
      
      // Note: You would typically add an axios.patch call here to update the backend
  };

  const getBadge = (risk) => {
    switch(risk) {
      case 'High': return <span className="badge bg-danger shadow-sm px-3 py-2 rounded-pill">🔴 High Risk</span>;
      case 'Medium': return <span className="badge bg-warning text-dark shadow-sm px-3 py-2 rounded-pill">🟡 Moderate</span>;
      case 'Low': return <span className="badge bg-success shadow-sm px-3 py-2 rounded-pill">🟢 Stable</span>;
      default: return <span className="badge bg-secondary rounded-pill">Unknown</span>;
    }
  };

  // 5. CLINICAL NARRATIVE GENERATOR
  const getClinicalExplanation = (patient) => {
      const bio = patient.biomarkers || {};
      const diag = patient.diagnosis ? patient.diagnosis.replace(' Pattern', '') : 'Unknown';
      const p_pneu = ((bio.prob_pneumonia || 0) * 100).toFixed(0);
      const p_asthma = ((bio.prob_asthma || 0) * 100).toFixed(0);
      const p_norm = ((bio.prob_normal || 0) * 100).toFixed(0);

      if (diag === "Pneumonia") {
          return `AI detected acoustic signatures of consolidation (${p_pneu}% confidence). Physics engines flagged abnormal Spectral Flux > 1.5.`;
      } else if (diag === "Asthma") {
          return `High-frequency continuous wheezing detected (${p_asthma}% confidence), distinct from normal airflow.`;
      } else if (diag === "Normal") {
          return `Clear airflow detected (${p_norm}% confidence). ZCR and RMS levels are within healthy vesicular limits.`;
      } else {
          return "Mixed acoustic signals detected. Clinical correlation required.";
      }
  };

  if (loading) return <div className="p-5 text-center text-white animate-pulse">Loading patient records...</div>;

  return (
    <div className="container-fluid p-0">
      
      {/* Header */}
      <div className="d-flex flex-column flex-md-row justify-content-between align-items-md-center mb-4">
        <h3 className="fw-bold mb-2 mb-md-0 text-white">My Patients</h3>
        <span className="badge bg-primary text-white border border-primary shadow-sm align-self-start align-self-md-center">{filteredPatients.length} Records</span>
      </div>

      {/* Search & Filter Bar */}
      <div className="row g-3 mb-4">
        <div className="col-12 col-md-8">
          <div className="input-group shadow-sm">
            <span className="input-group-text border-0" style={glassInputStyle}><FaSearch className="text-white opacity-50"/></span>
            <input 
              type="text" 
              className="form-control border-0" 
              placeholder="Search by name or location..." 
              value={searchTerm}
              style={glassInputStyle}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
        <div className="col-12 col-md-4">
          <div className="input-group shadow-sm">
             <span className="input-group-text border-0" style={glassInputStyle}><FaFilter className="text-white opacity-50"/></span>
             <select 
                className="form-select border-0" 
                value={filterRisk} 
                style={glassInputStyle}
                onChange={(e) => setFilterRisk(e.target.value)}
             >
               <option value="All" style={{color:'black'}}>All Risks</option>
               <option value="High" style={{color:'black'}}>🔴 High Risk Only</option>
               <option value="Medium" style={{color:'black'}}>🟡 Moderate Risk</option>
               <option value="Low" style={{color:'black'}}>🟢 Stable</option>
             </select>
          </div>
        </div>
      </div>

      {/* Patient List */}
      <div className="row">
        {filteredPatients.length > 0 ? (
          filteredPatients.map((p) => (
            <div key={p.id} className="col-12 mb-3">
              {/* 🟢 Using global .glass-card class for consistency */}
              <div 
                className="glass-card border-0 p-3 transition-all hover-shadow" 
                style={{cursor: 'pointer', borderRadius: '12px'}}
                onClick={() => setSelectedPatient(p)}
              >
                <div className="d-flex align-items-center">
                  <div className="me-3">
                    <div className="bg-primary rounded-circle p-3 text-white d-flex align-items-center justify-content-center" style={{width: '50px', height: '50px'}}>
                        <FaUserCircle size={24} />
                    </div>
                  </div>
                  <div className="flex-grow-1">
                    <h5 className="mb-1 fw-bold text-white">{p.name}</h5>
                    <div className="small text-white-50 mb-1 d-flex flex-wrap align-items-center">
                        <span className="me-3">Age: <strong>{p.age}</strong></span>
                        <span><FaMapMarkerAlt className="me-1 text-info"/>{p.location}</span>
                    </div>
                    <div className="small text-white-50 opacity-75">
                          <FaClock className="me-1"/> 
                          {new Date(p.created_at).toLocaleDateString()}
                    </div>
                  </div>
                  <div className="text-end">
                    <div className="mb-2 d-none d-sm-block">{getBadge(p.risk_level)}</div>
                    <small className="d-block text-white fw-bold" style={{fontSize: '0.8rem'}}>
                      {p.diagnosis}
                    </small>
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : (
            <div className="col-12 text-center py-5 text-muted glass-card">
                <p className="mb-0 text-white">No patients found matching "{searchTerm}"</p>
            </div>
        )}
      </div>

      {/* Detail Modal */}
      {selectedPatient && (
        <div style={{
            position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
            background: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(5px)', // Darker backdrop
            zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
           <div className="glass-card rounded-4 shadow-lg overflow-hidden animate-slide-in border border-secondary" style={{width: '90%', maxWidth: '800px', maxHeight: '90vh', overflowY: 'auto', background: '#1e272e'}}>
               
               {/* Modal Header */}
               <div className="p-3 border-bottom border-secondary d-flex justify-content-between align-items-center">
                   <h5 className="mb-0 fw-bold text-white"><FaStethoscope className="me-2"/>Clinical Review</h5>
                   <button className="btn btn-sm btn-outline-light rounded-circle" onClick={closeModal}><FaTimes/></button>
               </div>

               <div className="p-4">
                   <div className="row">
                       {/* Left: Info */}
                       <div className="col-md-6 border-end border-secondary">
                           <h2 className="fw-bold mb-0 text-white">{selectedPatient.name}</h2>
                           <p className="text-white-50 mb-4">{selectedPatient.age} • {selectedPatient.location}</p>
                           
                           <div className="mb-4">
                               <label className="small fw-bold text-white-50 text-uppercase mb-1">AI Diagnosis</label>
                               <div className="d-flex align-items-center">
                                   <h3 className="fw-bold me-3 mb-0 text-white">{selectedPatient.diagnosis}</h3>
                                   {getBadge(selectedPatient.risk_level)}
                               </div>
                           </div>
                           
                           {/* Confidence Bars */}
                           <div className="mb-4">
                               <label className="small fw-bold text-white-50 text-uppercase mb-2"><FaChartBar className="me-1"/> Confidence Scores</label>
                               <div className="p-3 rounded-3" style={{background: 'rgba(0,0,0,0.3)'}}>
                                   {/* Pneumonia */}
                                   <div className="d-flex justify-content-between small mb-1 text-white">
                                       <span>Pneumonia</span>
                                       <span className="fw-bold">{((selectedPatient.biomarkers?.prob_pneumonia || 0) * 100).toFixed(1)}%</span>
                                   </div>
                                   <div className="progress mb-3" style={{height: '6px', background:'rgba(255,255,255,0.1)'}}>
                                       <div className="progress-bar bg-danger" style={{width: `${(selectedPatient.biomarkers?.prob_pneumonia || 0) * 100}%`}}></div>
                                   </div>
                                   {/* Normal */}
                                   <div className="d-flex justify-content-between small mb-1 text-white">
                                       <span>Normal / Healthy</span>
                                       <span className="fw-bold">{((selectedPatient.biomarkers?.prob_normal || 0) * 100).toFixed(1)}%</span>
                                   </div>
                                   <div className="progress" style={{height: '6px', background:'rgba(255,255,255,0.1)'}}>
                                       <div className="progress-bar bg-success" style={{width: `${(selectedPatient.biomarkers?.prob_normal || 0) * 100}%`}}></div>
                                   </div>
                               </div>
                           </div>

                           <div className="mb-3">
                               <label className="small fw-bold text-white-50 text-uppercase">Symptoms</label>
                               <div className="p-2 rounded fst-italic text-white-50" style={{background: 'rgba(0,0,0,0.3)'}}>"{selectedPatient.symptoms}"</div>
                           </div>
                       </div>

                       {/* Right: Analysis & Actions */}
                       <div className="col-md-6 ps-md-4">
                           <div className="mb-3">
                               <label className="small fw-bold text-white-50 text-uppercase mb-1"><FaNotesMedical className="me-1"/> Interpretation</label>
                               {/* Dark Blue Glass Interpretation Box */}
                               <div className="p-3 border border-info rounded-3 text-white small" style={{background: 'rgba(9, 132, 227, 0.15)'}}>
                                   {getClinicalExplanation(selectedPatient)}
                               </div>
                           </div>

                           <div className="mb-3">
                               <label className="small fw-bold text-white-50 text-uppercase mb-1">Audio Recording</label>
                               {selectedPatient.recording_url ? (
                                   <div className="p-3 rounded-3 text-center" style={{background: 'rgba(0,0,0,0.3)'}}>
                                       {audioLoading && <div className="text-info small fw-bold"><FaSpinner className="spin me-2"/>Loading...</div>}
                                       {audioError && <div className="text-danger small fw-bold">{audioError}</div>}
                                       {audioBlobUrl && (
                                           <audio controls style={{width: '100%'}} src={audioBlobUrl} />
                                       )}
                                       <div className="mt-2 pt-2 border-top border-secondary">
                                            <a href={selectedPatient.recording_url.startsWith('http') ? selectedPatient.recording_url : `${config.SERVER_URL}/uploads/${selectedPatient.recording_url}`} 
                                               target="_blank" rel="noopener noreferrer" className="small text-white-50 text-decoration-underline">
                                                (Download Original)
                                            </a>
                                       </div>
                                   </div>
                               ) : (
                                   <div className="alert alert-warning small py-2 text-center">No Audio Available</div>
                               )}
                           </div>

                           {/* ACTION BUTTONS */}
                           <div className="d-grid gap-2 mt-4">
                               <button onClick={handleWhatsApp} className="btn btn-success fw-bold text-white shadow-sm d-flex align-items-center justify-content-center">
                                   <FaWhatsapp className="me-2"/> Contact Caregiver
                               </button>
                               <button onClick={handleArchive} className="btn btn-primary fw-bold text-white shadow-sm d-flex align-items-center justify-content-center">
                                   <FaArchive className="me-2"/> Resolve & Archive
                               </button>
                           </div>
                       </div>
                   </div>
               </div>
           </div>
        </div>
      )}

    </div>
  );
};

export default PatientList;