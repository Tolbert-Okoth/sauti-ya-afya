/* client/src/components/PatientList.js */
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { auth } from '../firebase';
import config from '../config'; // ✅ IMPORT CONFIG
import { 
  FaSearch, FaFilter, FaUserCircle, FaMapMarkerAlt, FaClock, 
  FaTimes, FaWhatsapp, FaCheck, FaSpinner, FaExclamationTriangle
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

  const glassInputStyle = {
      background: 'rgba(255,255,255,0.4)',
      border: '1px solid rgba(255,255,255,0.3)',
      color: '#2d3436',
      backdropFilter: 'blur(5px)'
  };

  useEffect(() => {
    const fetchMyPatients = async () => {
      try {
        const token = await auth.currentUser.getIdToken();
        // ✅ USE CONFIG API URL
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

  // 🛠️ FETCH AUDIO BLOB WHEN PATIENT IS SELECTED
  useEffect(() => {
    if (selectedPatient && selectedPatient.recording_url) {
        const fetchAudio = async () => {
            setAudioLoading(true);
            setAudioError(null);
            setAudioBlobUrl(null);

            try {
                // ✅ FIXED LOGIC: Handle both Cloud URLs (https://) and Local filenames
                const fileUrl = selectedPatient.recording_url.startsWith('http')
                    ? selectedPatient.recording_url
                    : `${config.SERVER_URL}/uploads/${selectedPatient.recording_url}`;

                const response = await fetch(fileUrl);
                
                if (!response.ok) throw new Error(`Server returned ${response.status}`);

                const blob = await response.blob();
                const objectUrl = URL.createObjectURL(blob);
                
                setAudioBlobUrl(objectUrl);
            } catch (err) {
                console.error("Audio Load Error:", err);
                setAudioError(`Load Failed: ${err.message}`);
            } finally {
                setAudioLoading(false);
            }
        };
        fetchAudio();
    }
  }, [selectedPatient]);

  const closeModal = () => {
      setSelectedPatient(null);
      setAudioBlobUrl(null);
  };

  const filteredPatients = patients.filter(p => {
    const matchesSearch = p.name.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          p.location.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesRisk = filterRisk === 'All' || p.risk_level === filterRisk;
    return matchesSearch && matchesRisk;
  });

  const getBadge = (risk) => {
    switch(risk) {
      case 'High': return <span className="badge bg-danger shadow-sm px-3 py-2 rounded-pill">🔴 High Risk</span>;
      case 'Medium': return <span className="badge bg-warning text-dark shadow-sm px-3 py-2 rounded-pill">🟡 Moderate</span>;
      case 'Low': return <span className="badge bg-success shadow-sm px-3 py-2 rounded-pill">🟢 Stable</span>;
      default: return <span className="badge bg-secondary rounded-pill">Unknown</span>;
    }
  };

  if (loading) return <div className="p-4 text-center text-dark-brown animate-pulse">Loading patient records...</div>;

  return (
    <div className="container-fluid p-0">
      
      {/* Header Section */}
      <div className="d-flex flex-column flex-md-row justify-content-between align-items-md-center mb-4">
        <h3 className="fw-bold mb-2 mb-md-0 text-dark-brown">My Patients</h3>
        <span className="badge bg-white text-dark-brown border shadow-sm align-self-start align-self-md-center">{filteredPatients.length} Records</span>
      </div>

      <div className="row g-3 mb-4">
        <div className="col-12 col-md-8">
          <div className="input-group">
            <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.4)'}}><FaSearch className="text-muted"/></span>
            <input 
              type="text" 
              className="form-control border-0" 
              placeholder="Search by name or location..." 
              style={glassInputStyle}
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
        <div className="col-12 col-md-4">
          <div className="input-group">
             <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.4)'}}><FaFilter className="text-muted"/></span>
             <select 
                className="form-select border-0" 
                style={glassInputStyle}
                value={filterRisk} 
                onChange={(e) => setFilterRisk(e.target.value)}
             >
               <option value="All">All Risks</option>
               <option value="High">🔴 High Risk Only</option>
               <option value="Medium">🟡 Moderate Risk</option>
               <option value="Low">🟢 Stable</option>
             </select>
          </div>
        </div>
      </div>

      {/* Patient Cards List */}
      <div className="row">
        {filteredPatients.length > 0 ? (
          filteredPatients.map((p) => (
            <div key={p.id} className="col-12 mb-3">
              <div 
                className="glass-card border-0 p-3 transition-all hover-shadow" 
                style={{cursor: 'pointer'}}
                onClick={() => setSelectedPatient(p)}
              >
                <div className="d-flex align-items-center">
                  <div className="me-3">
                    <div className="bg-white rounded-circle p-2 p-md-3 text-accent shadow-sm d-flex align-items-center justify-content-center" style={{width: '50px', height: '50px'}}>
                        <FaUserCircle size={24} />
                    </div>
                  </div>
                  <div className="flex-grow-1">
                    <h5 className="mb-1 fw-bold text-dark-brown">{p.name}</h5>
                    <div className="small text-muted mb-1 d-flex flex-wrap align-items-center">
                        <span className="me-3 bg-white bg-opacity-50 px-2 rounded mb-1 mb-md-0">Age: <strong>{p.age}</strong></span>
                        <span><FaMapMarkerAlt className="me-1 text-accent"/>{p.location}</span>
                    </div>
                    <div className="small text-muted opacity-75">
                        <FaClock className="me-1"/> 
                        {new Date(p.created_at).toLocaleDateString()}
                    </div>
                  </div>
                  <div className="text-end">
                    <div className="mb-2 d-none d-sm-block">{getBadge(p.risk_level)}</div>
                    <small className="d-block text-dark-brown fw-bold" style={{fontSize: '0.8rem'}}>
                      {p.diagnosis}
                    </small>
                  </div>
                </div>
              </div>
            </div>
          ))
        ) : (
            <div className="col-12 text-center py-5 text-muted glass-card">
                <p className="mb-0">No patients found matching your search.</p>
            </div>
        )}
      </div>

      {/* CLINICAL MODAL */}
      {selectedPatient && (
        <div style={{
            position: 'fixed', top: 0, left: 0, width: '100%', height: '100%',
            background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(5px)',
            zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
           <div className="bg-white p-0 rounded-4 shadow-lg overflow-hidden animate-slide-in" style={{width: '90%', maxWidth: '800px', maxHeight: '90vh', overflowY: 'auto'}}>
               
               <div className="p-3 border-bottom d-flex justify-content-between align-items-center bg-light">
                   <h5 className="mb-0 fw-bold text-dark-brown">🩺 Clinical Case Review</h5>
                   <button className="btn btn-sm btn-light rounded-circle" onClick={closeModal}><FaTimes/></button>
               </div>

               <div className="p-4">
                   <div className="row">
                       <div className="col-md-6 border-end">
                           <h2 className="fw-bold text-dark-brown mb-0">{selectedPatient.name}</h2>
                           <p className="text-muted">{selectedPatient.age} • {selectedPatient.location}</p>
                           <div className="mb-3">
                               <label className="small fw-bold text-muted text-uppercase">Symptoms</label>
                               <div className="p-2 bg-light rounded text-dark">{selectedPatient.symptoms}</div>
                           </div>
                           <div className="mb-3">
                               <label className="small fw-bold text-muted text-uppercase">AI Diagnosis</label>
                               <h4 className={`fw-bold ${selectedPatient.risk_level === 'High' ? 'text-danger' : 'text-success'}`}>
                                   {selectedPatient.diagnosis}
                               </h4>
                           </div>
                       </div>

                       <div className="col-md-6 ps-md-4">
                           {/* Spectrogram */}
                           <div className="border border-success rounded-3 overflow-hidden mb-3 position-relative" style={{height: '180px', background: 'black'}}>
                               {selectedPatient.spectrogram ? (
                                   <img src={selectedPatient.spectrogram} alt="Spectrogram" style={{width: '100%', height: '100%', objectFit: 'cover'}} />
                               ) : (
                                   <div className="d-flex align-items-center justify-content-center h-100 text-white-50">No Visual Data</div>
                               )}
                           </div>

                           {/* 🛠️ AUDIO PLAYER */}
                           <div className="mb-3">
                               <label className="small fw-bold text-muted text-uppercase mb-1">Audio Recording</label>
                               
                               {selectedPatient.recording_url ? (
                                   <div className="p-3 bg-light rounded-3 text-center">
                                       
                                       {/* Loading State */}
                                       {audioLoading && (
                                           <div className="text-primary fw-bold">
                                               <FaSpinner className="spin me-2"/> Downloading Audio...
                                           </div>
                                       )}

                                       {/* Error State */}
                                       {audioError && (
                                           <div className="text-danger small fw-bold">
                                               <FaExclamationTriangle className="me-1"/> {audioError}
                                           </div>
                                       )}

                                       {/* Player (Only shows when blob is ready) */}
                                       {audioBlobUrl && (
                                           <audio 
                                               controls 
                                               autoPlay 
                                               style={{width: '100%'}} 
                                               src={audioBlobUrl}
                                           >
                                               Your browser does not support the audio element.
                                           </audio>
                                       )}
                                       
                                       {/* Fallback Direct Link */}
                                       <div className="mt-2 pt-2 border-top">
                                           <a 
                                               // ✅ FIXED: Check if it's already a full URL or a local file
                                               href={selectedPatient.recording_url.startsWith('http') 
                                                   ? selectedPatient.recording_url 
                                                   : `${config.SERVER_URL}/uploads/${selectedPatient.recording_url}`
                                               }
                                               target="_blank" 
                                               rel="noopener noreferrer"
                                               className="small text-muted text-decoration-underline"
                                           >
                                               (Direct Download Link)
                                           </a>
                                       </div>
                                   </div>
                               ) : (
                                   <div className="alert alert-warning text-center small p-2">
                                       No Audio File Attached
                                   </div>
                               )}
                           </div>

                           <div className="d-grid gap-2">
                               <button className="btn btn-success text-white fw-bold shadow-sm d-flex align-items-center justify-content-center">
                                   <FaWhatsapp className="me-2"/> Contact Caregiver
                               </button>
                               <button className="btn btn-primary text-white fw-bold shadow-sm d-flex align-items-center justify-content-center" onClick={closeModal}>
                                   <FaCheck className="me-2"/> Resolve & Archive
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