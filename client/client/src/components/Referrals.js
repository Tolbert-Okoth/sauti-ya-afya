/* client/src/components/Referrals.js */
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { auth } from '../firebase';
import { FaCheckCircle, FaHourglassHalf, FaExternalLinkAlt, FaUserInjured, FaArrowRight } from 'react-icons/fa';
import config from '../config'; // 🟢 IMPORT CONFIG FOR LIVE URL

const Referrals = () => {
  const [referrals, setReferrals] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchReferrals = async () => {
      try {
        const token = await auth.currentUser.getIdToken();
        // 🟢 FIX: Use config.API_BASE_URL instead of localhost
        const res = await axios.get(`${config.API_BASE_URL}/patients`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        
        const relevantCases = res.data.filter(p => p.risk_level === 'High' || p.risk_level === 'Medium');
        setReferrals(relevantCases);
      } catch (err) {
        console.error("Error fetching referrals:", err);
      } finally {
        setLoading(false);
      }
    };

    fetchReferrals();
  }, []);

  if (loading) return <div className="p-4 text-center text-white animate-pulse">Loading Referral Status...</div>;

  return (
    <div className="container p-0">
      <div className="d-flex align-items-center mb-4">
        <div className="bg-warning rounded-circle p-2 text-dark me-3 shadow-sm">
            <FaExternalLinkAlt />
        </div>
        <h3 className="fw-bold text-white mb-0">My Referrals</h3>
      </div>
      
      <div className="row">
        {referrals.length > 0 ? (
          referrals.map((p) => (
            <div key={p.id} className="col-12 mb-3">
              <div className="glass-card p-4 transition-all hover-shadow">
                <div className="d-flex justify-content-between align-items-start mb-3">
                    <div className="d-flex align-items-center">
                        <div className="bg-primary rounded-circle p-3 text-white me-3 shadow-sm d-flex align-items-center justify-content-center" style={{width: '50px', height: '50px'}}>
                            <FaUserInjured size={24} />
                        </div>
                        <div>
                            <h5 className="fw-bold text-white mb-0">{p.name}</h5>
                            <span className="badge bg-secondary text-light border border-secondary mt-1">Ticket #{p.id.toString().substring(0,4)}</span>
                        </div>
                    </div>
                    
                    <span className="badge bg-warning text-dark shadow-sm rounded-pill px-3 py-2">
                        <FaHourglassHalf className="me-1"/> Pending Review
                    </span>
                </div>
                
                {/* 🟢 Darker Glass Box for Diagnosis */}
                <div className="p-3 rounded mb-3" style={{background: 'rgba(0,0,0,0.2)'}}>
                    <small className="text-muted d-block text-uppercase fw-bold" style={{fontSize: '0.7rem'}}>Diagnosis</small>
                    <div className="fw-bold text-white">{p.diagnosis}</div>
                    <small className="text-muted mt-1 d-block"><FaArrowRight className="me-1 text-info"/> Referred on {new Date(p.created_at).toLocaleDateString()}</small>
                </div>

                {/* Glass Progress Bar */}
                <div className="mt-3">
                    <div className="d-flex justify-content-between small text-muted fw-bold mb-1">
                        <span>Upload</span>
                        <span>Doctor Review</span>
                        <span className="opacity-50">Resolution</span>
                    </div>
                    <div className="progress" style={{height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px'}}>
                        <div 
                            className="progress-bar progress-bar-striped progress-bar-animated bg-warning" 
                            role="progressbar" 
                            style={{ width: '60%', borderRadius: '4px' }} 
                        ></div>
                    </div>
                </div>

              </div>
            </div>
          ))
        ) : (
          <div className="col-12 text-center py-5 glass-card text-muted">
            <FaCheckCircle size={40} className="text-success mb-3"/>
            <h5 className="text-white">All Clear!</h5>
            <p className="mb-0">You have no pending high-risk referrals.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Referrals;