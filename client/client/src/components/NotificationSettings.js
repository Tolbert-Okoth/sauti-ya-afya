/* client/src/components/NotificationSettings.js */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaBell, FaWhatsapp, FaExclamationTriangle } from 'react-icons/fa';
import axios from 'axios';
import { auth } from '../firebase';
import config from '../config'; // 🟢 IMPORT CONFIG FOR LIVE URL

const NotificationSettings = () => {
  const navigate = useNavigate();
  const [enabled, setEnabled] = useState(true);

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const token = await auth.currentUser.getIdToken();
        // 🟢 FIX: Use config.API_BASE_URL
        const res = await axios.get(`${config.API_BASE_URL}/settings`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setEnabled(res.data.notifications_enabled);
      } catch (err) {
        console.error(err);
      }
    };
    fetchSettings();
  }, []);

  const toggleNotifications = async () => {
    const newValue = !enabled;
    setEnabled(newValue);
    try {
      const token = await auth.currentUser.getIdToken();
      // 🟢 FIX: Use config.API_BASE_URL
      await axios.put(`${config.API_BASE_URL}/settings`, 
        { notifications_enabled: newValue }, 
        { headers: { Authorization: `Bearer ${token}` } }
      );
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="container p-0" style={{ maxWidth: '600px' }}>
      <button className="btn btn-link text-white text-decoration-none mb-3 p-0 fw-bold" onClick={() => navigate(-1)}>
        <FaArrowLeft /> Back
      </button>

      <div className="d-flex align-items-center mb-4">
        <div className="bg-warning rounded-circle p-2 text-dark me-3 shadow-sm">
            <FaBell size={20} />
        </div>
        <h4 className="fw-bold text-white mb-0">Notifications & Alerts</h4>
      </div>

      <div className="glass-card mb-4 overflow-hidden border-start border-4 border-danger">
        <div className="p-4">
            <h6 className="fw-bold text-danger mb-3 d-flex align-items-center">
                <FaExclamationTriangle className="me-2"/> Urgent Clinical Alerts
            </h6>
            <div className="d-flex justify-content-between align-items-center">
                <div>
                    <span className="d-block fw-bold text-white">High-Risk Case Detected</span>
                    <small className="text-white-50">Receive immediate popups for critical triage results.</small>
                </div>
                <div className="form-check form-switch">
                    <input 
                        className="form-check-input" 
                        type="checkbox" 
                        style={{width: '3em', height: '1.5em'}}
                        checked={enabled}
                        onChange={toggleNotifications}
                    />
                </div>
            </div>
        </div>
      </div>

      <div className="glass-card p-4">
        <h6 className="fw-bold text-white mb-4">Delivery Channels</h6>
        
        {/* Darker Glass Inner Card */}
        <div className="d-flex justify-content-between align-items-center p-3 rounded" style={{background: 'rgba(0,0,0,0.3)'}}>
            <div className="d-flex align-items-center">
                <FaWhatsapp className="text-success fs-4 me-3"/>
                <div>
                    <span className="fw-bold text-white d-block">WhatsApp Alerts</span>
                    <small className="text-white-50 fst-italic">Coming soon in v2.0</small>
                </div>
            </div>
            <div className="form-check form-switch">
                <input className="form-check-input" type="checkbox" disabled checked={false} />
            </div>
        </div>
      </div>
    </div>
  );
};

export default NotificationSettings;