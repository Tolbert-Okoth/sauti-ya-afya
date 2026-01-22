/* client/src/components/NotificationSettings.js */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaBell, FaWhatsapp, FaExclamationTriangle } from 'react-icons/fa';
import axios from 'axios';
import { auth } from '../firebase';

const NotificationSettings = () => {
  const navigate = useNavigate();
  const [enabled, setEnabled] = useState(true);

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const token = await auth.currentUser.getIdToken();
        const res = await axios.get('http://localhost:5000/api/settings', {
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
      await axios.put('http://localhost:5000/api/settings', 
        { notifications_enabled: newValue }, 
        { headers: { Authorization: `Bearer ${token}` } }
      );
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="container p-0" style={{ maxWidth: '600px' }}>
      <button className="btn btn-link text-dark-brown text-decoration-none mb-3 p-0 fw-bold" onClick={() => navigate(-1)}>
        <FaArrowLeft /> Back
      </button>

      <div className="d-flex align-items-center mb-4">
        <div className="bg-white rounded-circle p-2 text-warning me-3 shadow-sm">
            <FaBell size={20} />
        </div>
        <h4 className="fw-bold text-dark-brown mb-0">Notifications & Alerts</h4>
      </div>

      <div className="glass-card mb-4 overflow-hidden border-start border-4 border-danger">
        <div className="p-4">
            <h6 className="fw-bold text-danger mb-3 d-flex align-items-center">
                <FaExclamationTriangle className="me-2"/> Urgent Clinical Alerts
            </h6>
            <div className="d-flex justify-content-between align-items-center">
                <div>
                    <span className="d-block fw-bold text-dark-brown">High-Risk Case Detected</span>
                    <small className="text-muted">Receive immediate popups for critical triage results.</small>
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
        <h6 className="fw-bold text-dark-brown mb-4">Delivery Channels</h6>
        
        <div className="d-flex justify-content-between align-items-center p-3 rounded" style={{background: 'rgba(255,255,255,0.3)'}}>
            <div className="d-flex align-items-center">
                <FaWhatsapp className="text-success fs-4 me-3"/>
                <div>
                    <span className="fw-bold text-dark-brown d-block">WhatsApp Alerts</span>
                    <small className="text-muted fst-italic">Coming soon in v2.0</small>
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