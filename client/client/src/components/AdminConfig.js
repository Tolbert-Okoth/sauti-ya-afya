/* client/src/components/AdminConfig.js */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaRobot, FaLock, FaServer, FaSlidersH } from 'react-icons/fa';
import axios from 'axios';
import { auth } from '../firebase';
import { useTranslation } from '../hooks/useTranslation';

const AdminConfig = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();
  const [loading, setLoading] = useState(true);
  
  const [config, setConfig] = useState({
    ai_model: 'Llama 3.3 (v2.3) + Librosa DSP',
    confidence_threshold: 0.75,
    export_moh: false,
    retain_logs: true
  });

  // Custom Glass Input Style
  const glassInputStyle = {
      background: 'rgba(255,255,255,0.4)',
      border: '1px solid rgba(255,255,255,0.3)',
      color: '#2d3436',
      fontWeight: '500'
  };

  // 1. Load Config
  useEffect(() => {
    const fetchConfig = async () => {
      try {
        const token = await auth.currentUser.getIdToken();
        const res = await axios.get('http://localhost:5000/api/settings', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setConfig(prev => ({ ...prev, ...res.data }));
      } catch (err) {
        console.error("Load failed", err);
      } finally {
        setLoading(false);
      }
    };
    fetchConfig();
  }, []);

  // 2. Save Config Helper
  const saveConfig = async (newConfig) => {
    setConfig(newConfig); // Optimistic Update
    try {
      const token = await auth.currentUser.getIdToken();
      await axios.put('http://localhost:5000/api/settings', 
        newConfig, 
        { headers: { Authorization: `Bearer ${token}` } }
      );
    } catch (err) {
      console.error("Save failed", err);
    }
  };

  const handleToggle = (key) => {
    saveConfig({ ...config, [key]: !config[key] });
  };

  const handleSliderChange = (e) => {
    saveConfig({ ...config, confidence_threshold: parseFloat(e.target.value) });
  };

  if (loading) return <div className="p-5 text-center text-dark-brown">Loading Admin Config...</div>;

  return (
    <div className="container p-0" style={{ maxWidth: '700px' }}>
      <button className="btn btn-link text-dark-brown text-decoration-none mb-3 p-0 fw-bold" onClick={() => navigate(-1)}>
        <FaArrowLeft /> {t('back')}
      </button>

      <div className="glass-card p-5 border-top border-4 border-danger position-relative overflow-hidden">
        {/* Decorative Background Icon */}
        <FaRobot className="position-absolute text-dark opacity-10" style={{right: '-30px', bottom: '-30px', fontSize: '12rem'}} />

        <div className="d-flex align-items-center mb-5 position-relative">
            <div className="bg-danger bg-opacity-10 p-3 rounded-circle me-3 text-danger">
                <FaSlidersH size={24} />
            </div>
            <div>
                <h4 className="fw-bold text-dark-brown mb-0">{t('menu_admin_config')}</h4>
                <small className="text-muted">Global System Parameters</small>
            </div>
        </div>

        <div className="position-relative">
            <h6 className="text-danger fw-bold mb-4 d-flex align-items-center">
                <FaLock className="me-2" /> {t('admin_access')}
            </h6>

            {/* AI MODEL (Read Only) */}
            <div className="mb-4">
                <label className="form-label fw-bold text-dark-brown small text-uppercase">{t('ai_model')}</label>
                <div className="input-group">
                    <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.4)'}}><FaServer className="text-muted"/></span>
                    <input 
                        type="text" 
                        className="form-control" 
                        style={glassInputStyle} 
                        value={config.ai_model} 
                        readOnly 
                    />
                </div>
            </div>

            {/* CONFIDENCE SLIDER */}
            <div className="mb-5 bg-white bg-opacity-25 p-3 rounded border border-light">
                <label className="form-label fw-bold d-flex justify-content-between align-items-center text-dark-brown mb-3">
                    <span>{t('confidence_threshold')}</span>
                    <span className="badge bg-primary shadow-sm" style={{fontSize: '1rem'}}>{config.confidence_threshold}</span>
                </label>
                <input 
                    type="range" 
                    className="form-range" 
                    min="0.5" 
                    max="0.99" 
                    step="0.01" 
                    value={config.confidence_threshold}
                    onChange={handleSliderChange}
                />
                <div className="d-flex justify-content-between small text-muted fw-bold mt-2">
                    <span>{t('low')} (0.5)</span>
                    <span>{t('strict')} (1.0)</span>
                </div>
            </div>

            <hr className="text-muted opacity-25 my-4" />

            {/* SYSTEM TOGGLES */}
            <div className="d-flex justify-content-between align-items-center mb-3 p-2 rounded hover-glass">
                <label className="form-check-label fw-bold text-dark-brown">{t('export_moh')}</label>
                <div className="form-check form-switch">
                    <input 
                        className="form-check-input" 
                        type="checkbox" 
                        style={{width: '3em', height: '1.5em'}}
                        checked={config.export_moh}
                        onChange={() => handleToggle('export_moh')}
                    />
                </div>
            </div>

            <div className="d-flex justify-content-between align-items-center p-2 rounded hover-glass">
                <label className="form-check-label fw-bold text-dark-brown">{t('retain_logs')}</label>
                <div className="form-check form-switch">
                    <input 
                        className="form-check-input" 
                        type="checkbox" 
                        style={{width: '3em', height: '1.5em'}}
                        checked={config.retain_logs}
                        onChange={() => handleToggle('retain_logs')}
                    />
                </div>
            </div>

        </div>
      </div>
    </div>
  );
};

export default AdminConfig;