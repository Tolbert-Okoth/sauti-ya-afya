/* client/src/components/SyncSettings.js */
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaCloudUploadAlt, FaDatabase, FaTrashAlt, FaWifi } from 'react-icons/fa';
import axios from 'axios';
import { auth } from '../firebase';
import config from '../config'; // 🟢 IMPORT CONFIG FOR LIVE URL

const SyncSettings = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [storageUsed, setStorageUsed] = useState("0.00");
  const [settings, setSettings] = useState({
    offline_mode: false,
    auto_upload: true
  });

  const calculateUsage = () => {
    let totalBytes = 0;
    for (let key in localStorage) {
      if (localStorage.hasOwnProperty(key)) {
        const amount = (localStorage[key].length + key.length) * 2;
        totalBytes += amount;
      }
    }
    setStorageUsed((totalBytes / 1024 / 1024).toFixed(2));
  };

  useEffect(() => {
    const initData = async () => {
      try {
        const token = await auth.currentUser.getIdToken();
        // 🟢 FIX: Use config.API_BASE_URL
        const res = await axios.get(`${config.API_BASE_URL}/settings`, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setSettings(res.data);
        calculateUsage(); 
      } catch (err) {
        console.error("Failed to load data", err);
      } finally {
        setLoading(false);
      }
    };
    initData();
  }, []);

  const handleToggle = async (key) => {
    const newValue = !settings[key];
    const updatedSettings = { ...settings, [key]: newValue };
    setSettings(updatedSettings);

    try {
      const token = await auth.currentUser.getIdToken();
      // 🟢 FIX: Use config.API_BASE_URL
      await axios.put(`${config.API_BASE_URL}/settings`, updatedSettings, { 
          headers: { Authorization: `Bearer ${token}` } 
      });
    } catch (err) {
      setSettings(settings);
    }
  };

  const handleClearCache = () => {
    if (window.confirm("⚠️ Are you sure? This will delete all offline records stored on this device.")) {
      localStorage.removeItem('encrypt_local');
      calculateUsage();
      alert("✅ Storage cleared successfully.");
    }
  };

  if (loading) return <div className="p-4 text-center text-white">Loading preferences...</div>;

  return (
    <div className="container p-0" style={{ maxWidth: '600px' }}>
      <button className="btn btn-link text-white text-decoration-none mb-3 p-0 fw-bold" onClick={() => navigate(-1)}>
        <FaArrowLeft /> Back
      </button>

      <div className="d-flex align-items-center mb-4">
        <div className="bg-info rounded-circle p-2 text-white me-3 shadow-sm">
            <FaCloudUploadAlt size={20} />
        </div>
        <h4 className="fw-bold text-white mb-0">Data & Sync</h4>
      </div>

      {/* SYNC CONTROLS */}
      <div className="glass-card mb-4 p-4">
        <div className="d-flex justify-content-between align-items-center mb-3">
            <div>
                <label className="form-check-label fw-bold text-white">Offline Mode</label>
                <p className="small text-white-50 mb-0">Disables all network calls. Data saves locally.</p>
            </div>
            <div className="form-check form-switch">
                <input 
                    className="form-check-input" 
                    type="checkbox" 
                    checked={settings.offline_mode} 
                    onChange={() => handleToggle('offline_mode')} 
                    style={{width: '3em', height: '1.5em'}}
                />
            </div>
        </div>

        <hr className="text-light opacity-25 my-3"/>

        <div className="d-flex justify-content-between align-items-center">
            <div className="d-flex align-items-center">
                 <FaWifi className="me-3 text-white-50"/>
                 <label className="form-check-label fw-bold text-white">Auto-Upload on Wi-Fi</label>
            </div>
            <div className="form-check form-switch">
                <input 
                    className="form-check-input" 
                    type="checkbox" 
                    checked={settings.auto_upload} 
                    onChange={() => handleToggle('auto_upload')} 
                    disabled={settings.offline_mode}
                    style={{width: '3em', height: '1.5em'}}
                />
            </div>
        </div>
      </div>

      {/* REAL STORAGE STATUS */}
      <div className="glass-card p-4">
          <h6 className="fw-bold text-white mb-4 border-bottom border-secondary pb-2"><FaDatabase className="me-2 text-info"/>Storage Status</h6>
          
          <div className="d-flex justify-content-between align-items-center mb-2">
              <span className="text-white">Pending Uploads</span>
              <span className="badge bg-white text-dark shadow-sm">0 Cases</span>
          </div>

          <div className="d-flex justify-content-between align-items-center mb-1">
              <span className="text-white">Browser Storage Used</span>
              <span className={`fw-bold ${storageUsed > 4.5 ? 'text-danger' : 'text-info'}`}>{storageUsed} MB</span>
          </div>
          
          {/* Glass Storage Bar */}
          <div className="progress mb-2" style={{ height: '12px', background: 'rgba(255,255,255,0.1)', borderRadius: '6px' }}>
              <div 
                  className={`progress-bar rounded-pill ${storageUsed > 4.5 ? 'bg-danger' : 'bg-info'}`} 
                  role="progressbar" 
                  style={{ width: `${(storageUsed / 5) * 100}%`, transition: 'width 0.3s ease' }} 
              ></div>
          </div>
          <p className="text-white-50 small text-end mb-4">Max Capacity: ~5.00 MB</p>
          
          <div className="d-grid gap-2">
              <button className="btn btn-primary shadow-sm" disabled={settings.offline_mode}>
                  <FaCloudUploadAlt className="me-2" /> Sync Now
              </button>
              
              <button className="btn btn-outline-danger shadow-sm border-2" onClick={handleClearCache}>
                  <FaTrashAlt className="me-2" /> Free Up Space
              </button>
          </div>

      </div>
    </div>
  );
};

export default SyncSettings;