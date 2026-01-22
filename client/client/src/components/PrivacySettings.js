/* client/src/components/PrivacySettings.js */
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaKey, FaShieldAlt, FaMobileAlt, FaLock } from 'react-icons/fa';
import { sendPasswordResetEmail } from 'firebase/auth';
import { auth } from '../firebase';

const PrivacySettings = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  
  const [encryptLocal, setEncryptLocal] = useState(
    localStorage.getItem('encrypt_local') === 'true'
  );

  const handleChangePassword = async () => {
    const user = auth.currentUser;
    if (!user || !user.email) return;

    if (window.confirm(`Send a password reset email to ${user.email}?`)) {
        setLoading(true);
        try {
            await sendPasswordResetEmail(auth, user.email);
            alert("✅ Password reset link sent! Check your email.");
        } catch (error) {
            console.error(error);
            alert("❌ Error: " + error.message);
        } finally {
            setLoading(false);
        }
    }
  };

  const handle2FAToggle = () => {
    alert("ℹ️ To enable Two-Factor Authentication (MFA), please contact the System Administrator.");
  };

  const handleEncryptionToggle = () => {
    const newValue = !encryptLocal;
    setEncryptLocal(newValue);
    localStorage.setItem('encrypt_local', newValue);
  };

  return (
    <div className="container p-0" style={{ maxWidth: '600px' }}>
      <button className="btn btn-link text-dark-brown text-decoration-none mb-3 p-0 fw-bold" onClick={() => navigate(-1)}>
        <FaArrowLeft /> Back
      </button>

      <div className="d-flex align-items-center mb-4">
        <div className="bg-white rounded-circle p-2 text-dark me-3 shadow-sm">
            <FaLock />
        </div>
        <h4 className="fw-bold text-dark-brown mb-0">Privacy & Security</h4>
      </div>

      {/* PASSWORD SECTION */}
      <div className="glass-card mb-4 overflow-hidden">
        <div className="p-4">
            <div className="d-flex justify-content-between align-items-center mb-3 p-2 rounded hover-glass">
                <div className="d-flex align-items-center">
                    <FaKey className="text-accent me-3 fs-5" />
                    <div>
                        <span className="fw-bold d-block text-dark-brown">Change Password</span>
                        <small className="text-muted">Via Email Reset</small>
                    </div>
                </div>
                <button 
                    className="btn btn-sm btn-dark rounded-pill px-3" 
                    onClick={handleChangePassword}
                    disabled={loading}
                >
                    {loading ? "Sending..." : "Update"}
                </button>
            </div>
            
            <hr className="text-light opacity-50 my-2"/>

            <div className="d-flex justify-content-between align-items-center p-2 rounded hover-glass">
                <div className="d-flex align-items-center">
                      <FaMobileAlt className="text-secondary me-3 fs-5" />
                      <span className="text-dark-brown fw-bold">Two-Factor Auth (2FA)</span>
                </div>
                <div className="form-check form-switch">
                    <input 
                        className="form-check-input" 
                        type="checkbox" 
                        onChange={handle2FAToggle} 
                        style={{ cursor: 'pointer', width: '3em', height: '1.5em' }}
                    />
                </div>
            </div>
        </div>
      </div>

      {/* DATA PROTECTION SECTION */}
      <div className="glass-card p-4">
          <h6 className="fw-bold mb-3 text-dark-brown d-flex align-items-center">
              <FaShieldAlt className="me-2 text-accent"/> Data Protection
          </h6>
          
          <div className="d-flex justify-content-between align-items-center mb-3">
            <div>
                <label className="form-check-label fw-bold text-dark-brown">Encrypt Local Storage</label>
                <p className="small text-muted mb-0">Obfuscate patient data cached on this device.</p>
            </div>
            <div className="form-check form-switch">
                <input 
                    className="form-check-input" 
                    type="checkbox" 
                    checked={encryptLocal}
                    onChange={handleEncryptionToggle}
                    style={{ width: '3em', height: '1.5em' }}
                />
            </div>
          </div>

          <div className="d-flex justify-content-between align-items-center opacity-50">
             <label className="form-check-label text-dark-brown">Share Crash Reports</label>
             <div className="form-check form-switch">
                <input className="form-check-input" type="checkbox" disabled style={{ width: '3em', height: '1.5em' }}/>
             </div>
          </div>
      </div>
    </div>
  );
};

export default PrivacySettings;