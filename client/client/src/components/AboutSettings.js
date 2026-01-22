/* client/src/components/AboutSettings.js */
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaHeartbeat, FaEnvelope, FaCode, FaBuilding } from 'react-icons/fa';
import { useTranslation } from '../hooks/useTranslation';

const AboutSettings = () => {
  const navigate = useNavigate();
  const { t } = useTranslation();

  // 📧 The Contact Function
  const handleContact = () => {
    const subject = "SautiYaAfya Support Request";
    const body = "Hello Admin,\n\nI am experiencing an issue with...";
    const email = "tolbert.okoth@moh.go.ke"; 
    window.location.href = `mailto:${email}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
  };

  return (
    <div className="container p-0" style={{ maxWidth: '600px' }}>
      <button className="btn btn-link text-dark-brown text-decoration-none mb-3 p-0 fw-bold" onClick={() => navigate(-1)}>
        <FaArrowLeft /> {t('back')}
      </button>

      <div className="glass-card p-5 text-center">
          {/* HEADER LOGO */}
          <div className="mb-4">
            <div className="bg-white rounded-circle d-inline-flex align-items-center justify-content-center shadow-sm" style={{width: '80px', height: '80px'}}>
                <FaHeartbeat className="text-accent display-4 animate-pulse" />
            </div>
            <h2 className="fw-bold mt-3 mb-0 text-dark-brown">SautiYaAfya</h2>
            <p className="text-dark-brown opacity-75 lead mb-2">"The Voice of Health"</p>
            <span className="badge bg-white text-dark border shadow-sm">
                {t('version')} 9.6.0 (Beta)
            </span>
          </div>

          <hr className="my-4" style={{borderColor: 'rgba(255,255,255,0.4)'}} />

          {/* INFO SECTION */}
          <div className="text-start px-md-4">
            <div className="d-flex align-items-center mb-3">
                <div className="me-3 text-accent"><FaBuilding size={20} /></div>
                <div>
                    <small className="text-muted fw-bold text-uppercase d-block" style={{fontSize: '0.7rem'}}>{t('developed_for')}</small>
                    <div className="fw-bold text-dark-brown">Ministry of Health (Kenya)</div>
                </div>
            </div>
            
            <div className="d-flex align-items-center mb-4">
                <div className="me-3 text-accent"><FaCode size={20} /></div>
                <div>
                    <small className="text-muted fw-bold text-uppercase d-block" style={{fontSize: '0.7rem'}}>{t('tech_lead')}</small>
                    <div className="fw-bold text-dark-brown">Tolbert Okoth</div>
                </div>
            </div>

            <div className="bg-white bg-opacity-50 p-3 rounded mb-4 border border-light">
                <small className="text-muted fw-bold text-uppercase mb-1 d-block" style={{fontSize: '0.7rem'}}>{t('purpose_label')}</small>
                <div className="text-dark-brown small" style={{lineHeight: '1.6'}}>
                    {t('purpose_text')}
                </div>
            </div>
          </div>

          {/* ACTION BUTTON */}
          <button onClick={handleContact} className="btn w-100 py-2 shadow-sm" 
            style={{
                background: 'rgba(255,255,255,0.8)', 
                color: '#2d3436',
                border: '1px solid #fff'
            }}>
            <FaEnvelope className="me-2 text-accent" /> {t('contact_admin')}
          </button>

          <div className="mt-4 text-dark-brown opacity-50 small">
            &copy; 2026 Republic of Kenya. All Rights Reserved.
          </div>
      </div>
    </div>
  );
};

export default AboutSettings;