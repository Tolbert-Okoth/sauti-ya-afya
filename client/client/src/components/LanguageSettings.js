/* client/src/components/LanguageSettings.js */
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { FaArrowLeft, FaCheck, FaGlobeAfrica } from 'react-icons/fa';
import { useTranslation } from '../hooks/useTranslation';

const LanguageSettings = () => {
  const navigate = useNavigate();
  const { language, changeLanguage, t } = useTranslation(); 

  const handleLangChange = (newLang) => {
    changeLanguage(newLang);
  };

  return (
    <div className="container p-0" style={{ maxWidth: '600px' }}>
      <button className="btn btn-link text-dark-brown text-decoration-none mb-3 p-0 fw-bold" onClick={() => navigate(-1)}>
        <FaArrowLeft /> {t('back')}
      </button>

      <div className="d-flex align-items-center mb-4">
        <div className="bg-white rounded-circle p-2 text-accent me-3 shadow-sm">
            <FaGlobeAfrica size={24} />
        </div>
        <h4 className="fw-bold text-dark-brown mb-0">{t('language_title')}</h4>
      </div>

      <div className="glass-card p-2">
        <button 
            className={`w-100 border-0 p-3 mb-2 rounded d-flex justify-content-between align-items-center transition-all ${language === 'en' ? 'bg-white shadow-sm' : 'bg-transparent hover-glass'}`}
            onClick={() => handleLangChange('en')}
            style={{textAlign: 'left'}}
        >
            <div className="d-flex align-items-center">
                <span className="fs-2 me-3">🇬🇧</span>
                <div>
                    <span className="fw-bold text-dark-brown d-block">English (UK)</span>
                    <small className="text-muted">Default</small>
                </div>
            </div>
            {language === 'en' && <FaCheck className="text-accent" />}
        </button>
        
        <button 
            className={`w-100 border-0 p-3 rounded d-flex justify-content-between align-items-center transition-all ${language === 'sw' ? 'bg-white shadow-sm' : 'bg-transparent hover-glass'}`}
            onClick={() => handleLangChange('sw')}
            style={{textAlign: 'left'}}
        >
            <div className="d-flex align-items-center">
                <span className="fs-2 me-3">🇰🇪</span>
                <div>
                    <span className="fw-bold text-dark-brown d-block">Kiswahili</span>
                    <small className="text-muted">East Africa</small>
                </div>
            </div>
            {language === 'sw' && <FaCheck className="text-accent" />}
        </button>
      </div>
      
      <p className="text-dark-brown opacity-75 mt-3 small text-center px-4">
        {t('change_lang_note')}
      </p>
    </div>
  );
};

export default LanguageSettings;