/* client/src/components/SettingsHub.js */
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from '../hooks/useTranslation';
import { 
  FaUser, FaMicrophone, FaSync, FaBell, FaShieldAlt, FaGlobe, FaInfoCircle, FaChevronRight, FaRobot, FaCog 
} from 'react-icons/fa';

const SettingsHub = ({ role }) => {
  const navigate = useNavigate();
  const { t } = useTranslation();

  // Helper component for list items
  const SettingItem = ({ icon, title, path, restrictedTo, color }) => {
    if (restrictedTo && !restrictedTo.includes(role)) return null;
    
    return (
      <div 
        className="d-flex justify-content-between align-items-center p-3 transition-all cursor-pointer hover-glass"
        onClick={() => navigate(path)}
        style={{ 
            cursor: 'pointer',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)' // 🟢 Subtle Glass Border
        }}
      >
        <div className="d-flex align-items-center">
          <div className={`me-3 text-${color || 'secondary'}`}>{icon}</div>
          {/* 🟢 Text White for Dark Mode */}
          <span className="fw-bold text-white">{title}</span>
        </div>
        <FaChevronRight className="text-white opacity-25" />
      </div>
    );
  };

  return (
    <div className="container p-0" style={{ maxWidth: '600px' }}>
      
      {/* Header */}
      <div className="d-flex align-items-center mb-4">
        {/* 🟢 Icon Container: Primary Blue instead of White */}
        <div className="bg-primary rounded-circle p-2 text-white me-3 shadow-sm">
            <FaCog size={20} />
        </div>
        {/* 🟢 Title: White */}
        <h3 className="fw-bold text-white mb-0">{t('settings_title')}</h3>
      </div>
      
      <div className="glass-card overflow-hidden p-0">
        
        {/* 1. PROFILE */}
        <SettingItem icon={<FaUser />} title={t('menu_profile')} path="/settings/profile" color="primary" />
        
        {/* 2. DEVICE (CHW Only) */}
        <SettingItem icon={<FaMicrophone />} title={t('menu_device')} path="/settings/device" restrictedTo={['CHW']} color="danger" />
        
        {/* 3. SYNC (CHW/Doc) */}
        <SettingItem icon={<FaSync />} title={t('menu_sync')} path="/settings/sync" restrictedTo={['CHW', 'DOCTOR']} color="info" />

        {/* 4. ALERTS (Doc Only) */}
        <SettingItem icon={<FaBell />} title={t('menu_alerts')} path="/settings/notifications" restrictedTo={['DOCTOR']} color="warning" />

        {/* 5. ADMIN CONFIG (Admin Only) */}
        <SettingItem icon={<FaRobot />} title={t('menu_admin_config')} path="/settings/admin-config" restrictedTo={['ADMIN']} color="danger" />

        {/* 6. GENERAL */}
        <SettingItem icon={<FaShieldAlt />} title={t('menu_privacy')} path="/settings/privacy" color="success" />
        <SettingItem icon={<FaGlobe />} title={t('menu_language')} path="/settings/language" color="primary" />
        <SettingItem icon={<FaInfoCircle />} title={t('menu_about')} path="/settings/about" color="secondary" />
        
      </div>

      <div className="text-center mt-4 text-white opacity-50 small">
        SautiYaAfya v9.6.0 (Beta)
      </div>
    </div>
  );
};

export default SettingsHub;