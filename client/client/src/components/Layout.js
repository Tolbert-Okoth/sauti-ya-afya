/* client/src/components/Layout.js */
import React from 'react';
import Sidebar from './Sidebar'; 
import { useTranslation } from '../hooks/useTranslation';

const Layout = ({ role, children, logout }) => {
  const { t } = useTranslation();

  return (
    // 1. MAIN FLEX CONTAINER
    <div className="glass-window" style={{ display: 'flex', flexDirection: 'row', overflow: 'hidden', alignItems: 'stretch' }}>
      
      {/* 🛑 CRITICAL FIX: Sidebar Isolation Wrapper 
          We wrap the Sidebar in a real <div>. This creates a stable DOM node 
          so React doesn't crash when toggling the mobile menu elements. 
      */}
      <div className="sidebar-wrapper" style={{ height: '100%', flexShrink: 0 }}>
        <Sidebar role={role} onLogout={logout} />
      </div>

      {/* 2. MAIN CONTENT AREA */}
      <main className="flex-grow-1 mobile-content-wrapper" 
            style={{ 
              position: 'relative', 
              height: '100%', 
              overflowY: 'auto',
              width: '100%' 
            }}>
        
        {/* Header */}
        <div className="d-flex justify-content-between align-items-center mb-4 pt-5 pt-md-0 px-4 mt-4">
           <div>
              <h2 className="fw-bold text-dark-brown mb-0">Dashboard</h2>
              <p className="text-muted mb-0">Welcome back, {role ? role.toLowerCase() : 'User'}.</p>
           </div>
        </div>

        {/* Content */}
        <div className="container-fluid p-4">
          {children}
        </div>
      </main>

    </div>
  );
};

export default Layout;