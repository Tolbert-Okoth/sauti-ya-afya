/* client/src/components/Layout.js */
import React from 'react';
import Sidebar from './Sidebar'; 
// import { useTranslation } from '../hooks/useTranslation'; // Uncomment if using translation

const Layout = ({ role, children, logout }) => {
  return (
    // 1. MAIN FLEX CONTAINER (The "Glass Window")
    <div className="glass-window d-flex">
      
      {/* SIDEBAR WRAPPER */}
      {/* We keep this simple wrapper so Sidebar.css handles the fixed/relative positioning */}
      <div className="sidebar-wrapper">
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
        
        {/* Content */}
        <div className="container-fluid p-3 p-md-4 pt-5 mt-4 mt-md-0">
          {children}
        </div>
      </main>

    </div>
  );
};

export default Layout;