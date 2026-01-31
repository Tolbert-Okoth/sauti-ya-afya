/* client/src/components/Layout.js */
import React from 'react';
import Sidebar from './Sidebar'; 
import Footer from './Footer'; 

const Layout = ({ role, children, logout }) => {
  return (
    // 1. MAIN FLEX CONTAINER (The "Glass Window")
    <div className="glass-window d-flex">
      
      {/* SIDEBAR WRAPPER */}
      <div className="sidebar-wrapper">
        <Sidebar role={role} onLogout={logout} />
      </div>

      {/* 2. MAIN CONTENT AREA */}
      <main className="flex-grow-1 mobile-content-wrapper d-flex flex-column" 
            style={{ 
              position: 'relative', 
              height: '100%', 
              overflowY: 'auto',
              width: '100%' 
            }}>
        
        {/* Content Wrapper - Pushes footer down */}
        <div className="container-fluid p-3 p-md-4 pt-5 mt-4 mt-md-0 flex-grow-1">
          {children}
        </div>

        {/* 🟢 FOOTER (Stays at bottom) */}
        <Footer />

      </main>

    </div>
  );
};

export default Layout;