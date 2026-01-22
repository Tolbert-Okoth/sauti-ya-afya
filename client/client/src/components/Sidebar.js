/* client/src/components/Sidebar.js */
import React, { useState, useEffect } from 'react';
import { NavLink } from 'react-router-dom';
import { 
  FaUserMd, FaChartPie, FaGlobeAfrica, FaCog, 
  FaSignOutAlt, FaBars, FaTimes, FaUserInjured 
} from 'react-icons/fa'; // 👈 Added FaUserInjured icon

const Sidebar = ({ role, onLogout }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  // 1. LISTEN TO SCREEN RESIZE
  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const toggle = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);

  // 2. DYNAMIC STYLES
  const sidebarStyle = {
    width: '260px',
    height: '100%', 
    backgroundColor: 'white', 
    display: 'flex',
    flexDirection: 'column',
    padding: '1rem',
    borderRight: '1px solid rgba(0,0,0,0.1)', 
    transition: 'all 0.3s ease-in-out',
    zIndex: 1000,
    overflowY: 'auto', 
    
    // MOBILE OVERRIDES:
    position: isMobile ? 'fixed' : 'relative',
    height: isMobile ? '100vh' : '100%', 
    left: isMobile ? (isOpen ? '0' : '-100%') : '0', 
    top: 0,
    boxShadow: isMobile ? '2px 0 10px rgba(0,0,0,0.1)' : 'none'
  };

  const toggleBtnStyle = {
    display: isMobile ? 'flex' : 'none', 
    position: 'fixed',
    top: '15px',
    right: '15px',
    zIndex: 1100,
    width: '45px',
    height: '45px',
    borderRadius: '50%',
    alignItems: 'center',
    justifyContent: 'center',
    border: '1px solid #ddd',
    background: 'white',
    color: '#333',
    boxShadow: '0 2px 5px rgba(0,0,0,0.2)'
  };

  const backdropStyle = {
    position: 'fixed',
    top: 0, left: 0, width: '100vw', height: '100vh',
    background: 'rgba(0,0,0,0.5)',
    zIndex: 900,
    display: (isMobile && isOpen) ? 'block' : 'none',
    backdropFilter: 'blur(2px)'
  };

  return (
    <>
      {/* 📱 HAMBURGER BUTTON */}
      <button style={toggleBtnStyle} onClick={toggle}>
        {isOpen ? <FaTimes /> : <FaBars />}
      </button>

      {/* 🌑 BACKDROP */}
      <div style={backdropStyle} onClick={closeMenu}></div>

      {/* SIDEBAR CONTAINER */}
      <div style={sidebarStyle} className="sidebar-manual-container">
        
        {/* LOGO */}
        <a href="/" className="d-flex align-items-center mb-3 text-decoration-none">
          <span className="fs-4 fw-bold text-primary">SautiYaAfya</span>
        </a>
        <hr />
        
        {/* NAV LINKS */}
        <ul className="nav nav-pills flex-column mb-auto">
          {role === 'DOCTOR' || role === 'ADMIN' ? (
            <>
              {/* 1. DASHBOARD */}
              <li className="nav-item">
                <NavLink to="/doctor" onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? 'active' : 'link-dark'}`} end>
                  <FaGlobeAfrica className="me-2" /> Dashboard
                </NavLink>
              </li>

              {/* 2. PATIENTS LIST (NEW BUTTON) */}
              <li>
                <NavLink to="/doctor/patients" onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? 'active' : 'link-dark'}`}>
                  <FaUserInjured className="me-2" /> Patients List
                </NavLink>
              </li>

              {/* 3. ANALYTICS */}
              <li>
                <NavLink to="/doctor/analytics" onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? 'active' : 'link-dark'}`}>
                  <FaChartPie className="me-2" /> Analytics
                </NavLink>
              </li>
            </>
          ) : (
            <li className="nav-item">
              <NavLink to="/chw" onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? 'active' : 'link-dark'}`} end>
                <FaUserMd className="me-2" /> New Screening
              </NavLink>
            </li>
          )}
          
          <li className="mt-3">
            <NavLink to="/settings" onClick={closeMenu} className={({ isActive }) => `nav-link ${isActive ? 'active' : 'link-dark'}`}>
              <FaCog className="me-2" /> Settings
            </NavLink>
          </li>
        </ul>
        
        <hr />
        
        {/* LOGOUT BUTTON */}
        <div className="dropdown mt-auto">
          <button className="btn btn-outline-danger w-100 d-flex align-items-center justify-content-center" onClick={onLogout}>
            <FaSignOutAlt className="me-2"/> Log out
          </button>
        </div>
      </div>
    </>
  );
};

export default Sidebar;