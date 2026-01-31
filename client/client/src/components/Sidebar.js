/* client/src/components/Sidebar.js */
import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { 
  FaUserMd, FaChartPie, FaGlobeAfrica, FaCog, 
  FaSignOutAlt, FaBars, FaTimes, FaUserInjured,
  FaBook, FaUserShield 
} from 'react-icons/fa'; 
import './Sidebar.css'; 

const Sidebar = ({ role, onLogout }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggle = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);

  // 🔹 GHOST MODE STYLE
  const sidebarStyle = {
    background: 'rgba(0, 0, 0, 0.2)',
    borderRight: '1px solid rgba(255, 255, 255, 0.1)',
    backdropFilter: 'none', 
  };

  return (
    <>
      {/* 📱 HAMBURGER BUTTON */}
      <button className="mobile-toggle-btn text-white" onClick={toggle}>
        {isOpen ? <FaTimes /> : <FaBars />}
      </button>

      {/* 🌑 BACKDROP OVERLAY */}
      <div 
        className={`mobile-overlay ${isOpen ? 'open' : ''}`} 
        onClick={closeMenu}
      />

      {/* SIDEBAR CONTAINER */}
      <div className={`sidebar ${isOpen ? 'open' : ''}`} style={sidebarStyle}>
        
        {/* LOGO AREA */}
        <div className="d-flex align-items-center mb-4 px-2">
           <div style={{
             width:'35px', 
             height:'35px', 
             background:'#0984e3', 
             borderRadius:'8px', 
             display:'flex', 
             alignItems:'center', 
             justifyContent:'center', 
             marginRight:'10px', 
             color:'white'
           }}>
             {role === 'ADMIN' ? <FaUserShield /> : <FaUserMd />}
           </div>
           <div>
             <h5 className="mb-0 fw-bold text-white">SautiYaAfya</h5>
             <small className="text-white-50" style={{fontSize:'0.7rem'}}>
               {role === 'ADMIN' ? 'ADMIN PANEL' : 'AI TRIAGE SYSTEM'}
             </small>
           </div>
        </div>
        
        {/* NAV LINKS */}
        <div className="nav flex-column mb-auto">
          
          {/* 🔴 ADMIN VIEW: Dashboard Only (No Clinical Data) */}
          {role === 'ADMIN' && (
            <NavLink to="/admin" onClick={closeMenu} className="nav-link text-white-50" end>
              <FaUserShield className="me-3" /> Dashboard
            </NavLink>
          )}

          {/* 🔵 DOCTOR VIEW: Dashboard + Patients + Analytics */}
          {role === 'DOCTOR' && (
            <>
              <NavLink to="/doctor" onClick={closeMenu} className="nav-link text-white-50" end>
                <FaGlobeAfrica className="me-3" /> Dashboard
              </NavLink>

              <NavLink to="/doctor/patients" onClick={closeMenu} className="nav-link text-white-50">
                <FaUserInjured className="me-3" /> Patients List
              </NavLink>

              <NavLink to="/doctor/analytics" onClick={closeMenu} className="nav-link text-white-50">
                <FaChartPie className="me-3" /> Analytics
              </NavLink>
            </>
          )}

          {/* 🟢 CHW VIEW */}
          {role === 'CHW' && (
            <NavLink to="/chw" onClick={closeMenu} className="nav-link text-white-50" end>
              <FaUserMd className="me-3" /> New Screening
            </NavLink>
          )}
          
          <div className="my-2 border-top" style={{borderColor: 'rgba(255,255,255,0.1)'}}></div>

          <NavLink to="/guide" onClick={closeMenu} className="nav-link text-white-50">
            <FaBook className="me-3" /> User Guide
          </NavLink>

          <NavLink to="/settings" onClick={closeMenu} className="nav-link text-white-50">
            <FaCog className="me-3" /> Settings
          </NavLink>
        </div>
        
        {/* LOGOUT BUTTON */}
        <div className="mt-auto">
          <button 
            className="btn btn-outline-danger w-100 d-flex align-items-center justify-content-center py-2 shadow-sm text-white border-danger" 
            style={{borderRadius: '12px', borderWidth: '1px'}}
            onClick={onLogout}
          >
            <FaSignOutAlt className="me-2"/> Log out
          </button>
        </div>
      </div>
    </>
  );
};

export default Sidebar;