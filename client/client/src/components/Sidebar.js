/* client/src/components/Sidebar.js */
import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import { 
  FaUserMd, FaChartPie, FaGlobeAfrica, FaCog, 
  FaSignOutAlt, FaBars, FaTimes, FaUserInjured 
} from 'react-icons/fa'; 
import './Sidebar.css'; // Import our new styles

const Sidebar = ({ role, onLogout }) => {
  const [isOpen, setIsOpen] = useState(false);

  const toggle = () => setIsOpen(!isOpen);
  const closeMenu = () => setIsOpen(false);

  return (
    <>
      {/* 📱 HAMBURGER BUTTON (Visible only on mobile via CSS) */}
      <button className="mobile-toggle-btn" onClick={toggle}>
        {isOpen ? <FaTimes /> : <FaBars />}
      </button>

      {/* 🌑 BACKDROP OVERLAY (Visible only when open on mobile) */}
      <div 
        className={`mobile-overlay ${isOpen ? 'open' : ''}`} 
        onClick={closeMenu}
      />

      {/* SIDEBAR CONTAINER */}
      {/* CSS handles the sliding via the 'open' class */}
      <div className={`sidebar ${isOpen ? 'open' : ''}`}>
        
        {/* LOGO AREA */}
        <div className="d-flex align-items-center mb-4 px-2">
           <div style={{width:'35px', height:'35px', background:'#0d6efd', borderRadius:'8px', display:'flex', alignItems:'center', justifyContent:'center', marginRight:'10px', color:'white'}}>
             <FaUserMd />
           </div>
           <div>
             <h5 className="mb-0 fw-bold" style={{color:'#2d3436'}}>SautiYaAfya</h5>
             <small className="text-muted" style={{fontSize:'0.7rem'}}>AI TRIAGE SYSTEM</small>
           </div>
        </div>
        
        {/* NAV LINKS */}
        <div className="nav flex-column mb-auto">
          {role === 'DOCTOR' || role === 'ADMIN' ? (
            <>
              <NavLink to="/doctor" onClick={closeMenu} className="nav-link" end>
                <FaGlobeAfrica className="me-3" /> Dashboard
              </NavLink>

              <NavLink to="/doctor/patients" onClick={closeMenu} className="nav-link">
                <FaUserInjured className="me-3" /> Patients List
              </NavLink>

              <NavLink to="/doctor/analytics" onClick={closeMenu} className="nav-link">
                <FaChartPie className="me-3" /> Analytics
              </NavLink>
            </>
          ) : (
            <NavLink to="/chw" onClick={closeMenu} className="nav-link" end>
              <FaUserMd className="me-3" /> New Screening
            </NavLink>
          )}
          
          <div className="my-2 border-top"></div>

          <NavLink to="/settings" onClick={closeMenu} className="nav-link">
            <FaCog className="me-3" /> Settings
          </NavLink>
        </div>
        
        {/* LOGOUT BUTTON */}
        <div className="mt-auto">
          <button 
            className="btn btn-outline-danger w-100 d-flex align-items-center justify-content-center py-2" 
            style={{borderRadius: '12px'}}
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