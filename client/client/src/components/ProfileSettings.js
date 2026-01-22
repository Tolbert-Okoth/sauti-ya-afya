/* client/src/components/ProfileSettings.js */
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { auth } from '../firebase';
import { FaArrowLeft, FaUserCircle, FaIdBadge, FaMapMarkerAlt, FaUserMd } from 'react-icons/fa';

const ProfileSettings = ({ role }) => {
  const navigate = useNavigate();
  const user = auth.currentUser;

  return (
    <div className="container p-0" style={{ maxWidth: '600px' }}>
      <button className="btn btn-link text-dark-brown text-decoration-none mb-3 p-0 fw-bold" onClick={() => navigate(-1)}>
        <FaArrowLeft /> Back
      </button>

      <div className="glass-card text-center p-5 mb-4 position-relative overflow-hidden">
        {/* Decorative background circle */}
        <div className="position-absolute bg-white opacity-25 rounded-circle" style={{width: '200px', height: '200px', top: '-100px', left: '50%', transform: 'translateX(-50%)'}}></div>
        
        <div className="position-relative">
            <div className="mb-3 d-inline-block p-1 bg-white rounded-circle shadow-sm">
                 <div className="bg-light rounded-circle p-4 text-secondary">
                    <FaUserCircle size={80} />
                 </div>
            </div>
            
            <h4 className="fw-bold text-dark-brown mb-1">{user?.email}</h4>
            
            <div className="mt-2">
                <span className={`badge rounded-pill shadow-sm px-3 py-2 ${role === 'DOCTOR' ? 'bg-primary' : role === 'ADMIN' ? 'bg-dark' : 'bg-success'}`}>
                    {role === 'DOCTOR' && <FaUserMd className="me-2"/>}
                    {role} ACCOUNT
                </span>
            </div>
        </div>
      </div>

      <div className="glass-card p-4">
        <h6 className="fw-bold text-dark-brown mb-3 border-bottom border-light pb-2">System Identification</h6>
        
        <div className="d-flex align-items-center mb-4">
            <div className="bg-white rounded-circle p-2 me-3 text-accent shadow-sm">
                <FaIdBadge size={20}/>
            </div>
            <div>
                <label className="small text-muted d-block text-uppercase fw-bold" style={{fontSize: '0.7rem'}}>Unique ID (UID)</label>
                <span className="fw-bold text-dark-brown font-monospace">{user?.uid}</span>
            </div>
        </div>
        
        <div className="d-flex align-items-center">
            <div className="bg-white rounded-circle p-2 me-3 text-danger shadow-sm">
                <FaMapMarkerAlt size={20}/>
            </div>
            <div>
                <label className="small text-muted d-block text-uppercase fw-bold" style={{fontSize: '0.7rem'}}>Assigned Jurisdiction</label>
                <span className="fw-bold text-dark-brown">Turkana County (Default)</span>
            </div>
        </div>
      </div>
    </div>
  );
};

export default ProfileSettings;