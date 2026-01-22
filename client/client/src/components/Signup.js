/* client/src/components/Signup.js */
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { createUserWithEmailAndPassword, signInWithPopup } from 'firebase/auth';
import { auth, googleProvider } from '../firebase';
import axios from 'axios';
import { FaGoogle, FaUserPlus } from 'react-icons/fa';
import config from '../config'; // 👈 IMPORT CONFIGURATION

const Signup = ({ setRole }) => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: '', password: '', role: 'CHW', county_id: 1
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Custom Glass Input Style
  const glassInputStyle = {
      background: 'rgba(255,255,255,0.3)',
      border: '1px solid rgba(255,255,255,0.4)',
      color: '#2d3436',
      backdropFilter: 'blur(10px)',
      borderRadius: '12px',
      padding: '10px 15px'
  };

  const registerInBackend = async (user) => {
    const token = await user.getIdToken();
    
    // ✅ FIX: Use dynamic config.API_BASE_URL instead of localhost
    await axios.post(`${config.API_BASE_URL}/register`, {
      role: formData.role,
      county_id: formData.county_id
    }, {
      headers: { Authorization: `Bearer ${token}` }
    });
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const userCredential = await createUserWithEmailAndPassword(auth, formData.email, formData.password);
      await registerInBackend(userCredential.user);
      setRole(formData.role);
      navigate(formData.role === 'CHW' ? '/chw' : '/doctor');
    } catch (err) {
      setError(err.response?.data?.message || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignup = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      await registerInBackend(result.user);
      setRole(formData.role);
      navigate(formData.role === 'CHW' ? '/chw' : '/doctor');
    } catch (err) {
      setError("Google Signup Failed: " + (err.response?.data?.message || err.message));
    }
  };

  return (
    <div className="min-vh-100 d-flex align-items-center justify-content-center" 
         style={{
             background: 'radial-gradient(circle at 10% 20%, #bcaaa4 0%, #8d6e63 40%, #5d4037 90%)'
         }}>
      
      <div className="glass-card p-5 shadow-lg position-relative overflow-hidden" style={{ maxWidth: '450px', width: '90%', borderRadius: '24px' }}>
        
        <div className="text-center mb-4">
            <div className="bg-white rounded-circle d-inline-flex align-items-center justify-content-center shadow-sm mb-3" style={{width: '50px', height: '50px'}}>
                <FaUserPlus className="text-accent fs-4" />
            </div>
            <h4 className="fw-bold text-dark-brown">Create Account</h4>
            <p className="text-dark-brown opacity-75 small">Join the SautiYaAfya Network</p>
        </div>
        
        {error && <div className="alert alert-danger small border-0 text-white shadow-sm" style={{background: 'rgba(220, 53, 69, 0.9)'}}>{error}</div>}

        <form onSubmit={handleSignup}>
          <div className="mb-3">
            <label className="form-label small fw-bold text-dark-brown opacity-75 ms-1">Email</label>
            <input type="email" className="form-control" required style={glassInputStyle}
              onChange={(e) => setFormData({...formData, email: e.target.value})} />
          </div>
          <div className="mb-3">
            <label className="form-label small fw-bold text-dark-brown opacity-75 ms-1">Password</label>
            <input type="password" className="form-control" required style={glassInputStyle}
              onChange={(e) => setFormData({...formData, password: e.target.value})} />
          </div>
          
          <div className="row mb-4">
            <div className="col-6">
               <label className="form-label small fw-bold text-dark-brown opacity-75 ms-1">Role</label>
               <select className="form-select" style={glassInputStyle}
                 onChange={(e) => setFormData({...formData, role: e.target.value})}>
                 <option value="CHW">CHW</option>
                 <option value="DOCTOR">Doctor</option>
               </select>
            </div>
            <div className="col-6">
               <label className="form-label small fw-bold text-dark-brown opacity-75 ms-1">County ID</label>
               <input type="number" className="form-control" value={formData.county_id} style={glassInputStyle}
                 onChange={(e) => setFormData({...formData, county_id: e.target.value})} />
            </div>
          </div>

          <button type="submit" className="btn btn-dark w-100 py-2 rounded-pill shadow-sm fw-bold mb-3" disabled={loading} style={{background: '#3e2723', border: 'none'}}>
            {loading ? "Creating Account..." : "Sign Up"}
          </button>
        </form>

        <div className="d-flex align-items-center my-3">
            <div className="flex-grow-1 border-bottom border-light opacity-50"></div>
            <span className="mx-3 text-dark-brown opacity-50 small">OR</span>
            <div className="flex-grow-1 border-bottom border-light opacity-50"></div>
        </div>

        <button onClick={handleGoogleSignup} className="btn bg-white w-100 py-2 rounded-pill shadow-sm text-dark fw-bold mb-4">
          <FaGoogle className="me-2 text-danger"/> Sign up with Google
        </button>

        <div className="text-center">
          <small className="text-dark-brown opacity-75">Already have an account? <Link to="/" className="fw-bold text-dark-brown">Login here</Link></small>
        </div>
      </div>
    </div>
  );
};

export default Signup;