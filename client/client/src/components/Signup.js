/* client/src/components/Signup.js */
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { createUserWithEmailAndPassword, signInWithPopup } from 'firebase/auth';
import { auth, googleProvider } from '../firebase';
import axios from 'axios';
import { 
  FaGoogle, FaUserPlus, FaUserMd, FaLock, 
  FaArrowRight, FaStethoscope, FaBrain, FaChartLine, FaShieldAlt
} from 'react-icons/fa';
import config from '../config';

const Signup = ({ setRole }) => {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: ''
  });

  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;
  const strongPasswordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;

  // 🔹 DARK GLASS INPUT STYLE
  const glassInputStyle = {
      background: 'rgba(0, 0, 0, 0.2)', 
      border: '1px solid rgba(255, 255, 255, 0.1)',
      color: '#fff',
      padding: '12px',
      borderRadius: '8px'
  };

  const registerInBackend = async (user) => {
    const token = await user.getIdToken();
    const response = await axios.post(
      `${config.API_BASE_URL}/register`,
      {},
      { headers: { Authorization: `Bearer ${token}` } }
    );
    // Auto-set role after registration
    if (response.data.role && setRole) {
        setRole(response.data.role);
    }
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');

    if (!emailRegex.test(formData.email)) {
      return setError('Please enter a valid email address.');
    }
    if (!strongPasswordRegex.test(formData.password)) {
      return setError('Password must be at least 8 chars (A-Z, a-z, 0-9, special char).');
    }
    if (formData.password !== formData.confirmPassword) {
      return setError('Passwords do not match.');
    }

    setLoading(true);

    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        formData.email,
        formData.password
      );

      await registerInBackend(userCredential.user);
      // Navigate based on default role (usually CHW for new signups unless admin changes it)
      navigate('/chw'); 
    } catch (err) {
      console.error(err);
      setError(err.response?.data?.message || err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGoogleSignup = async () => {
    try {
      const result = await signInWithPopup(auth, googleProvider);
      await registerInBackend(result.user);
      navigate('/chw');
    } catch (err) {
      setError('Google signup failed.');
    }
  };

  return (
    // 🟢 FIX: Use 'min-vh-100' instead of 'vh-100' to allow scrolling on small screens
    <div className="container-fluid min-vh-100 d-flex align-items-center justify-content-center p-4">
      
      {/* ✨ HERO CARD: Split Layout */}
      <div 
        className="d-flex flex-column flex-md-row overflow-hidden shadow-lg animate-slide-in"
        style={{ 
          maxWidth: '1100px', 
          width: '100%', 
          minHeight: '650px',
          borderRadius: '24px',
          boxShadow: '0 20px 40px rgba(0,0,0,0.6)'
        }}
      >
        
        {/* 👈 LEFT PANEL: "Clinical Navy" */}
        <div className="col-12 col-md-5 p-5 d-flex flex-column justify-content-center text-white position-relative overflow-hidden" 
             style={{
               background: 'linear-gradient(160deg, #1a2b3c 0%, #2c3e50 100%)'
             }}>
          
          <div style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', background: 'url("https://www.transparenttextures.com/patterns/cubes.png")', opacity: 0.05}}></div>

          <div className="mb-5 position-relative z-1">
            <div className="d-inline-flex align-items-center justify-content-center bg-white text-dark rounded-3 mb-4 shadow-sm" style={{width: '64px', height: '64px'}}>
              <FaUserPlus size={28} className="text-dark" />
            </div>
            <h1 className="fw-bold display-5 mb-2">Join SautiYaAfya</h1>
            <p className="lead fw-light opacity-75 mb-0">AI-Powered Pediatric Care</p>
          </div>

          <div className="d-flex flex-column gap-5 position-relative z-1">
            <div className="d-flex">
              <div className="me-3"><FaStethoscope className="fs-4 text-white opacity-75" /></div>
              <div>
                <h6 className="fw-bold mb-1">For Medical Professionals</h6>
                <p className="small opacity-75 mb-0">Designed for CHWs, Nurses, and Doctors to streamline respiratory triage.</p>
              </div>
            </div>

            <div className="d-flex">
              <div className="me-3"><FaChartLine className="fs-4 text-white opacity-75" /></div>
              <div>
                <h6 className="fw-bold mb-1">Instant Analysis</h6>
                <p className="small opacity-75 mb-0">Get real-time preliminary assessments for pneumonia and asthma.</p>
              </div>
            </div>
          </div>

          <div className="mt-auto pt-5 opacity-50 small">
            &copy; 2026 SautiYaAfya Research • System v2.0.4
          </div>
        </div>

        {/* 👉 RIGHT PANEL: Signup Form */}
        <div className="col-12 col-md-7 p-5 d-flex align-items-center" style={{background: '#1e272e'}}>
          <div className="w-100 px-md-4">
            <div className="text-center mb-5">
              <h3 className="fw-bold text-white mb-1">Create Account</h3>
              <p className="text-white-50">Register to access the diagnostic tools.</p>
            </div>

            {error && <div className="alert alert-danger text-center shadow-sm py-2 small border-0 bg-danger text-white mb-4">{error}</div>}

            <form onSubmit={handleSignup}>
              <div className="mb-3">
                <label className="form-label small fw-bold text-white-50 text-uppercase">Email Address</label>
                <div className="input-group">
                  <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.1)', color: '#ccc'}}><FaUserMd /></span>
                  <input 
                    type="email" 
                    className="form-control border-0" 
                    placeholder="doctor@hospital.com"
                    value={formData.email}
                    onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                    required 
                    style={glassInputStyle}
                  />
                </div>
              </div>

              <div className="mb-3">
                <label className="form-label small fw-bold text-white-50 text-uppercase">Password</label>
                <div className="input-group">
                  <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.1)', color: '#ccc'}}><FaLock /></span>
                  <input 
                    type="password" 
                    className="form-control border-0" 
                    placeholder="••••••••"
                    value={formData.password}
                    onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                    required 
                    style={glassInputStyle}
                  />
                </div>
                <small className="text-white-50" style={{fontSize: '0.75rem'}}>
                   Must include: 8+ chars, uppercase, number, symbol.
                </small>
              </div>

              <div className="mb-4">
                <label className="form-label small fw-bold text-white-50 text-uppercase">Confirm Password</label>
                <div className="input-group">
                  <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.1)', color: '#ccc'}}><FaLock /></span>
                  <input 
                    type="password" 
                    className="form-control border-0" 
                    placeholder="••••••••"
                    value={formData.confirmPassword}
                    onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                    required 
                    style={glassInputStyle}
                  />
                </div>
              </div>

              <button 
                type="submit" 
                className="btn w-100 py-3 rounded-3 fw-bold shadow-lg d-flex align-items-center justify-content-center mb-3 transition-transform"
                disabled={loading}
                style={{background: '#0984e3', border: 'none', color: '#fff'}}
              >
                {loading ? 'Registering...' : <>Create Account <FaArrowRight className="ms-2"/></>}
              </button>
            </form>

            <div className="d-flex align-items-center mb-4">
                <hr className="flex-grow-1" style={{borderColor: 'rgba(255,255,255,0.1)'}}/>
                <span className="mx-3 small text-white-50 fw-bold">OR</span>
                <hr className="flex-grow-1" style={{borderColor: 'rgba(255,255,255,0.1)'}}/>
            </div>

            <button
               onClick={handleGoogleSignup}
               className="btn w-100 py-3 rounded-3 shadow-sm d-flex align-items-center justify-content-center mb-4 border border-secondary"
               disabled={loading}
               style={{background: 'transparent', color: '#fff'}}
            >
               <FaGoogle className="me-2 text-danger" />
               Sign up with Google
            </button>

            <div className="text-center">
              <small className="text-white-50">
                Already have an account? <Link to="/" className="fw-bold text-white" style={{textDecoration: 'none'}}>Login Here</Link>
              </small>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Signup;