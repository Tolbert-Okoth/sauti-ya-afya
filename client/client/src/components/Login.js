/* client/src/components/Login.js */
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { signInWithEmailAndPassword, signInWithPopup } from 'firebase/auth';
import { auth, googleProvider, resetPassword } from '../firebase';
import axios from 'axios';
import { 
  FaGoogle, FaStethoscope, FaChartLine, FaShieldAlt, 
  FaUserMd, FaLock, FaArrowRight, FaBrain 
} from 'react-icons/fa';
import config from '../config';

const Login = ({ setRole }) => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [msg, setMsg] = useState('');
  const [loading, setLoading] = useState(false);

  // Matches your transparent glass theme
  const glassInputStyle = {
    background: 'rgba(255, 255, 255, 0.45)', 
    border: '1px solid rgba(255, 255, 255, 0.6)',
    color: '#2d3436',
    borderRadius: '12px',
    padding: '12px',
    backdropFilter: 'blur(6px)'
  };

  const executeLogin = async (user) => {
    try {
      setLoading(true);
      const token = await user.getIdToken();

      const response = await axios.post(
        `${config.API_BASE_URL}/login`,
        {},
        { headers: { Authorization: `Bearer ${token}` } }
      );

      const { role } = response.data;
      if (setRole) setRole(role); // Guard check in case prop is missing
      navigate(role === 'CHW' ? '/chw' : '/doctor');
    } catch (err) {
      console.error("Login Error:", err);
      if (err.response?.status === 404) {
        setError('Account not found. Please Sign Up first.');
      } else {
        setError(err.response?.data?.message || 'Login Failed');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleEmailLogin = async (e) => {
    e.preventDefault();
    setError('');
    setMsg('');
    setLoading(true);
    try {
      const cred = await signInWithEmailAndPassword(auth, email, password);
      await executeLogin(cred.user);
    } catch (err) {
      console.error(err);
      setError('Invalid Email or Password.');
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setError('');
    setMsg('');
    try {
      const result = await signInWithPopup(auth, googleProvider);
      setLoading(true);
      await executeLogin(result.user);
    } catch (err) {
      if (err.code === 'auth/popup-blocked') {
        setError('Popup blocked! Please allow popups for this site.');
      } else if (err.code === 'auth/popup-closed-by-user') {
        setError('Login cancelled.');
      } else {
        setError('Google Login Failed.');
      }
      setLoading(false);
    }
  };

  const handleForgotPassword = async () => {
    if (!email) return setError('Please enter your email first.');
    try {
      await resetPassword(email);
      setMsg('Password reset email sent!');
      setError('');
    } catch (err) {
      setError('Error sending reset email.');
    }
  };

  return (
    <div className="container-fluid vh-100 d-flex align-items-center justify-content-center p-0">
      
      {/* ✨ HERO CARD: Split Layout (Glass + Dark Brand Panel) */}
      <div 
        className="d-flex flex-column flex-md-row overflow-hidden shadow-lg animate-slide-in"
        style={{ 
          maxWidth: '1100px', 
          width: '90%', 
          minHeight: '650px',
          background: 'rgba(255, 255, 255, 0.95)', // Solid background for readability
          borderRadius: '24px',
          boxShadow: '0 20px 40px rgba(0,0,0,0.4)'
        }}
      >
        
        {/* 👈 LEFT PANEL: Modern Health Teal Gradient */}
        <div className="col-12 col-md-5 p-5 d-flex flex-column justify-content-center text-white position-relative overflow-hidden" 
             style={{
               /* 🟢 UPDATED GRADIENT: Option 1 (Teal) */
               background: 'linear-gradient(135deg, #134E5E 0%, #71B280 100%)', 
             }}>
          
          {/* Decorative Background Circles */}
          <div style={{position: 'absolute', top: '-50px', left: '-50px', width: '200px', height: '200px', background: 'rgba(255,255,255,0.05)', borderRadius: '50%'}}></div>
          <div style={{position: 'absolute', bottom: '-20px', right: '-20px', width: '150px', height: '150px', background: 'rgba(255,255,255,0.05)', borderRadius: '50%'}}></div>

          <div className="mb-5 position-relative z-1">
            <div className="d-inline-flex align-items-center justify-content-center bg-white text-dark rounded-4 mb-4 shadow-lg" style={{width: '64px', height: '64px'}}>
              <FaStethoscope size={32} className="text-primary" />
            </div>
            <h1 className="fw-bold display-5 mb-2">SautiYaAfya</h1>
            <p className="lead fw-light opacity-75 mb-0">The "Two-Brain" AI Triage System</p>
          </div>

          <div className="d-flex flex-column gap-4 position-relative z-1">
            <div className="d-flex">
              <div className="me-3"><FaBrain className="fs-4 text-white" /></div>
              <div>
                <h6 className="fw-bold mb-1">Dual-Engine Intelligence</h6>
                <p className="small opacity-75 mb-0">Combining ResNet50 deep learning with clinical logic.</p>
              </div>
            </div>
            <div className="d-flex">
              <div className="me-3"><FaChartLine className="fs-4 text-white" /></div>
              <div>
                <h6 className="fw-bold mb-1">98% Diagnostic Accuracy</h6>
                <p className="small opacity-75 mb-0">Precision audio analysis for Pneumonia & Asthma.</p>
              </div>
            </div>
            <div className="d-flex">
              <div className="me-3"><FaShieldAlt className="fs-4 text-warning" /></div>
              <div>
                <h6 className="fw-bold mb-1">Secure & Encrypted</h6>
                <p className="small opacity-75 mb-0">Patient data is protected by enterprise-grade security.</p>
              </div>
            </div>
          </div>

          <div className="mt-auto pt-5 opacity-50 small">
            &copy; 2026 SautiYaAfya Research • Defense Build v2.0
          </div>
        </div>

        {/* 👉 RIGHT PANEL: Login Form */}
        <div className="col-12 col-md-7 p-5 d-flex align-items-center bg-white">
          <div className="w-100 px-md-4">
            <div className="text-center mb-5">
              <h3 className="fw-bold text-dark-brown mb-1">Welcome Back</h3>
              <p className="text-muted">Enter your credentials to access the secure portal.</p>
            </div>

            {error && <div className="alert alert-danger text-center shadow-sm py-2 small border-0 bg-danger text-white mb-4">{error}</div>}
            {msg && <div className="alert alert-success text-center shadow-sm py-2 small border-0 bg-success text-white mb-4">{msg}</div>}

            <form onSubmit={handleEmailLogin}>
              <div className="mb-4">
                <label className="form-label small fw-bold text-muted text-uppercase">Email Address</label>
                <div className="input-group">
                  <span className="input-group-text border-0 bg-light"><FaUserMd className="text-muted"/></span>
                  <input 
                    type="email" 
                    className="form-control bg-light border-0 py-3" 
                    placeholder="doctor@hospital.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required 
                    style={{boxShadow: 'none'}}
                  />
                </div>
              </div>

              <div className="mb-2">
                <label className="form-label small fw-bold text-muted text-uppercase">Password</label>
                <div className="input-group">
                  <span className="input-group-text border-0 bg-light"><FaLock className="text-muted"/></span>
                  <input 
                    type="password" 
                    className="form-control bg-light border-0 py-3" 
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required 
                    style={{boxShadow: 'none'}}
                  />
                </div>
              </div>

              <div className="text-end mb-4">
                 <button type="button" onClick={handleForgotPassword} className="btn btn-link btn-sm p-0 text-muted fw-bold" style={{textDecoration: 'none', fontSize: '0.85rem'}}>
                   Forgot Password?
                 </button>
              </div>

              <button 
                type="submit" 
                className="btn btn-primary w-100 py-3 rounded-3 fw-bold shadow-lg d-flex align-items-center justify-content-center mb-3 transition-transform"
                disabled={loading}
                style={{background: 'linear-gradient(45deg, #0984e3, #00b894)', border: 'none'}}
              >
                {loading ? 'Authenticating...' : <>Access Dashboard <FaArrowRight className="ms-2"/></>}
              </button>
            </form>

            <div className="d-flex align-items-center mb-4">
                <hr className="flex-grow-1" style={{borderColor: '#eee'}}/>
                <span className="mx-3 small text-muted fw-bold">OR</span>
                <hr className="flex-grow-1" style={{borderColor: '#eee'}}/>
            </div>

            <button
               onClick={handleGoogleLogin}
               className="btn w-100 py-3 rounded-3 shadow-sm d-flex align-items-center justify-content-center mb-4 border"
               disabled={loading}
               style={{background: '#fff', color: '#444'}}
            >
               <FaGoogle className="me-2 text-danger" />
               Sign in with Google
            </button>

            <div className="text-center">
              <small className="text-muted">
                Don't have an account? <Link to="/signup" className="fw-bold text-primary" style={{textDecoration: 'none'}}>Register Now</Link>
              </small>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Login;