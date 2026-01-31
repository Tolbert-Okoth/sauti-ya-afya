/* client/src/components/Login.js */
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { signInWithEmailAndPassword, signInWithPopup } from 'firebase/auth';
import { auth, googleProvider, resetPassword } from '../firebase';
import axios from 'axios';
import { 
  FaGoogle, FaStethoscope, FaShieldAlt, 
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

  // 🔹 DARK GLASS INPUT STYLE
  const glassInputStyle = {
      background: 'rgba(0, 0, 0, 0.2)', 
      border: '1px solid rgba(255, 255, 255, 0.1)',
      color: '#fff',
      padding: '12px',
      borderRadius: '8px'
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
      if (setRole) setRole(role); 
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
      
      {/* ✨ HERO CARD: Split Layout (Dark Mode) */}
      <div 
        className="d-flex flex-column flex-md-row overflow-hidden shadow-lg animate-slide-in"
        style={{ 
          maxWidth: '1000px', 
          width: '90%', 
          minHeight: '600px', // Reduced height to fit screens better
          borderRadius: '24px',
          boxShadow: '0 20px 40px rgba(0,0,0,0.6)'
        }}
      >
        
        {/* 👈 LEFT PANEL: "Clinical Navy" Gradient */}
        <div className="col-12 col-md-5 p-5 d-flex flex-column justify-content-center text-white position-relative overflow-hidden" 
             style={{
               background: 'linear-gradient(160deg, #1a2b3c 0%, #2c3e50 100%)'
             }}>
          
          <div style={{position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', background: 'url("https://www.transparenttextures.com/patterns/cubes.png")', opacity: 0.05}}></div>

          <div className="mb-5 position-relative z-1">
            <div className="d-inline-flex align-items-center justify-content-center bg-white text-dark rounded-3 mb-4 shadow-sm" style={{width: '64px', height: '64px'}}>
              <FaStethoscope size={28} className="text-dark" />
            </div>
            <h1 className="fw-bold display-5 mb-2">SautiYaAfya</h1>
            <p className="lead fw-light opacity-75 mb-0">Pediatric Respiratory Triage</p>
          </div>

          <div className="d-flex flex-column gap-5 position-relative z-1">
            <div className="d-flex">
              <div className="me-3"><FaBrain className="fs-4 text-white opacity-75" /></div>
              <div>
                <h6 className="fw-bold mb-1">Clinical Decision Support</h6>
                <p className="small opacity-75 mb-0">Assists clinicians by analyzing lung sounds using standard medical algorithms.</p>
              </div>
            </div>

            <div className="d-flex">
              <div className="me-3"><FaShieldAlt className="fs-4 text-white opacity-75" /></div>
              <div>
                <h6 className="fw-bold mb-1">HIPAA Compliant Architecture</h6>
                <p className="small opacity-75 mb-0">End-to-end encryption ensures patient data remains private and secure.</p>
              </div>
            </div>
          </div>

          <div className="mt-auto pt-5 opacity-50 small">
            &copy; 2026 SautiYaAfya Research • System v2.0.4
          </div>
        </div>

        {/* 👉 RIGHT PANEL: Login Form (Dark Slate Background) */}
        <div className="col-12 col-md-7 p-5 d-flex align-items-center" style={{background: '#1e272e'}}>
          <div className="w-100 px-md-4">
            <div className="text-center mb-5">
              <h3 className="fw-bold text-white mb-1">Authorized Access</h3>
              <p className="text-white-50">Please authenticate to continue.</p>
            </div>

            {error && <div className="alert alert-danger text-center shadow-sm py-2 small border-0 bg-danger text-white mb-4">{error}</div>}
            {msg && <div className="alert alert-success text-center shadow-sm py-2 small border-0 bg-success text-white mb-4">{msg}</div>}

            <form onSubmit={handleEmailLogin}>
              <div className="mb-4">
                <label className="form-label small fw-bold text-white-50 text-uppercase">Email Address</label>
                <div className="input-group">
                  <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.1)', color: '#ccc'}}><FaUserMd /></span>
                  <input 
                    type="email" 
                    className="form-control border-0" 
                    placeholder="doctor@hospital.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required 
                    style={glassInputStyle}
                  />
                </div>
              </div>

              <div className="mb-2">
                <label className="form-label small fw-bold text-white-50 text-uppercase">Password</label>
                <div className="input-group">
                  <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.1)', color: '#ccc'}}><FaLock /></span>
                  <input 
                    type="password" 
                    className="form-control border-0" 
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required 
                    style={glassInputStyle}
                  />
                </div>
              </div>

              <div className="text-end mb-4">
                 <button type="button" onClick={handleForgotPassword} className="btn btn-link btn-sm p-0 text-white-50 fw-bold" style={{textDecoration: 'none', fontSize: '0.85rem'}}>
                   Forgot Password?
                 </button>
              </div>

              <button 
                type="submit" 
                className="btn w-100 py-3 rounded-3 fw-bold shadow-lg d-flex align-items-center justify-content-center mb-3 transition-transform"
                disabled={loading}
                style={{background: '#0984e3', border: 'none', color: '#fff'}} 
              >
                {loading ? 'Verifying Credentials...' : <>Log In <FaArrowRight className="ms-2"/></>}
              </button>
            </form>

            <div className="d-flex align-items-center mb-4">
                <hr className="flex-grow-1" style={{borderColor: 'rgba(255,255,255,0.1)'}}/>
                <span className="mx-3 small text-white-50 fw-bold">OR</span>
                <hr className="flex-grow-1" style={{borderColor: 'rgba(255,255,255,0.1)'}}/>
            </div>

            <button
               onClick={handleGoogleLogin}
               className="btn w-100 py-3 rounded-3 shadow-sm d-flex align-items-center justify-content-center mb-4 border border-secondary"
               disabled={loading}
               style={{background: 'transparent', color: '#fff'}}
            >
               <FaGoogle className="me-2 text-danger" />
               Sign in with Google
            </button>

            <div className="text-center">
              <small className="text-white-50">
                Need access? <Link to="/signup" className="fw-bold text-white" style={{textDecoration: 'none'}}>Request Account</Link>
              </small>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default Login;