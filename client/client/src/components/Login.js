/* client/src/components/Login.js */
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { signInWithEmailAndPassword, signInWithPopup } from 'firebase/auth';
import { auth, googleProvider, resetPassword } from '../firebase';
import axios from 'axios';
import { 
  FaGoogle, FaStethoscope, FaChartLine, FaShieldAlt, 
  FaUserMd, FaLock, FaArrowRight 
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
      
      {/* MAIN GLASS CARD - SPLIT LAYOUT */}
      <div 
        className="glass-card d-flex flex-column flex-md-row overflow-hidden shadow-lg animate-slide-in"
        style={{ 
          maxWidth: '1000px', 
          width: '90%', 
          minHeight: '600px',
          background: 'rgba(255, 255, 255, 0.65)', 
          border: '1px solid rgba(255, 255, 255, 0.8)'
        }}
      >
        
        {/* 👈 LEFT SIDE: INTRODUCTION & BRANDING */}
        <div className="col-12 col-md-6 p-5 d-flex flex-column justify-content-center text-dark-brown" 
             style={{
               background: 'linear-gradient(135deg, rgba(255,255,255,0.4), rgba(255,255,255,0.1))',
               borderRight: '1px solid rgba(255,255,255,0.3)'
             }}>
          
          <div className="mb-4">
            <div className="d-inline-flex align-items-center justify-content-center bg-primary text-white rounded-circle mb-3 shadow-sm" style={{width: '60px', height: '60px'}}>
              <FaStethoscope size={28} />
            </div>
            <h1 className="fw-bold display-5 mb-2">SautiYaAfya</h1>
            <p className="lead fw-normal mb-0">AI-Powered Pediatric Triage</p>
          </div>

          <p className="mb-4 text-muted" style={{lineHeight: '1.6'}}>
            Revolutionizing respiratory screening in Kenya with our "Two-Brain" AI system. 
            We combine standard medical algorithms with advanced deep learning to detect pneumonia and asthma.
          </p>

          <div className="d-flex flex-column gap-3">
            <div className="d-flex align-items-center">
              <div className="bg-white p-2 rounded-circle me-3 shadow-sm text-success"><FaChartLine /></div>
              <span className="fw-bold text-muted">Real-Time Acoustic Analysis</span>
            </div>
            <div className="d-flex align-items-center">
              <div className="bg-white p-2 rounded-circle me-3 shadow-sm text-info"><FaShieldAlt /></div>
              <span className="fw-bold text-muted">Secure Patient Data Architecture</span>
            </div>
            <div className="d-flex align-items-center">
              <div className="bg-white p-2 rounded-circle me-3 shadow-sm text-warning"><FaUserMd /></div>
              <span className="fw-bold text-muted">Empowering Community Health Workers</span>
            </div>
          </div>

         

        {/* 👉 RIGHT SIDE: LOGIN FORM */}
        <div className="col-12 col-md-6 p-5 d-flex align-items-center">
          <div className="w-100">
            <div className="text-center mb-4">
              <h3 className="fw-bold text-dark-brown">Welcome Back</h3>
              <p className="text-muted">Please sign in to access the dashboard</p>
            </div>

            {error && <div className="alert alert-danger text-center shadow-sm py-2 small">{error}</div>}
            {msg && <div className="alert alert-success text-center shadow-sm py-2 small">{msg}</div>}

            <form onSubmit={handleEmailLogin}>
              <div className="mb-3">
                <label className="form-label small fw-bold text-muted text-uppercase">Email Address</label>
                <div className="input-group">
                  <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.4)', borderRight: 'none'}}><FaUserMd className="text-muted"/></span>
                  <input 
                    type="email" 
                    className="form-control" 
                    placeholder="doctor@hospital.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    style={{...glassInputStyle, borderLeft: 'none'}}
                    required 
                  />
                </div>
              </div>

              <div className="mb-2">
                <label className="form-label small fw-bold text-muted text-uppercase">Password</label>
                <div className="input-group">
                  <span className="input-group-text border-0" style={{background: 'rgba(255,255,255,0.4)', borderRight: 'none'}}><FaLock className="text-muted"/></span>
                  <input 
                    type="password" 
                    className="form-control" 
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    style={{...glassInputStyle, borderLeft: 'none'}}
                    required 
                  />
                </div>
              </div>

              <div className="text-end mb-4">
                 <button type="button" onClick={handleForgotPassword} className="btn btn-link btn-sm p-0 text-muted" style={{textDecoration: 'none'}}>
                   Forgot Password?
                 </button>
              </div>

              <button 
                type="submit" 
                className="btn btn-primary w-100 py-3 rounded-3 fw-bold shadow-sm d-flex align-items-center justify-content-center mb-3"
                disabled={loading}
              >
                {loading ? 'Authenticating...' : <>Access Portal <FaArrowRight className="ms-2"/></>}
              </button>
            </form>

            <div className="d-flex align-items-center mb-3">
                <hr className="flex-grow-1" style={{borderColor: 'rgba(0,0,0,0.1)'}}/>
                <span className="mx-2 small text-muted">OR</span>
                <hr className="flex-grow-1" style={{borderColor: 'rgba(0,0,0,0.1)'}}/>
            </div>

            <button
               onClick={handleGoogleLogin}
               className="btn bg-white w-100 py-2 rounded-3 shadow-sm d-flex align-items-center justify-content-center mb-4"
               disabled={loading}
               style={{border: '1px solid #eee'}}
            >
               <FaGoogle className="me-2 text-danger" />
               Continue with Google
            </button>

            <div className="text-center">
              <small className="text-muted">
                New here? <Link to="/signup" className="fw-bold text-primary" style={{textDecoration: 'none'}}>Create Account</Link>
              </small>
            </div>
          </div>
        </div>

      </div>
    </div>
    </div>
  );
};

export default Login;