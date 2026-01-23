/* client/src/components/Login.js */
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { 
  signInWithEmailAndPassword, 
  signInWithPopup 
} from 'firebase/auth';
import { auth, googleProvider, resetPassword } from '../firebase';
import axios from 'axios';
import { FaGoogle, FaStethoscope } from 'react-icons/fa';
import config from '../config'; 

const Login = ({ setRole }) => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [msg, setMsg] = useState('');
  const [loading, setLoading] = useState(false);

  const executeLogin = async (user) => {
    try {
      setLoading(true);
      const token = await user.getIdToken();
      
      const response = await axios.post(`${config.API_BASE_URL}/login`, {}, {
        headers: { Authorization: `Bearer ${token}` }
      });
      
      const { role } = response.data;
      setRole(role);
      navigate(role === 'CHW' ? '/chw' : '/doctor');
    } catch (err) {
       console.error("Backend Login Error:", err);
       if (err.response && err.response.status === 404) {
         setError("Account not found. Please Sign Up first.");
       } else {
         setError(err.response?.data?.message || "Login Failed");
       }
    } finally {
      setLoading(false);
    }
  };

  const handleEmailLogin = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    try {
      const cred = await signInWithEmailAndPassword(auth, email, password);
      await executeLogin(cred.user);
    } catch (err) {
      setError("Invalid Email or Password.");
      setLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    setError('');
    try {
      const result = await signInWithPopup(auth, googleProvider);
      setLoading(true);
      await executeLogin(result.user);
    } catch (err) {
      console.error("Google Popup Error:", err);
      if (err.code === 'auth/popup-blocked') {
         setError("Popup blocked! Please allow popups.");
      } else {
         setError("Google Login Failed: " + err.message);
      }
      setLoading(false);
    }
  };

  const handleForgotPassword = async () => {
    if(!email) return setError("Please enter your email first.");
    try {
      await resetPassword(email);
      setMsg("Password reset email sent! Check your inbox.");
      setError('');
    } catch (err) {
      setError("Error sending reset email: " + err.message);
    }
  };

  return (
    // 1. New Wrapper Class (Fixes Centering)
    <div className="login-page-wrapper">
      
      {/* 2. New Compact Card Class (Fixes Stretching) */}
      <div className="login-card-compact shadow-lg position-relative overflow-hidden">
        
        {/* Header - Reduced Margins */}
        <div className="text-center mb-4">
            <div className="login-header-icon bg-white rounded-circle d-inline-flex align-items-center justify-content-center shadow-sm mb-3" style={{width: '60px', height: '60px'}}>
                <FaStethoscope className="text-primary fs-3" />
            </div>
            <h3 className="fw-bold text-dark-brown mb-1">SautiYaAfya</h3>
            <p className="text-dark-brown opacity-75 small fw-bold mb-0">AI Respiratory Triage System</p>
        </div>
        
        {error && <div className="alert alert-danger small py-2 border-0 shadow-sm mb-3">{error}</div>}
        {msg && <div className="alert alert-success small py-2 border-0 shadow-sm mb-3">{msg}</div>}

        <form onSubmit={handleEmailLogin}>
          <div className="mb-3">
            <input 
              type="email" 
              className="form-control login-input-glass"
              placeholder="Email Address"
              required
              value={email} onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="mb-3">
            <input 
              type="password" 
              className="form-control login-input-glass"
              placeholder="Password"
              required
              value={password} onChange={(e) => setPassword(e.target.value)}
            />
            <div className="text-end mt-2">
              <button type="button" onClick={handleForgotPassword} className="btn btn-link btn-sm p-0 text-decoration-none text-dark-brown opacity-75 fw-bold" style={{fontSize: '0.8rem'}}>
                Forgot Password?
              </button>
            </div>
          </div>

          <button type="submit" className="btn btn-primary w-100 py-2 rounded-pill shadow-sm fw-bold mb-3" disabled={loading} style={{border: 'none'}}>
            {loading ? "Verifying..." : "Login to Portal"}
          </button>
        </form>

        <div className="d-flex align-items-center my-3">
            <div className="flex-grow-1 border-bottom border-secondary opacity-25"></div>
            <span className="mx-3 text-dark-brown opacity-50 small">OR</span>
            <div className="flex-grow-1 border-bottom border-secondary opacity-25"></div>
        </div>

        <button onClick={handleGoogleLogin} className="btn bg-white w-100 py-2 rounded-pill shadow-sm text-dark fw-bold mb-4" disabled={loading}>
           <FaGoogle className="me-2 text-danger"/> 
           {loading ? "Processing..." : "Continue with Google"}
        </button>

        <div className="text-center">
          <small className="text-dark-brown opacity-75">New here? <Link to="/signup" className="fw-bold text-primary">Create Account</Link></small>
        </div>
      </div>
    </div>
  );
};

export default Login;