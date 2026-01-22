/* client/src/components/Login.js */
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { 
  signInWithEmailAndPassword, 
  signInWithPopup // <--- BACK TO POPUP (It is the only way on localhost)
} from 'firebase/auth';
import { auth, googleProvider, resetPassword } from '../firebase';
import axios from 'axios';
import { FaGoogle, FaStethoscope } from 'react-icons/fa';

const Login = ({ setRole }) => {
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [msg, setMsg] = useState('');
  const [loading, setLoading] = useState(false);

  // Custom Glass Input Style
  const glassInputStyle = {
      background: 'rgba(255,255,255,0.4)',
      border: '1px solid rgba(255,255,255,0.6)',
      color: '#2d3436',
      backdropFilter: 'blur(5px)',
      borderRadius: '12px',
      padding: '12px'
  };

  const executeLogin = async (user) => {
    try {
      setLoading(true);
      const token = await user.getIdToken();
      
      const response = await axios.post('http://localhost:5000/api/login', {}, {
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

  // 🚨 FIXED GOOGLE LOGIN
  const handleGoogleLogin = async () => {
    setError('');
    
    // IMPORTANT: Do NOT set loading=true here. 
    // Updating state before the popup causes browsers to block it.
    
    try {
      // 1. Open Popup IMMEDIATELY
      const result = await signInWithPopup(auth, googleProvider);
      
      // 2. NOW we can show loading and process the user
      setLoading(true);
      await executeLogin(result.user);
    } catch (err) {
      console.error("Google Popup Error:", err);
      
      if (err.code === 'auth/popup-blocked') {
         setError("Popup blocked! Please allow popups for this site in your browser address bar.");
      } else if (err.code === 'auth/popup-closed-by-user') {
         setError("Login cancelled.");
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
    <div className="min-vh-100 d-flex align-items-center justify-content-center">
      
      <div className="glass-card p-5 shadow-lg position-relative overflow-hidden" style={{ maxWidth: '420px', width: '90%', borderRadius: '24px' }}>
        
        <div className="text-center mb-5 position-relative">
            <div className="bg-white rounded-circle d-inline-flex align-items-center justify-content-center shadow-sm mb-3" style={{width: '70px', height: '70px'}}>
                <FaStethoscope className="text-primary fs-2" />
            </div>
            <h3 className="fw-bold text-dark-brown">SautiYaAfya</h3>
            <p className="text-dark-brown opacity-75 small fw-bold">AI Respiratory Triage System</p>
        </div>
        
        {error && <div className="alert alert-danger small border-0 shadow-sm" style={{background: 'rgba(220, 53, 69, 0.9)', color: 'white'}}>{error}</div>}
        {msg && <div className="alert alert-success small border-0 shadow-sm" style={{background: 'rgba(46, 204, 113, 0.9)', color: 'white'}}>{msg}</div>}

        <form onSubmit={handleEmailLogin}>
          <div className="mb-3">
            <input 
              type="email" 
              className="form-control" 
              placeholder="Email Address"
              required
              style={glassInputStyle}
              value={email} onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="mb-4">
            <input 
              type="password" 
              className="form-control" 
              placeholder="Password"
              required
              style={glassInputStyle}
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