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

  const glassInputStyle = {
    background: 'rgba(255,255,255,0.45)',
    border: '1px solid rgba(255,255,255,0.6)',
    color: '#2d3436',
    backdropFilter: 'blur(6px)',
    borderRadius: '12px',
    padding: '11px'
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
      setRole(role);
      navigate(role === 'CHW' ? '/chw' : '/doctor');
    } catch (err) {
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
    setLoading(true);
    try {
      const cred = await signInWithEmailAndPassword(auth, email, password);
      await executeLogin(cred.user);
    } catch {
      setError('Invalid Email or Password.');
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
    <div className="min-vh-100 d-flex align-items-center justify-content-center">
      <div
        className="glass-card p-4 shadow-lg"
        style={{ maxWidth: '380px', width: '92%', borderRadius: '22px' }}
      >
        <div className="text-center mb-4">
          <div
            className="bg-white rounded-circle d-inline-flex align-items-center justify-content-center shadow-sm mb-2"
            style={{ width: '64px', height: '64px' }}
          >
            <FaStethoscope className="text-primary fs-4" />
          </div>
          <h4 className="fw-bold text-dark-brown mb-1">SautiYaAfya</h4>
          <p className="text-dark-brown opacity-75 small">
            AI Respiratory Triage System
          </p>
        </div>

        {error && <div className="alert alert-danger small">{error}</div>}
        {msg && <div className="alert alert-success small">{msg}</div>}

        <form onSubmit={handleEmailLogin}>
          <input
            type="email"
            className="form-control mb-3"
            placeholder="Email Address"
            style={glassInputStyle}
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />

          <input
            type="password"
            className="form-control mb-2"
            placeholder="Password"
            style={glassInputStyle}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />

          <div className="text-end mb-3">
            <button
              type="button"
              onClick={handleForgotPassword}
              className="btn btn-link btn-sm p-0"
            >
              Forgot Password?
            </button>
          </div>

          <button
            type="submit"
            className="btn btn-primary w-100 py-2 rounded-pill"
            disabled={loading}
          >
            {loading ? 'Verifying...' : 'Login to Portal'}
          </button>
        </form>

        <hr className="my-4" />

        <button
          onClick={handleGoogleLogin}
          className="btn bg-white w-100 py-2 rounded-pill shadow-sm"
          disabled={loading}
        >
          <FaGoogle className="me-2 text-danger" />
          Continue with Google
        </button>

        <div className="text-center mt-3">
          <small className="text-dark-brown opacity-75">
            New here? <Link to="/signup" className="fw-bold">Create Account</Link>
          </small>
        </div>
      </div>
    </div>
  );
};

export default Login;
