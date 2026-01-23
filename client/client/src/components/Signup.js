/* client/src/components/Signup.js */
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { createUserWithEmailAndPassword, signInWithPopup } from 'firebase/auth';
import { auth, googleProvider } from '../firebase';
import axios from 'axios';
import { FaGoogle, FaUserPlus } from 'react-icons/fa';
import config from '../config';

const Signup = () => {
  const navigate = useNavigate();

  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: ''
  });

  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // ✅ Email regex (RFC-compliant enough for real apps)
  const emailRegex =
    /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

  // ✅ Strong password regex
  // Minimum 8 chars, 1 uppercase, 1 lowercase, 1 number, 1 special char
  const strongPasswordRegex =
    /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&])[A-Za-z\d@$!%*?&]{8,}$/;

  // Glass input style
  const glassInputStyle = {
    background: 'rgba(255,255,255,0.45)',
    border: '1px solid rgba(255,255,255,0.6)',
    color: '#2d3436',
    backdropFilter: 'blur(6px)',
    borderRadius: '12px',
    padding: '11px'
  };

  const registerInBackend = async (user) => {
    const token = await user.getIdToken();

    await axios.post(
      `${config.API_BASE_URL}/register`,
      {},
      {
        headers: { Authorization: `Bearer ${token}` }
      }
    );
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setError('');

    // 🔐 Frontend validations
    if (!emailRegex.test(formData.email)) {
      return setError('Please enter a valid email address.');
    }

    if (!strongPasswordRegex.test(formData.password)) {
      return setError(
        'Password must be at least 8 characters and include uppercase, lowercase, number, and special character.'
      );
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
      navigate('/dashboard');
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
      navigate('/dashboard');
    } catch (err) {
      setError('Google signup failed.');
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
            style={{ width: '56px', height: '56px' }}
          >
            <FaUserPlus className="text-accent fs-5" />
          </div>
          <h4 className="fw-bold text-dark-brown mb-1">Create Account</h4>
          <p className="text-dark-brown opacity-75 small">
            Join the SautiYaAfya Platform
          </p>
        </div>

        {error && (
          <div className="alert alert-danger small border-0 shadow-sm">
            {error}
          </div>
        )}

        <form onSubmit={handleSignup}>
          <input
            type="email"
            className="form-control mb-3"
            placeholder="Email Address"
            style={glassInputStyle}
            required
            onChange={(e) =>
              setFormData({ ...formData, email: e.target.value })
            }
          />

          <input
            type="password"
            className="form-control mb-3"
            placeholder="Password"
            style={glassInputStyle}
            required
            onChange={(e) =>
              setFormData({ ...formData, password: e.target.value })
            }
          />

          <input
            type="password"
            className="form-control mb-3"
            placeholder="Confirm Password"
            style={glassInputStyle}
            required
            onChange={(e) =>
              setFormData({ ...formData, confirmPassword: e.target.value })
            }
          />

          <button
            type="submit"
            className="btn btn-primary w-100 py-2 rounded-pill fw-bold mb-3"
            disabled={loading}
          >
            {loading ? 'Creating Account...' : 'Sign Up'}
          </button>
        </form>

        <hr className="my-4" />

        <button
          onClick={handleGoogleSignup}
          className="btn bg-white w-100 py-2 rounded-pill shadow-sm fw-bold mb-3"
          disabled={loading}
        >
          <FaGoogle className="me-2 text-danger" />
          Sign up with Google
        </button>

        <div className="text-center">
          <small className="text-dark-brown opacity-75">
            Already have an account?{' '}
            <Link to="/" className="fw-bold">
              Login here
            </Link>
          </small>
        </div>
      </div>
    </div>
  );
};

export default Signup;
