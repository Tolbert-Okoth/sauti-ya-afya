/* client/src/components/Signup.js */
import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { createUserWithEmailAndPassword, signInWithPopup } from 'firebase/auth';
import { auth, googleProvider } from '../firebase';
import axios from 'axios';
import { FaGoogle, FaUserPlus } from 'react-icons/fa';
import config from '../config';

const Signup = ({ setRole }) => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    role: 'CHW',
    county_id: 1
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  // Glass input style (matched with Login)
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
      {
        role: formData.role,
        county_id: formData.county_id
      },
      {
        headers: { Authorization: `Bearer ${token}` }
      }
    );
  };

  const handleSignup = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      const userCredential = await createUserWithEmailAndPassword(
        auth,
        formData.email,
        formData.password
      );

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
      setError('Google Signup Failed.');
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
            Join the SautiYaAfya Network
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

          <div className="row mb-3">
            <div className="col-6">
              <select
                className="form-select"
                style={glassInputStyle}
                value={formData.role}
                onChange={(e) =>
                  setFormData({ ...formData, role: e.target.value })
                }
              >
                <option value="CHW">CHW</option>
                <option value="DOCTOR">Doctor</option>
              </select>
            </div>

            <div className="col-6">
              <input
                type="number"
                className="form-control"
                style={glassInputStyle}
                value={formData.county_id}
                onChange={(e) =>
                  setFormData({ ...formData, county_id: e.target.value })
                }
              />
            </div>
          </div>

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
