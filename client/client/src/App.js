/* client/src/App.js */
import React, { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { onAuthStateChanged } from 'firebase/auth';
import { auth } from './firebase';
import axios from 'axios';
import { LanguageProvider } from './context/LanguageContext'; 
import config from './config'; // 👈 IMPORT CONFIG

// ==========================================
// 🧩 STANDARD COMPONENTS
// ==========================================
import Layout from './components/Layout';
import SmartRecorder from './components/SmartRecorder';
import DoctorDashboard from './components/DoctorDashboard';
import PatientList from './components/PatientList';
import AdminDashboard from './components/AdminDashboard';
import Analytics from './components/Analytics';
import Login from './components/Login';
import Signup from './components/Signup';

// ==========================================
// ⚙️ SETTINGS COMPONENTS
// ==========================================
import SettingsHub from './components/SettingsHub';
import DeviceSettings from './components/DeviceSettings';
import SyncSettings from './components/SyncSettings';
import AdminConfig from './components/AdminConfig';
import ProfileSettings from './components/ProfileSettings';
import PrivacySettings from './components/PrivacySettings';
import LanguageSettings from './components/LanguageSettings';
import AboutSettings from './components/AboutSettings';
import NotificationSettings from './components/NotificationSettings';

function App() {
  const [userRole, setUserRole] = useState(null);
  const [loading, setLoading] = useState(true);

  // ----------------------------------------------------------------
  // 🔐 AUTHENTICATION & ROLE CHECK
  // ----------------------------------------------------------------
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
      if (currentUser) {
        try {
          const token = await currentUser.getIdToken();
          // ✅ FIX: Use dynamic config.API_BASE_URL
          const res = await axios.post(`${config.API_BASE_URL}/login`, {}, {
            headers: { Authorization: `Bearer ${token}` }
          });
          setUserRole(res.data.role);
        } catch (err) {
          console.error("Session restore failed", err);
          setUserRole(null);
        }
      } else {
        setUserRole(null);
      }
      setLoading(false);
    });
    return () => unsubscribe();
  }, []);

  const logout = async () => {
    await auth.signOut();
    setUserRole(null);
  };

  if (loading) return <div className="text-center mt-5 text-dark-brown">Loading SautiYaAfya...</div>;

  return (
    <LanguageProvider> {/* 1. WRAP APP IN PROVIDER */}
      <Router>
        <div className="app-container">
          <Routes>
            
            {/* ========================================================
                1. MAIN LOGIN ROUTE
               ======================================================== */}
            <Route 
              path="/" 
              element={
                !userRole ? (
                  <Login setRole={setUserRole} />
                ) : (
                  <Navigate to={
                    userRole === 'CHW' ? "/chw" : 
                    userRole === 'DOCTOR' ? "/doctor" : 
                    "/admin"
                  } />
                )
              } 
            />

            <Route path="/signup" element={<Signup setRole={setUserRole} />} />
            
            {/* ========================================================
                2. CHW ROUTES (RESTRICTED)
               ======================================================== */}
            <Route path="/chw" element={
              userRole === 'CHW' ? (
                <Layout role="CHW" logout={logout}>
                  <SmartRecorder />
                </Layout>
              ) : <Navigate to="/" />
            } />

            {/* ========================================================
                3. DOCTOR ROUTES (EXPANDED)
               ======================================================== */}
            <Route path="/doctor" element={
              userRole === 'DOCTOR' ? (
                <Layout role="DOCTOR" logout={logout}>
                  <DoctorDashboard />
                </Layout>
              ) : <Navigate to="/" />
            } />

            <Route path="/doctor/patients" element={
              userRole === 'DOCTOR' ? (
                <Layout role="DOCTOR" logout={logout}>
                  <PatientList />
                </Layout>
              ) : <Navigate to="/" />
            } />

            <Route path="/doctor/analytics" element={
              userRole === 'DOCTOR' ? (
                <Layout role="DOCTOR" logout={logout}>
                  <Analytics />
                </Layout>
              ) : <Navigate to="/" />
            } />

            {/* ========================================================
                4. ADMIN ROUTES
               ======================================================== */}
            <Route path="/admin" element={
              userRole === 'ADMIN' ? (
                <Layout role="ADMIN" logout={logout}>
                  <AdminDashboard />
                </Layout>
              ) : <Navigate to="/" />
            } />


            {/* ========================================================
                5. SETTINGS ROUTES (Shared + Role Protected)
               ======================================================== */}
            
            {/* Main Settings Menu (All Roles) */}
            <Route path="/settings" element={
              userRole ? (
                <Layout role={userRole} logout={logout}>
                  <SettingsHub role={userRole} />
                </Layout>
              ) : <Navigate to="/" />
            } />

            {/* Device Settings (CHW Only) */}
            <Route path="/settings/device" element={
              userRole === 'CHW' ? (
                <Layout role={userRole} logout={logout}>
                  <DeviceSettings />
                </Layout>
              ) : <Navigate to="/settings" />
            } />

            {/* Sync Settings (Shared) */}
            <Route path="/settings/sync" element={
              userRole ? (
                <Layout role={userRole} logout={logout}>
                  <SyncSettings />
                </Layout>
              ) : <Navigate to="/" />
            } />

            {/* Admin Config (Admin Only) */}
            <Route path="/settings/admin-config" element={
              userRole === 'ADMIN' ? (
                <Layout role={userRole} logout={logout}>
                  <AdminConfig />
                </Layout>
              ) : <Navigate to="/settings" />
            } />

            {/* Profile (Shared) */}
            <Route path="/settings/profile" element={
              userRole ? (
                <Layout role={userRole} logout={logout}>
                  <ProfileSettings role={userRole} />
                </Layout>
              ) : <Navigate to="/" />
            } />

            {/* Privacy (Shared) */}
            <Route path="/settings/privacy" element={
              userRole ? (
                <Layout role={userRole} logout={logout}>
                  <PrivacySettings />
                </Layout>
              ) : <Navigate to="/" />
            } />

            {/* Language (Shared) */}
            <Route path="/settings/language" element={
              userRole ? (
                <Layout role={userRole} logout={logout}>
                  <LanguageSettings />
                </Layout>
              ) : <Navigate to="/" />
            } />

            {/* About (Shared) */}
            <Route path="/settings/about" element={
              userRole ? (
                <Layout role={userRole} logout={logout}>
                  <AboutSettings />
                </Layout>
              ) : <Navigate to="/" />
            } />

            {/* Notifications (Doctor Only) */}
            <Route path="/settings/notifications" element={
              userRole === 'DOCTOR' ? (
                <Layout role={userRole} logout={logout}>
                  <NotificationSettings />
                </Layout>
              ) : <Navigate to="/settings" />
            } />

          </Routes>
        </div>
      </Router>
    </LanguageProvider> 
  );
}

export default App;