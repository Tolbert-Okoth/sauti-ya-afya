/* client/src/index.js */
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
// import reportWebVitals from './reportWebVitals'; // Optional

const root = ReactDOM.createRoot(document.getElementById('root'));

// 🛑 REMOVED <React.StrictMode> TO FIX GOOGLE LOGIN CRASH
root.render(
    <App />
);