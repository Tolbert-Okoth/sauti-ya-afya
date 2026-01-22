/* client/src/config.js */
const isLocal = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";

const SERVER_URL = isLocal 
  ? "http://localhost:5000" 
  : "https://sauti-backend.onrender.com"; // ✅ Points to your Render Node App

const AI_URL = isLocal
  ? "http://localhost:8000"
  : "https://sauti-ai-engine.onrender.com"; // ✅ Points to your Render Python App

const config = {
  SERVER_URL: SERVER_URL,
  API_BASE_URL: `${SERVER_URL}/api`,
  AI_URL: AI_URL, 
  
  // These load from your .env file locally, or Vercel Environment Variables in prod
  FIREBASE: {
    apiKey: process.env.REACT_APP_FIREBASE_API_KEY,
    authDomain: process.env.REACT_APP_FIREBASE_AUTH_DOMAIN,
    projectId: process.env.REACT_APP_FIREBASE_PROJECT_ID,
    storageBucket: process.env.REACT_APP_FIREBASE_STORAGE_BUCKET,
    messagingSenderId: process.env.REACT_APP_FIREBASE_MESSAGING_SENDER_ID,
    appId: process.env.REACT_APP_FIREBASE_APP_ID,
  },
};

export default config;