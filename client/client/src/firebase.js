/* client/src/firebase.js */
import { initializeApp } from "firebase/app";
import { 
  getAuth, 
  GoogleAuthProvider, 
  sendPasswordResetEmail,
  signInWithPopup
} from "firebase/auth";
import config from './config'; // Import the central config

// ✅ FIX: Use keys from config.js (which loads from process.env)
const firebaseConfig = {
  apiKey: config.FIREBASE.apiKey,
  authDomain: config.FIREBASE.authDomain,
  projectId: config.FIREBASE.projectId,
  storageBucket: config.FIREBASE.storageBucket,
  messagingSenderId: config.FIREBASE.messagingSenderId,
  appId: config.FIREBASE.appId
};

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
export const googleProvider = new GoogleAuthProvider();

// Helper functions for cleaner components
export const resetPassword = (email) => sendPasswordResetEmail(auth, email);
export const signInWithGoogle = () => signInWithPopup(auth, googleProvider);