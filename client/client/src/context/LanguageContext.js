/* client/src/context/LanguageContext.js */
import React, { createContext, useState, useEffect } from 'react';
import axios from 'axios';
import { auth } from '../firebase';
import { translations } from '../translations';
import config from '../config'; // 👈 IMPORT CONFIG

export const LanguageContext = createContext();

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState('en'); // Default to English

  // 1. Load initial language from DB on startup
  useEffect(() => {
    const fetchLang = async () => {
      try {
        const user = auth.currentUser;
        if (user) {
          const token = await user.getIdToken();
          // ✅ FIX: Use dynamic config.API_BASE_URL
          const res = await axios.get(`${config.API_BASE_URL}/settings`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          if (res.data && res.data.language) {
            setLanguage(res.data.language);
          }
        }
      } catch (err) {
        console.error("Language load failed", err);
      }
    };
    fetchLang();
  }, []);

  // 2. The Real-Time Switcher Function
  const changeLanguage = async (newLang) => {
    // A. Optimistic Update (Update UI instantly)
    setLanguage(newLang);

    // B. Save to Database in background
    try {
      const user = auth.currentUser;
      if (user) {
        const token = await user.getIdToken();
        // ✅ FIX: Use dynamic config.API_BASE_URL
        await axios.put(`${config.API_BASE_URL}/settings`, 
          { language: newLang }, 
          { headers: { Authorization: `Bearer ${token}` } }
        );
      }
    } catch (err) {
      console.error("Failed to persist language", err);
      // Optional: Revert on failure, but usually better to leave UI as is
    }
  };

  // 3. The Translation Helper (t function)
  const t = (key) => {
    return translations[language][key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, changeLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
};