/* client/src/components/Footer.js */
import React from 'react';
import { FaExclamationCircle } from 'react-icons/fa';

const Footer = () => {
  return (
    <div className="mt-auto py-3 px-4 text-center" 
      style={{
        borderTop: '1px solid rgba(255,255,255,0.2)',
        background: 'rgba(255,255,255,0.02)', // Very faint glass
        backdropFilter: 'blur(5px)',
        color: '#636e72',
        fontSize: '0.75rem',
        marginTop: '20px'
      }}
    >
      <div className="container-fluid">
        <p className="mb-0 d-flex align-items-center justify-content-center flex-wrap">
          <FaExclamationCircle className="me-2 text-warning" />
          <strong>MEDICAL DISCLAIMER:</strong> &nbsp; 
          SautiYaAfya is an AI-assisted triage support tool. 
          Results are preliminary and must be verified by a qualified clinical officer. 
        </p>
        <p className="mb-0 opacity-50" style={{fontSize: '0.65rem', marginTop:'4px'}}>
          v2.0.4 (Defense Build) • Powered by PyTorch & Librosa • © 2026 SautiYaAfya Research
        </p>
      </div>
    </div>
  );
};

export default Footer;