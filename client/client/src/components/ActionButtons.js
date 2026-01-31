/* client/src/components/ActionButtons.js */
import React from 'react';
import { FaWhatsapp, FaPhone } from 'react-icons/fa';

const ActionButtons = ({ patient, riskLevel }) => {
  // Kenyan Phone Format Helper
  const formatPhone = (number) => {
    // In a real app, this comes from the User DB. 
    return "254700000000"; 
  };

  const generateMessage = () => {
    const emoji = riskLevel === 'High' ? '🔴' : '🟡';
    return `URGENT REFERRAL ${emoji}%0a
Patient: ${patient.name} (${patient.age})%0a
Location: ${patient.location}%0a
Symptoms: ${patient.symptoms}%0a
AI Finding: ${patient.diagnosis}%0a
%0aPlease advise immediately.`;
  };

  const whatsappUrl = `https://wa.me/${formatPhone()}?text=${generateMessage()}`;
  const callUrl = `tel:${formatPhone()}`;

  return (
    <div className="btn-group shadow-sm" role="group">
      <a href={whatsappUrl} target="_blank" rel="noreferrer" className="btn btn-success btn-sm text-white fw-bold">
        <FaWhatsapp className="me-1" /> Refer
      </a>
      {/* 🟢 Updated Call Button to Outline White for better contrast on dark bg */}
      <a href={callUrl} className="btn btn-outline-light btn-sm fw-bold">
        <FaPhone className="me-1" /> Call
      </a>
    </div>
  );
};

export default ActionButtons;