/* client/src/components/UserGuide.js */
import React from 'react';
import { 
  FaBook, FaMicrophone, FaServer, FaStethoscope, 
  FaExclamationTriangle, FaInfoCircle 
} from 'react-icons/fa';

const UserGuide = () => {

  const sectionStyle = {
    background: 'rgba(255, 255, 255, 0.45)', 
    borderRadius: '16px',
    padding: '25px',
    marginBottom: '20px',
    border: '1px solid rgba(255, 255, 255, 0.4)',
    backdropFilter: 'blur(12px)'
  };

  const titleStyle = {
      color: '#2c3e50',
      fontWeight: 'bold',
      marginBottom: '15px',
      display: 'flex',
      alignItems: 'center'
  };

  return (
    <div className="container-fluid p-0 animate-slide-in">
      
      {/* Header */}
      <div className="d-flex align-items-center mb-4">
        <div className="bg-white p-3 rounded-circle shadow-sm text-primary me-3">
            <FaBook size={24} />
        </div>
        <div>
            <h3 className="fw-bold text-dark-brown mb-0">System User Guide</h3>
            <p className="text-muted mb-0">How to use the SautiYaAfya AI Triage System</p>
        </div>
      </div>

      <div className="row g-4">
        
        {/* LEFT COLUMN */}
        <div className="col-12 col-lg-6">
            
            {/* 1. RECORDING BEST PRACTICES */}
            <div style={sectionStyle} className="shadow-sm">
                <h5 style={titleStyle}><FaMicrophone className="me-2 text-danger"/> 1. Recording Best Practices</h5>
                <p className="text-dark small" style={{lineHeight: '1.6'}}>
                    To ensure the AI gives an accurate diagnosis (98% accuracy), follow these strict rules during screening:
                </p>
                <ul className="small text-muted ps-3 mb-0">
                    <li className="mb-2"><strong>Silence is Key:</strong> Ensure the room is completely quiet. Background talking will confuse the AI.</li>
                    
                    {/* 🟢 NEW: NOSE/MOUTH POSITIONING */}
                    <li className="mb-2">
                        <strong>Microphone Position:</strong> Hold the phone microphone approx. <strong>5cm away</strong> from the nose/mouth. 
                        Hold it at a <strong>90-degree angle</strong> (sideways) to avoid air blowing directly into the mic.
                    </li>

                    {/* 🟢 NEW: BREATHING INSTRUCTION */}
                    <li className="mb-2">
                        <strong>Patient Breathing:</strong> Instruct the patient to <strong>breathe deeply and audibly</strong> (a bit loud) through their mouth.
                    </li>
                    
                    <li><strong>Duration:</strong> Record for at least 10-15 seconds to capture multiple breath cycles.</li>
                </ul>
            </div>

            {/* 2. UNDERSTANDING AI RESULTS */}
            <div style={sectionStyle} className="shadow-sm">
                <h5 style={titleStyle}><FaStethoscope className="me-2 text-primary"/> 2. Interpreting AI Results</h5>
                <div className="d-flex align-items-center mb-3">
                    <span className="badge bg-danger me-2">High Risk</span>
                    <small className="text-muted">Immediate referral required. AI detected strong signs of Pneumonia.</small>
                </div>
                <div className="d-flex align-items-center mb-3">
                    <span className="badge bg-warning text-dark me-2">Moderate</span>
                    <small className="text-muted">Signs of Asthma or wheezing. Observe and re-screen in 24hrs.</small>
                </div>
                <div className="d-flex align-items-center">
                    <span className="badge bg-success me-2">Stable</span>
                    <small className="text-muted">Normal vesicular breath sounds. No distress detected.</small>
                </div>
            </div>

        </div>

        {/* RIGHT COLUMN */}
        <div className="col-12 col-lg-6">
            
            {/* 3. SERVER STATUS */}
            <div style={sectionStyle} className="shadow-sm">
                <h5 style={titleStyle}><FaServer className="me-2 text-info"/> 3. AI Engine Status</h5>
                <p className="small text-muted mb-3">
                    The AI Engine runs on a secure cloud server that "sleeps" to save energy when not in use.
                </p>
                <div className="alert alert-warning small border-0 d-flex align-items-center">
                    <FaInfoCircle className="me-2"/>
                    <span>
                        <strong>"Waking Up":</strong> If you see this, please wait 30-60 seconds. The server is booting up. 
                        Do not refresh the page.
                    </span>
                </div>
            </div>

             {/* 4. TROUBLESHOOTING */}
             <div style={sectionStyle} className="shadow-sm">
                <h5 style={titleStyle}><FaExclamationTriangle className="me-2 text-warning"/> 4. Troubleshooting</h5>
                <ul className="small text-muted ps-3 mb-0">
                    <li className="mb-2"><strong>Microphone Blocked?</strong> Click the "Lock" icon in your browser URL bar and set Microphone to "Allow".</li>
                    <li className="mb-2"><strong>No Audio?</strong> Check that your volume is up and the file downloaded correctly.</li>
                    <li><strong>Unexpected Errors?</strong> Try refreshing the page or restarting your browser.</li>
                </ul>
            </div>

        </div>
      </div>
    </div>
  );
};

export default UserGuide;