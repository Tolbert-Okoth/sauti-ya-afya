/* client/src/components/OutbreakMap.js */
import React, { useMemo } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { FaMapMarkedAlt, FaCircle } from 'react-icons/fa';

// 📍 COMPLETE COORDINATE DICTIONARY (All 47 Counties)
const KENYA_COORDS = {
  "Mombasa": { lat: -4.0435, lng: 39.6682 },
  "Kwale": { lat: -4.1737, lng: 39.4521 },
  "Kilifi": { lat: -3.6305, lng: 39.8499 },
  "Tana River": { lat: -1.5000, lng: 40.0000 },
  "Lamu": { lat: -2.2696, lng: 40.9006 },
  "Taita/Taveta": { lat: -3.3161, lng: 38.4850 },
  "Garissa": { lat: -0.4532, lng: 39.6460 },
  "Wajir": { lat: 1.7471, lng: 40.0573 },
  "Mandera": { lat: 3.9373, lng: 41.8569 },
  "Marsabit": { lat: 2.3369, lng: 37.9904 },
  "Isiolo": { lat: 0.3546, lng: 37.5822 },
  "Meru": { lat: 0.0463, lng: 37.6559 },
  "Tharaka-Nithi": { lat: -0.2965, lng: 37.8747 },
  "Embu": { lat: -0.5380, lng: 37.4580 },
  "Kitui": { lat: -1.3733, lng: 38.0106 },
  "Machakos": { lat: -1.5177, lng: 37.2634 },
  "Makueni": { lat: -1.8041, lng: 37.6203 },
  "Nyandarua": { lat: -0.1804, lng: 36.3707 },
  "Nyeri": { lat: -0.4167, lng: 36.9500 },
  "Kirinyaga": { lat: -0.5000, lng: 37.3333 },
  "Murang'a": { lat: -0.7167, lng: 37.1500 },
  "Kiambu": { lat: -1.1714, lng: 36.8356 },
  "Turkana": { lat: 3.1167, lng: 35.6000 },
  "West Pokot": { lat: 1.2333, lng: 35.1167 },
  "Samburu": { lat: 1.1667, lng: 36.6667 },
  "Trans Nzoia": { lat: 1.0167, lng: 35.0000 },
  "Uasin Gishu": { lat: 0.5143, lng: 35.2698 },
  "Elgeyo/Marakwet": { lat: 0.8000, lng: 35.5000 },
  "Nandi": { lat: 0.1667, lng: 35.0833 },
  "Baringo": { lat: 0.5000, lng: 35.7500 },
  "Laikipia": { lat: 0.3333, lng: 36.8333 },
  "Nakuru": { lat: -0.3031, lng: 36.0800 },
  "Narok": { lat: -1.0833, lng: 35.8667 },
  "Kajiado": { lat: -1.8500, lng: 36.7833 },
  "Kericho": { lat: -0.3667, lng: 35.2833 },
  "Bomet": { lat: -0.7833, lng: 35.3500 },
  "Kakamega": { lat: 0.2833, lng: 34.7500 },
  "Vihiga": { lat: 0.0833, lng: 34.7167 },
  "Bungoma": { lat: 0.5667, lng: 34.5667 },
  "Busia": { lat: 0.4608, lng: 34.1115 },
  "Siaya": { lat: -0.0667, lng: 34.2500 },
  "Kisumu": { lat: -0.0917, lng: 34.7680 },
  "Homa Bay": { lat: -0.5167, lng: 34.4500 },
  "Migori": { lat: -1.0667, lng: 34.4667 },
  "Kisii": { lat: -0.6833, lng: 34.7667 },
  "Nyamira": { lat: -0.5633, lng: 34.9358 },
  "Nairobi City": { lat: -1.2921, lng: 36.8219 }
};

const OutbreakMap = ({ patients = [] }) => {
  const position = [0.0236, 37.9062]; // Center of Kenya

  // 1. PROCESS DATA: Group patients by Location
  const clusters = useMemo(() => {
    const locationCounts = {};

    patients.forEach(patient => {
      // Normalize location string (Title Case / Trim)
      const loc = patient.location ? patient.location.trim() : "Unknown";
      
      // Initialize if not exists
      if (!locationCounts[loc]) {
        locationCounts[loc] = { count: 0, riskSum: 0, names: [] };
      }

      // Increment Count
      locationCounts[loc].count += 1;
      locationCounts[loc].names.push(patient.name); 
      
      // Track High Risk cases
      if (patient.risk_level === 'High') locationCounts[loc].riskSum += 1;
    });

    // Convert to Array for Rendering
    return Object.keys(locationCounts).map(key => {
      const coords = KENYA_COORDS[key] || KENYA_COORDS["Nairobi City"]; 
      
      return {
        name: key,
        lat: coords.lat,
        lng: coords.lng,
        count: locationCounts[key].count,
        highRiskCount: locationCounts[key].riskSum,
        patientNames: locationCounts[key].names
      };
    });
  }, [patients]);

  return (
    <div className="glass-card p-0 overflow-hidden shadow-sm h-100">
      <div className="p-3 border-bottom border-secondary d-flex justify-content-between align-items-center">
        <div className="d-flex align-items-center">
            <FaMapMarkedAlt className="text-info me-2" />
            <span className="fw-bold text-white">Live Surveillance</span>
        </div>
        <div className="d-flex align-items-center small text-white-50">
            <span className="me-3 d-flex align-items-center"><FaCircle size={8} className="text-danger me-1"/> Critical</span>
            <span className="d-flex align-items-center"><FaCircle size={8} className="text-warning me-1"/> Warning</span>
        </div>
      </div>
      
      <div className="position-relative">
        <MapContainer 
          center={position} 
          zoom={6} 
          style={{ height: "450px", width: "100%", background: 'transparent' }}
          scrollWheelZoom={false}
          className="z-0"
        >
          <TileLayer
            attribution='&copy; OpenStreetMap contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            opacity={0.7} // Darker map tiles
          />
          
          {clusters.map((cluster, index) => (
            <CircleMarker 
              key={index}
              center={[cluster.lat, cluster.lng]}
              pathOptions={{ 
                // Color logic: Red if any high risk cases, Orange if medium
                color: cluster.highRiskCount > 0 ? '#dc3545' : '#fd7e14', 
                fillColor: cluster.highRiskCount > 0 ? '#dc3545' : '#fd7e14',
                fillOpacity: 0.6 
              }}
              // Size logic: Bigger circle = More patients
              radius={10 + (cluster.count * 3)} 
            >
              <Popup className="glass-popup">
                <div className="text-center">
                    <strong className="text-dark">{cluster.name}</strong> <br/>
                    <span className="badge bg-secondary text-light border border-dark my-1">{cluster.count} Cases</span>
                    {cluster.highRiskCount > 0 && <span className="badge bg-danger ms-1">{cluster.highRiskCount} Critical</span>}
                    <hr style={{margin: "5px 0", borderColor: 'rgba(0,0,0,0.1)'}}/>
                    <small className="text-muted d-block">Recent: {cluster.patientNames.slice(0, 3).join(", ")}...</small>
                </div>
              </Popup>
            </CircleMarker>
          ))}
        </MapContainer>
        
        {/* Dark Gradient Overlay */}
        <div className="position-absolute bottom-0 start-0 end-0 p-2 text-center small text-white-50 bg-dark bg-opacity-75" style={{zIndex: 1000}}>
              Real-time data synced from CHW devices.
        </div>
      </div>
    </div>
  );
};
 
export default OutbreakMap;