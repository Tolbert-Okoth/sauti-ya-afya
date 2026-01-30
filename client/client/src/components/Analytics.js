/* client/src/components/Analytics.js */
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { auth } from '../firebase';
import config from '../config'; // ✅ Connected to shared config
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { FaChartPie, FaMapMarkedAlt, FaUsers } from 'react-icons/fa';

const Analytics = () => {
  const [data, setData] = useState(null);

  // 🎨 ROBUST COLOR MAPPING
  // Ensures consistent colors regardless of API data order
  const getColor = (risk) => {
      const r = (risk || '').toLowerCase();
      if (r.includes('high') || r.includes('pneumonia')) return '#dc3545'; // Red
      if (r.includes('medium') || r.includes('asthma')) return '#ffc107'; // Yellow
      if (r.includes('low') || r.includes('normal') || r.includes('stable')) return '#2ecc71'; // Green
      return '#6c757d'; // Grey for unknown
  };

  useEffect(() => {
    const fetchData = async () => {
      try {
        const token = await auth.currentUser.getIdToken();
        
        // 🔗 SECURE CONNECTION: Fetches from your live backend
        const res = await axios.get(`${config.API_BASE_URL}/analytics`, {
            headers: { Authorization: `Bearer ${token}` }
        });
        
        setData(res.data);
      } catch (e) {
         console.warn("Analytics API unavailable, using Demo Data:", e);
         // 🛡️ FALLBACK: Keeps dashboard alive during offline demos
         setData({
             risk: [
                 { risk_level: 'High', count: 12 },
                 { risk_level: 'Medium', count: 25 },
                 { risk_level: 'Low', count: 45 }
             ],
             location: [
                 { location: 'Nairobi', count: 30 },
                 { location: 'Mombasa', count: 20 },
                 { location: 'Kisumu', count: 15 },
                 { location: 'Eldoret', count: 12 }, // Your location included
                 { location: 'Nakuru', count: 10 }
             ]
         });
      }
    };
    fetchData();
  }, []);

  if (!data) return <div className="text-center mt-5 text-dark-brown animate-pulse">Loading Epidemiology Data...</div>;

  // Process Data for Recharts
  const riskData = data.risk.map(item => ({ name: item.risk_level, value: parseInt(item.count) }));
  const locData = data.location.map(item => ({ name: item.location, cases: parseInt(item.count) }));

  // 🧮 Dynamic Total Calculation
  const totalCases = riskData.reduce((acc, curr) => acc + curr.value, 0);

  // Custom Glass Tooltip
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-card p-2 border-0 shadow-sm text-dark-brown" style={{background: 'rgba(255,255,255,0.95)'}}>
          <p className="mb-0 fw-bold">{label ? `${label}` : payload[0].name}</p>
          <p className="mb-0 text-accent fw-bold">{`Cases: ${payload[0].value}`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="container-fluid p-0">
      
      {/* 📊 HEADER SUMMARY */}
      <div className="d-flex justify-content-between align-items-center mb-4">
          <div className="d-flex align-items-center">
              <div className="bg-white rounded-circle p-2 me-3 shadow-sm text-accent">
                <FaChartPie size={20} />
              </div>
              <div>
                  <h3 className="fw-bold text-dark-brown mb-0">Epidemiology Analytics</h3>
                  <small className="text-muted">Real-time surveillance data</small>
              </div>
          </div>
          
          {/* Total Cases Big Number */}
          <div className="glass-card px-4 py-2 text-center d-none d-md-block">
              <div className="small text-muted text-uppercase fw-bold">Total Screenings</div>
              <div className="h4 fw-bold text-dark-brown mb-0"><FaUsers className="me-2 opacity-50"/>{totalCases}</div>
          </div>
      </div>

      <div className="row g-4">
        {/* Chart 1: Disease Burden (Donut) */}
        <div className="col-md-6 col-lg-5">
            <div className="glass-card h-100 d-flex flex-column">
                <div className="d-flex justify-content-between align-items-center mb-3">
                    <h6 className="fw-bold text-dark-brown mb-0">Risk Distribution</h6>
                    <small className="text-muted">By Severity</small>
                </div>
                <div style={{ height: '300px', width: '100%' }}>
                    <ResponsiveContainer>
                        <PieChart>
                            <Pie 
                                data={riskData} 
                                cx="50%" 
                                cy="50%" 
                                innerRadius={60}
                                outerRadius={85} 
                                paddingAngle={5}
                                dataKey="value" 
                                stroke="none"
                            >
                                {riskData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={getColor(entry.name)} />
                                ))}
                            </Pie>
                            <Tooltip content={<CustomTooltip />} />
                            <Legend verticalAlign="bottom" height={36} iconType="circle" />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>

        {/* Chart 2: Regional Hotspots (Bar) */}
        <div className="col-md-6 col-lg-7">
            <div className="glass-card h-100 d-flex flex-column">
                <div className="d-flex justify-content-between align-items-center mb-3">
                    <h6 className="fw-bold text-dark-brown mb-0">Regional Hotspots</h6>
                    <FaMapMarkedAlt className="text-dark-brown opacity-50"/>
                </div>
                <div style={{ height: '300px', width: '100%' }}>
                    <ResponsiveContainer>
                        <BarChart data={locData} margin={{top: 10, right: 30, left: 0, bottom: 0}}>
                            <XAxis dataKey="name" stroke="#636e72" fontSize={11} tickLine={false} axisLine={false} />
                            <YAxis allowDecimals={false} stroke="#636e72" fontSize={11} tickLine={false} axisLine={false}/>
                            <Tooltip content={<CustomTooltip />} cursor={{fill: 'rgba(0,0,0,0.05)'}} />
                            <Bar dataKey="cases" fill="#8d6e63" radius={[4, 4, 0, 0]} barSize={40}>
                                {locData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={index % 2 === 0 ? '#8d6e63' : '#a1887f'} />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;