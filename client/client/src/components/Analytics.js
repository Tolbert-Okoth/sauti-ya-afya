/* client/src/components/Analytics.js */
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { auth } from '../firebase';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { FaChartPie, FaMapMarkedAlt } from 'react-icons/fa';

const Analytics = () => {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      const token = await auth.currentUser.getIdToken();
      // Mock data if API fails or for demo visualization
      try {
        const res = await axios.get('http://localhost:5000/api/analytics', {
            headers: { Authorization: `Bearer ${token}` }
        });
        setData(res.data);
      } catch (e) {
         // Fallback for UI demo
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
                 { location: 'Nakuru', count: 10 }
             ]
         });
      }
    };
    fetchData();
  }, []);

  if (!data) return <div className="text-center mt-5 text-dark-brown animate-pulse">Loading Analytics...</div>;

  // Wellnessist Theme Colors
  const COLORS = ['#dc3545', '#ffc107', '#2ecc71']; // Red, Yellow, Green
  const riskData = data.risk.map(item => ({ name: item.risk_level, value: parseInt(item.count) }));
  const locData = data.location.map(item => ({ name: item.location, cases: parseInt(item.count) }));

  // Custom Glass Tooltip for Charts
  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      return (
        <div className="glass-card p-2 border-0 shadow-sm text-dark-brown" style={{background: 'rgba(255,255,255,0.9)'}}>
          <p className="mb-0 fw-bold">{label ? `${label}` : payload[0].name}</p>
          <p className="mb-0 text-accent">{`Count: ${payload[0].value}`}</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className="container-fluid p-0">
      <div className="d-flex align-items-center mb-4">
          <div className="bg-white rounded-circle p-2 me-3 shadow-sm text-accent">
            <FaChartPie size={20} />
          </div>
          <h3 className="fw-bold text-dark-brown mb-0">Epidemiology Analytics</h3>
      </div>

      <div className="row g-4">
        {/* Chart 1: Disease Burden by Risk */}
        <div className="col-md-6">
            <div className="glass-card h-100 d-flex flex-column">
                <div className="d-flex justify-content-between align-items-center mb-3">
                    <h6 className="fw-bold text-dark-brown mb-0">Risk Distribution</h6>
                    <small className="text-muted">Total Cases</small>
                </div>
                <div style={{ height: '300px', width: '100%' }}>
                    <ResponsiveContainer>
                        <PieChart>
                            <Pie 
                                data={riskData} 
                                cx="50%" 
                                cy="50%" 
                                innerRadius={60}
                                outerRadius={80} 
                                paddingAngle={5}
                                dataKey="value" 
                            >
                                {riskData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.name === 'High' ? '#dc3545' : entry.name === 'Medium' ? '#ffc107' : '#2ecc71'} />
                                ))}
                            </Pie>
                            <Tooltip content={<CustomTooltip />} />
                            <Legend iconType="circle" />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>

        {/* Chart 2: Hotspots by Location */}
        <div className="col-md-6">
            <div className="glass-card h-100 d-flex flex-column">
                <div className="d-flex justify-content-between align-items-center mb-3">
                    <h6 className="fw-bold text-dark-brown mb-0">Regional Hotspots</h6>
                    <FaMapMarkedAlt className="text-dark-brown opacity-50"/>
                </div>
                <div style={{ height: '300px', width: '100%' }}>
                    <ResponsiveContainer>
                        <BarChart data={locData}>
                            <XAxis dataKey="name" stroke="#636e72" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis allowDecimals={false} stroke="#636e72" fontSize={12} tickLine={false} axisLine={false}/>
                            <Tooltip content={<CustomTooltip />} cursor={{fill: 'rgba(255,255,255,0.2)'}} />
                            <Bar dataKey="cases" fill="#8d6e63" radius={[4, 4, 0, 0]} barSize={30} />
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