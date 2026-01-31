/* client/src/components/AdminDashboard.js */
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { auth } from '../firebase';
import { useNavigate } from 'react-router-dom';
import { FaUserShield, FaTrash, FaUserPlus, FaIdBadge } from 'react-icons/fa';

const AdminDashboard = () => {
  const [users, setUsers] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    fetchUsers();
  }, []);

  const fetchUsers = async () => {
    try {
      const token = await auth.currentUser.getIdToken();
      const res = await axios.get('http://localhost:5000/api/users', {
        headers: { Authorization: `Bearer ${token}` }
      });
      setUsers(res.data);
    } catch (err) {
      console.error("Error fetching users:", err);
    }
  };

  const deleteUser = async (id) => {
    if(!window.confirm("Are you sure? This removes their access immediately.")) return;
    try {
      const token = await auth.currentUser.getIdToken();
      await axios.delete(`http://localhost:5000/api/users/${id}`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      fetchUsers(); 
    } catch (err) {
      alert("Failed to delete user.");
    }
  };

  const handleAddUser = () => {
    if(window.confirm("To add a new user, you must be redirected to the Signup page. This will log you out of the Admin panel. Continue?")) {
        auth.signOut(); 
        navigate('/signup'); 
    }
  };

  return (
    <div className="container p-0">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
            <h3 className="fw-bold text-white"><FaUserShield className="me-2"/> System Administration</h3>
            <p className="text-white-50 opacity-75 mb-0">Manage access and permissions</p>
        </div>
        
        <button className="btn btn-primary shadow-sm rounded-pill px-4" onClick={handleAddUser}>
            <FaUserPlus className="me-2"/> Add New Staff
        </button>
      </div>

      <div className="glass-card overflow-hidden">
        <div className="px-4 py-3 border-bottom border-secondary d-flex justify-content-between align-items-center" style={{background: 'rgba(0,0,0,0.2)'}}>
          <span className="fw-bold text-white">Authorized Personnel</span>
          <span className="badge bg-white text-dark shadow-sm">{users.length} Users</span>
        </div>
        
        <div className="table-responsive">
            <table className="table table-hover mb-0 align-middle text-white" style={{'--bs-table-bg': 'transparent', '--bs-table-hover-bg': 'rgba(255,255,255,0.05)'}}>
            <thead className="text-white-50 small text-uppercase" style={{background: 'rgba(255,255,255,0.05)'}}>
                <tr>
                <th className="px-4 border-0">Role</th>
                <th className="border-0">Identity</th>
                <th className="border-0">Jurisdiction</th>
                <th className="px-4 border-0 text-end">Actions</th>
                </tr>
            </thead>
            <tbody>
                {users.map(u => (
                <tr key={u.id} style={{borderBottom: '1px solid rgba(255,255,255,0.1)'}}>
                    <td className="px-4">
                        <span className={`badge rounded-pill ${u.role === 'DOCTOR' ? 'bg-primary' : u.role === 'ADMIN' ? 'bg-secondary' : 'bg-success'} shadow-sm`}>
                            {u.role}
                        </span>
                    </td>
                    <td>
                        <div className="d-flex align-items-center">
                            <div className="bg-light rounded-circle p-2 me-3 text-dark shadow-sm opacity-75">
                                <FaIdBadge />
                            </div>
                            <div>
                                <div className="fw-bold text-white">{u.email || "No Email"}</div>
                                <small className="text-white-50 font-monospace" style={{fontSize: '0.7rem'}}>{u.firebase_uid.substring(0, 12)}...</small>
                            </div>
                        </div>
                    </td>
                    <td className="text-white fw-bold">
                        {u.county || <span className="text-white-50 fst-italic fw-normal">All Counties (Global)</span>}
                    </td>
                    <td className="px-4 text-end">
                    {u.role !== 'ADMIN' && (
                        <button onClick={() => deleteUser(u.id)} className="btn btn-sm btn-outline-danger hover-shadow">
                            <FaTrash /> Remove
                        </button>
                    )}
                    </td>
                </tr>
                ))}
            </tbody>
            </table>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;