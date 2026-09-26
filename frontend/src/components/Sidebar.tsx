import { NavLink } from 'react-router-dom';
import { LayoutDashboard, BarChart3, AlertTriangle, Settings, LogOut, ShieldCheck } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export const Sidebar = () => {
  const { user, logout } = useAuth();

  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <img src="/logo.png" alt="TCS Logo" className="sidebar-logo-img" style={{ height: '42px', objectFit: 'contain' }} />
        <div>
          <div className="sidebar-title">TCS Portal</div>
          <div className="sidebar-subtitle">Traffic Congestion System</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <NavLink to="/" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <LayoutDashboard size={18} />
          <span>Live Feed</span>
        </NavLink>

        <NavLink to="/statistics" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <BarChart3 size={18} />
          <span>Statistics</span>
        </NavLink>

        <NavLink to="/alerts" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <AlertTriangle size={18} />
          <span>Alert History</span>
        </NavLink>

        <NavLink to="/settings" className={({ isActive }) => `nav-item ${isActive ? 'active' : ''}`}>
          <Settings size={18} />
          <span>Settings</span>
        </NavLink>
      </nav>

      {user && (
        <div className="sidebar-user">
          <div className="user-badge">
            <div className="user-avatar">
              {user.username.charAt(0).toUpperCase()}
            </div>
            <div>
              <div className="user-name">{user.username}</div>
              <div className="user-role" style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <ShieldCheck size={12} color="var(--primary)" />
                {user.role}
              </div>
            </div>
          </div>
          <button className="btn-logout" onClick={logout} title="Log out">
            <LogOut size={18} />
          </button>
        </div>
      )}
    </aside>
  );
};
