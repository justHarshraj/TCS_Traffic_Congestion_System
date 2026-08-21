import { useState, useEffect, useRef } from 'react';
import { Camera, AlertTriangle, CheckCircle2, Activity, MapPin, Power, MailCheck, Database, CheckCircle, XCircle, ExternalLink } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import './index.css';

const TopNav = ({ activeTab, onTabChange, onSettingsClick }: { activeTab: string; onTabChange: (tab: string) => void; onSettingsClick: () => void }) => (
  <nav style={{ 
    height: '64px', 
    backgroundColor: 'var(--canvas)',
    borderBottom: '1px solid var(--hairline)',
    display: 'flex',
    alignItems: 'center',
    padding: '0 var(--spacing-xl)',
    justifyContent: 'space-between'
  }}>
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
      <div style={{ width: '24px', height: '24px', backgroundColor: 'var(--ink)', borderRadius: 'var(--rounded-full)', position: 'relative' }}>
         <div style={{ position: 'absolute', top: '50%', left: '50%', transform: 'translate(-50%, -50%)', width: '12px', height: '12px', backgroundColor: 'var(--canvas)', borderRadius: '50%' }} />
      </div>
      <span style={{ fontFamily: 'var(--font-display)', fontSize: '22px', fontWeight: 'bold' }}>TCS Dashboard</span>
    </div>
    <div style={{ display: 'flex', gap: '24px', alignItems: 'center' }}>
      <a href="#" onClick={(e) => { e.preventDefault(); onTabChange('dashboard'); }} style={{ color: activeTab === 'dashboard' ? 'var(--primary)' : 'var(--ink)' }}>Dashboard</a>
      <a href="#" onClick={(e) => { e.preventDefault(); onTabChange('alerts'); }} style={{ color: activeTab === 'alerts' ? 'var(--primary)' : 'var(--ink)' }}>Alert History</a>
      <button className="btn-primary" style={{ marginLeft: '12px' }} onClick={onSettingsClick}>System Settings</button>
    </div>
  </nav>
);

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle: string;
  icon: LucideIcon;
  type?: 'normal' | 'alert';
}

const MetricCard = ({ title, value, subtitle, icon: Icon, type = 'normal' }: MetricCardProps) => {
  return (
    <div className="card-feature" style={{ display: 'flex', flexDirection: 'column', gap: '20px', padding: '24px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <h4 style={{ fontFamily: 'var(--font-body)', fontSize: '18px', fontWeight: 600, color: 'var(--ink)' }}>{title}</h4>
        <div style={{ color: type === 'alert' ? 'var(--error)' : 'var(--success)' }}>
          <Icon size={28} />
        </div>
      </div>
      
      <div>
        <div style={{ fontSize: '36px', fontFamily: 'var(--font-display)', color: type === 'alert' ? 'var(--error)' : 'var(--ink)', fontWeight: 600 }}>
          {value}
        </div>
        <div style={{ fontSize: '15px', color: 'var(--muted)', marginTop: '8px' }}>
          {subtitle}
        </div>
      </div>
    </div>
  );
};

interface CameraFeedProps {
  vehicleCount: number;
  isCongested: boolean;
  isCameraActive: boolean;
  onToggleCamera: () => void;
}

const CameraFeed = ({ vehicleCount, isCongested, isCameraActive, onToggleCamera }: CameraFeedProps) => {
  return (
    <div className="card-code" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '450px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px', borderBottom: '1px solid var(--surface-dark-elevated)', paddingBottom: '16px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{ display: 'flex', gap: '6px' }}>
            <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--error)' }} />
            <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--warning)' }} />
            <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--success)' }} />
          </div>
          <span style={{ color: 'var(--on-dark-soft)', fontFamily: 'var(--font-code)', fontSize: '14px' }}>camera_01_feed.py</span>
        </div>
        
        <div style={{ display: 'flex', gap: '16px', alignItems: 'center' }}>
          <button 
            onClick={onToggleCamera}
            style={{ 
              backgroundColor: isCameraActive ? 'var(--surface-dark-elevated)' : 'var(--error)', 
              color: isCameraActive ? 'var(--on-dark)' : 'var(--on-primary)',
              border: 'none',
              borderRadius: 'var(--rounded-md)',
              padding: '6px 12px',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: 'pointer',
              fontFamily: 'var(--font-body)',
              fontSize: '13px',
              fontWeight: 500
            }}
          >
            <Power size={14} />
            {isCameraActive ? 'Turn Off' : 'Turn On'}
          </button>
          
          <div className={isCongested ? "badge-alert" : "badge-success"} style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            {isCongested ? <AlertTriangle size={14} /> : <CheckCircle2 size={14} />}
            {isCongested ? "CONGESTED" : "NORMAL"}
          </div>
        </div>
      </div>

        {/* Real Video Feed Area */}
      <div style={{ 
        flex: 1, 
        backgroundColor: 'var(--surface-dark-soft)',
        borderRadius: 'var(--rounded-md)',
        position: 'relative',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        overflow: 'hidden'
      }}>
        <img 
          src="http://localhost:5001/video_feed" 
          alt="Live Camera Feed"
          style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          onError={(e) => {
            e.currentTarget.style.display = 'none';
            document.getElementById('video-error')!.style.display = 'block';
          }}
          onLoad={(e) => {
            e.currentTarget.style.display = 'block';
            document.getElementById('video-error')!.style.display = 'none';
          }}
        />
        <div id="video-error" style={{ textAlign: 'center', color: 'var(--on-dark-soft)' }}>
          <Camera size={48} style={{ margin: '0 auto', marginBottom: '16px', opacity: 0.5 }} />
          <p style={{ fontFamily: 'var(--font-code)' }}>Waiting for video stream connection...</p>
          <p style={{ fontSize: '12px', marginTop: '8px', opacity: 0.7 }}>Ensure Python server is running on port 5001</p>
        </div>
      </div>
      
      {/* Footer Info */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '16px', paddingTop: '16px', borderTop: '1px solid var(--surface-dark-elevated)', color: 'var(--on-dark-soft)' }}>
        <span>Frame Rate: 30 FPS</span>
        <span>Resolution: 1920x1080</span>
      </div>
    </div>
  );
};

// ===== Alert Data Types =====
interface AlertRecord {
  id: number;
  timestamp: string;
  vehicle_count: number;
  latitude: number;
  longitude: number;
  map_link: string;
  image_path: string;
  email_sent: boolean;
}

// ===== Image Modal / Lightbox =====
const ImageModal = ({ src, onClose }: { src: string; onClose: () => void }) => (
  <div className="image-modal-overlay" onClick={onClose}>
    <button className="image-modal-close" onClick={onClose}>×</button>
    <img
      className="image-modal-content"
      src={src}
      alt="Congestion snapshot"
      onClick={(e) => e.stopPropagation()}
    />
  </div>
);

// ===== Alert History Component =====
const AlertHistory = ({ alerts }: { alerts: AlertRecord[] }) => {
  const [modalImage, setModalImage] = useState<string | null>(null);

  const getImageUrl = (imagePath: string) => {
    // image_path is like "congestion_images/congestion_2026-08-21_16-30-00.jpg"
    // We need just the filename part to pass to our API
    const filename = imagePath.split('/').pop() || imagePath;
    return `http://localhost:5001/api/alerts/images/${filename}`;
  };

  return (
    <>
      {modalImage && <ImageModal src={modalImage} onClose={() => setModalImage(null)} />}

      <section className="alert-history-section">
        <div className="alert-history-header">
          <h2>Alert History</h2>
          <div className="alert-count-badge">
            <Database size={14} />
            {alerts.length} {alerts.length === 1 ? 'Record' : 'Records'}
          </div>
        </div>

        <div className="alert-table-wrapper">
          {alerts.length === 0 ? (
            <div className="alert-empty-state">
              <Database size={48} className="empty-icon" />
              <p>No congestion alerts recorded yet.</p>
              <p style={{ fontSize: '13px', marginTop: '4px', color: 'var(--muted-soft)' }}>
                Alerts will appear here when traffic congestion is detected.
              </p>
            </div>
          ) : (
            <div className="alert-table-scroll">
              <table className="alert-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Date / Time</th>
                    <th>Vehicles</th>
                    <th>Location</th>
                    <th>Map</th>
                    <th>Image</th>
                    <th>Email</th>
                  </tr>
                </thead>
                <tbody>
                  {alerts.map((alert) => (
                    <tr key={alert.id}>
                      <td className="cell-id">{alert.id}</td>
                      <td className="cell-timestamp">{alert.timestamp}</td>
                      <td className="cell-vehicle-count">{alert.vehicle_count}</td>
                      <td className="cell-coords">
                        {alert.latitude.toFixed(4)}, {alert.longitude.toFixed(4)}
                      </td>
                      <td className="cell-map-link">
                        {alert.map_link ? (
                          <a href={alert.map_link} target="_blank" rel="noopener noreferrer">
                            Open Map <ExternalLink size={12} />
                          </a>
                        ) : (
                          <span style={{ color: 'var(--muted-soft)', fontSize: '12px' }}>N/A</span>
                        )}
                      </td>
                      <td>
                        <img
                          className="alert-thumbnail"
                          src={getImageUrl(alert.image_path)}
                          alt={`Alert #${alert.id}`}
                          onClick={() => setModalImage(getImageUrl(alert.image_path))}
                          onError={(e) => {
                            e.currentTarget.style.display = 'none';
                          }}
                        />
                      </td>
                      <td>
                        {alert.email_sent ? (
                          <span className="status-sent">
                            <CheckCircle size={12} /> Sent
                          </span>
                        ) : (
                          <span className="status-failed">
                            <XCircle size={12} /> Failed
                          </span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>
    </>
  );
};

const SettingsModal = ({ 
  isOpen, 
  onClose, 
  initialThreshold, 
  initialEmail, 
  onSave 
}: { 
  isOpen: boolean; 
  onClose: () => void; 
  initialThreshold: number; 
  initialEmail: string; 
  onSave: (threshold: number, email: string) => void;
}) => {
  const [threshold, setThreshold] = useState(initialThreshold);
  const [email, setEmail] = useState(initialEmail);

  useEffect(() => {
    setThreshold(initialThreshold);
    setEmail(initialEmail);
  }, [initialThreshold, initialEmail, isOpen]);

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '400px', padding: '24px' }}>
        <h3 style={{ marginBottom: '24px', fontFamily: 'var(--font-display)', fontSize: '20px' }}>System Settings</h3>
        
        <div style={{ marginBottom: '16px' }}>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold', fontSize: '14px' }}>Congestion Threshold (vehicles)</label>
          <input 
            type="number" 
            value={threshold} 
            onChange={(e) => setThreshold(Number(e.target.value))}
            style={{ width: '100%', padding: '10px', borderRadius: 'var(--rounded-md)', border: '1px solid var(--hairline)' }}
          />
        </div>

        <div style={{ marginBottom: '24px' }}>
          <label style={{ display: 'block', marginBottom: '8px', fontWeight: 'bold', fontSize: '14px' }}>Notification Email</label>
          <input 
            type="email" 
            value={email} 
            onChange={(e) => setEmail(e.target.value)}
            style={{ width: '100%', padding: '10px', borderRadius: 'var(--rounded-md)', border: '1px solid var(--hairline)' }}
          />
        </div>

        <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
          <button className="btn-secondary" onClick={onClose}>Cancel</button>
          <button className="btn-primary" onClick={() => {
            onSave(threshold, email);
            onClose();
          }}>Save Settings</button>
        </div>
      </div>
    </div>
  );
};

function App() {
  const [vehicleCount, setVehicleCount] = useState(0);
  const [isCongested, setIsCongested] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [location, setLocation] = useState({ lat: 23.0225, lon: 72.5714 });
  const [showPopup, setShowPopup] = useState(false);
  const [alerts, setAlerts] = useState<AlertRecord[]>([]);
  const [activeTab, setActiveTab] = useState('dashboard');
  
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [settings, setSettings] = useState({ threshold: 10, receiver_email: 'rajharsh.23.cse@iite.indusuni.ac.in' });
  
  // Use a ref to hold the last alert time to avoid closure capture issues in setInterval
  const lastAlertTimeRef = useRef(0);
  
  // Toggle camera function
  const handleToggleCamera = async () => {
    try {
      const newState = !isCameraActive;
      setIsCameraActive(newState); // Optimistic UI update
      
      await fetch('http://localhost:5001/api/camera/toggle', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: newState })
      });
    } catch (err) {
      console.error("Failed to toggle camera:", err);
    }
  };

  // Fetch live data from backend API
  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const response = await fetch('http://localhost:5001/api/status');
        if (response.ok) {
          const data = await response.json();
          setVehicleCount(data.vehicle_count);
          setIsCongested(data.is_congested);
          if (data.camera_active !== undefined) {
            setIsCameraActive(data.camera_active);
          }
          
          if (data.last_alert_time > lastAlertTimeRef.current) {
            const wasInitialLoad = (lastAlertTimeRef.current === 0);
            lastAlertTimeRef.current = data.last_alert_time;
            
            if (!wasInitialLoad) {
              setShowPopup(true);
              setTimeout(() => setShowPopup(false), 5000); // Hide after 5 seconds
            }
          }
          
          if (data.latitude && data.longitude) {
            setLocation({ lat: data.latitude, lon: data.longitude });
          }
          
          if (data.settings) {
            setSettings(data.settings);
          }
        }
      } catch (err) {
        console.error("Could not fetch status from backend:", err);
      }
    };

    fetchStatus(); // initial fetch
    const interval = setInterval(fetchStatus, 1000); // Poll every second
    return () => clearInterval(interval);
  }, []);

  // Fetch alert history from backend
  useEffect(() => {
    const fetchAlerts = async () => {
      try {
        const response = await fetch('http://localhost:5001/api/alerts');
        if (response.ok) {
          const data = await response.json();
          setAlerts(data.alerts || []);
        }
      } catch (err) {
        console.error("Could not fetch alerts from backend:", err);
      }
    };

    fetchAlerts(); // initial fetch
    const interval = setInterval(fetchAlerts, 10000); // Refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);
  const handleSaveSettings = async (threshold: number, receiver_email: string) => {
    try {
      const response = await fetch('http://localhost:5001/api/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold, receiver_email })
      });
      if (response.ok) {
        const data = await response.json();
        setSettings(data.settings);
      }
    } catch (err) {
      console.error("Failed to save settings", err);
    }
  };


  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', position: 'relative' }}>
      
      {/* Toast Notification Popup */}
      <div style={{
        position: 'fixed',
        top: showPopup ? '24px' : '-100px',
        left: '50%',
        transform: 'translateX(-50%)',
        backgroundColor: 'var(--canvas)',
        color: 'var(--ink)',
        padding: '16px 24px',
        borderRadius: 'var(--rounded-lg)',
        display: 'flex',
        alignItems: 'center',
        gap: '16px',
        boxShadow: '0 12px 32px rgba(0,0,0,0.12), 0 4px 8px rgba(0,0,0,0.06)',
        border: '1px solid var(--hairline)',
        transition: 'all 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)',
        zIndex: 9999,
        fontFamily: 'var(--font-body)',
        width: '90%',
        maxWidth: '400px',
        opacity: showPopup ? 1 : 0,
      }}>
        <div style={{ 
          backgroundColor: 'var(--success)', 
          color: 'var(--on-primary)', 
          width: '40px', 
          height: '40px', 
          borderRadius: 'var(--rounded-full)', 
          display: 'flex', 
          alignItems: 'center', 
          justifyContent: 'center',
          flexShrink: 0
        }}>
          <MailCheck size={20} />
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <span style={{ fontWeight: 600, fontSize: '15px' }}>Report Sent Successfully</span>
          <span style={{ color: 'var(--muted)', fontSize: '13px' }}>Traffic authority has been notified.</span>
        </div>
        <button 
          onClick={() => setShowPopup(false)}
          style={{ 
            marginLeft: 'auto', 
            background: 'none', 
            border: 'none', 
            color: 'var(--muted)', 
            cursor: 'pointer',
            padding: '4px',
            fontSize: '18px',
            lineHeight: 1
          }}>
          ×
        </button>
      </div>

      <TopNav activeTab={activeTab} onTabChange={setActiveTab} onSettingsClick={() => setIsSettingsOpen(true)} />
      
      <main className="container" style={{ padding: 'var(--spacing-xxl) var(--spacing-xl)', position: 'relative' }}>
        
        {activeTab === 'dashboard' ? (
          <>
            {/* Dashboard Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 400px', gap: 'var(--spacing-xl)', alignItems: 'stretch' }}>
              
              {/* Main Feed */}
              <div style={{ minWidth: 0, height: '100%' }}>
                <CameraFeed 
                  vehicleCount={vehicleCount} 
                  isCongested={isCongested} 
                  isCameraActive={isCameraActive}
                  onToggleCamera={handleToggleCamera}
                />
              </div>
              
              {/* Sidebar Stats */}
              <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: '100%', gap: 'var(--spacing-lg)' }}>
                
                <MetricCard 
                  title="Current Status" 
                  value={isCongested ? "Congested" : "Normal Flow"} 
                  subtitle={isCongested ? "Traffic jam detected at intersection." : "Traffic is flowing smoothly."}
                  icon={isCongested ? AlertTriangle : CheckCircle2}
                  type={isCongested ? 'alert' : 'normal'}
                />
                
                <MetricCard 
                  title="Vehicles Detected" 
                  value={vehicleCount} 
                  subtitle={`Threshold: ${settings.threshold} vehicles`}
                  icon={Activity}
                  type={isCongested ? 'alert' : 'normal'}
                />
                
                <MetricCard 
                  title="Location" 
                  value="Live Tracker" 
                  subtitle={`Lat: ${location.lat}, Lon: ${location.lon}`}
                  icon={MapPin}
                />
                
                {/* Action Card */}
                {isCongested && (
                  <div className="card-coral" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    <h4 style={{ color: 'var(--on-primary)', marginBottom: '8px' }}>Alert Triggered</h4>
                    <p style={{ fontSize: '14px', marginBottom: '24px', opacity: 0.9 }}>Email notifications and system alerts have been sent to local authorities.</p>
                    <button className="btn-secondary" style={{ width: '100%' }}>View Incident Report</button>
                  </div>
                )}
                
              </div>
            </div>
          </>
        ) : (
          /* Alert History Section */
          <AlertHistory alerts={alerts} />
        )}
        
      </main>

      <SettingsModal 
        isOpen={isSettingsOpen} 
        onClose={() => setIsSettingsOpen(false)} 
        initialThreshold={settings.threshold}
        initialEmail={settings.receiver_email}
        onSave={handleSaveSettings}
      />
      

    </div>
  );
}

export default App;
