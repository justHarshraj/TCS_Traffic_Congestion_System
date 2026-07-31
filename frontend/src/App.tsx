import { useState, useEffect, useRef } from 'react';
import { Camera, AlertTriangle, CheckCircle2, Activity, MapPin, Power, MailCheck } from 'lucide-react';
import type { LucideIcon } from 'lucide-react';
import './index.css';

const TopNav = () => (
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
      <a href="#">Product</a>
      <a href="#">Analytics</a>
      <a href="#">Alerts</a>
      <button className="btn-primary" style={{ marginLeft: '12px' }}>System Settings</button>
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
    <div className="card-feature" style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <h4 style={{ fontFamily: 'var(--font-body)', fontSize: '16px', fontWeight: 500, color: 'var(--ink)' }}>{title}</h4>
        <div style={{ color: type === 'alert' ? 'var(--error)' : 'var(--success)' }}>
          <Icon size={20} />
        </div>
      </div>
      
      <div>
        <div style={{ fontSize: '36px', fontFamily: 'var(--font-display)', color: type === 'alert' ? 'var(--error)' : 'var(--ink)' }}>
          {value}
        </div>
        <div style={{ fontSize: '13px', color: 'var(--muted)', marginTop: '4px' }}>
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
    <div className="card-code" style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: '500px' }}>
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


function App() {
  const [vehicleCount, setVehicleCount] = useState(0);
  const [isCongested, setIsCongested] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [location, setLocation] = useState({ lat: 23.0225, lon: 72.5714 });
  const [showPopup, setShowPopup] = useState(false);
  
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
        }
      } catch (err) {
        console.error("Could not fetch status from backend:", err);
      }
    };

    fetchStatus(); // initial fetch
    const interval = setInterval(fetchStatus, 1000); // Poll every second
    return () => clearInterval(interval);
  }, []);
  
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

      <TopNav />
      
      <main className="container" style={{ flex: 1, padding: 'var(--spacing-section) var(--spacing-xl)' }}>
        
        {/* Header */}
        <div style={{ marginBottom: 'var(--spacing-xxl)' }}>
          <div className="badge-pill" style={{ marginBottom: '16px' }}>Live Monitoring System</div>
          <h1>Traffic Congestion System</h1>
          <p style={{ fontSize: '18px', color: 'var(--muted)', marginTop: '16px', maxWidth: '600px' }}>
            Real-time vehicle tracking and congestion detection using YOLOv8 bounding boxes and ByteTrack.
          </p>
        </div>
        
        {/* Dashboard Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: 'var(--spacing-xl)' }}>
          
          {/* Main Feed */}
          <div>
            <CameraFeed 
              vehicleCount={vehicleCount} 
              isCongested={isCongested} 
              isCameraActive={isCameraActive}
              onToggleCamera={handleToggleCamera}
            />
          </div>
          
          {/* Sidebar Stats */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--spacing-lg)' }}>
            
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
              subtitle="Threshold: 10 vehicles"
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
              <div className="card-coral" style={{ marginTop: 'auto' }}>
                <h4 style={{ color: 'var(--on-primary)', marginBottom: '8px' }}>Alert Triggered</h4>
                <p style={{ fontSize: '14px', marginBottom: '24px', opacity: 0.9 }}>Email notifications and system alerts have been sent to local authorities.</p>
                <button className="btn-secondary" style={{ width: '100%' }}>View Incident Report</button>
              </div>
            )}
            
          </div>
        </div>
      </main>
      
      {/* Footer */}
      <footer style={{ backgroundColor: 'var(--surface-dark)', padding: 'var(--spacing-xxl)', color: 'var(--on-dark-soft)', marginTop: 'var(--spacing-section)' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between' }}>
          <div>
            <div style={{ fontWeight: 'bold', color: 'var(--on-dark)', marginBottom: '16px' }}>Anthropic Inspired TCS</div>
            <p style={{ fontSize: '14px' }}>Traffic Congestion System</p>
          </div>
          <div style={{ fontSize: '14px' }}>
            &copy; 2026 Traffic Congestion System
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
