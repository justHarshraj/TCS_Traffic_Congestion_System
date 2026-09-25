import { useState, useEffect, useRef } from 'react';
import {
  Camera,
  AlertTriangle,
  CheckCircle2,
  Activity,
  MapPin,
  Power,
  MailCheck,
  Cpu,
  Layers,
  ShieldCheck,
  Smartphone,
  Radio
} from 'lucide-react';
import { StatCard } from '../components/StatCard';
import { fetchWithAuth } from '../api/client';

const formatCoord = (val: any, decimals = 3) => {
  const num = typeof val === 'number' ? val : parseFloat(val);
  return isNaN(num) ? '0.000' : num.toFixed(decimals);
};

const getSeverityInfo = (count: number, threshold: number) => {
  if (count <= threshold) {
    return { text: 'LOW', badgeClass: 'badge-success', variant: 'success' as const };
  }
  if (count <= Math.floor(threshold * 1.5)) {
    return { text: 'MEDIUM', badgeClass: 'badge-warning', variant: 'warning' as const };
  }
  if (count <= threshold * 2) {
    return { text: 'HIGH', badgeClass: 'badge-danger', variant: 'alert' as const };
  }
  return { text: 'CRITICAL', badgeClass: 'badge-danger', variant: 'alert' as const };
};

export const DashboardPage = () => {
  const [vehicleCount, setVehicleCount] = useState(0);
  const [isCongested, setIsCongested] = useState(false);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [location, setLocation] = useState({ lat: 23.0225, lon: 72.5714 });
  const [showPopup, setShowPopup] = useState(false);
  const [settings, setSettings] = useState({ threshold: 10, receiver_email: '' });

  const lastAlertTimeRef = useRef(0);

  const handleToggleCamera = async () => {
    try {
      const newState = !isCameraActive;
      const response = await fetchWithAuth('/api/camera/toggle', {
        method: 'POST',
        body: JSON.stringify({ active: newState })
      });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.error || 'Unable to change the camera state.');
      }
      setIsCameraActive(data.camera_active);
      setCameraError(null);
    } catch (err) {
      console.error("Failed to toggle camera:", err);
      setIsCameraActive(false);
      setCameraError(err instanceof Error ? err.message : 'Unable to change camera state.');
    }
  };

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
              setTimeout(() => setShowPopup(false), 6000);
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

    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  const severityInfo = getSeverityInfo(vehicleCount, settings.threshold);

  return (
    <div>
      {/* Toast Notification Banner */}
      {showPopup && (
        <div
          className="toast-notify"
          style={{
            position: 'fixed',
            top: '24px',
            right: '24px',
            backgroundColor: 'var(--surface-card)',
            color: 'var(--ink)',
            padding: '16px 24px',
            borderRadius: 'var(--rounded-lg)',
            display: 'flex',
            alignItems: 'center',
            gap: '16px',
            boxShadow: '0 12px 32px rgba(0,0,0,0.5)',
            border: '1px solid var(--success)',
            zIndex: 9999,
            maxWidth: '420px',
          }}
        >
          <div style={{
            backgroundColor: 'var(--success-glow)',
            color: 'var(--success)',
            width: '42px',
            height: '42px',
            borderRadius: 'var(--rounded-full)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0
          }}>
            <MailCheck size={22} />
          </div>
          <div>
            <div style={{ fontWeight: 600, fontSize: '15px' }}>Email + Telegram Alert Dispatched</div>
            <div style={{ color: 'var(--muted)', fontSize: '13px', marginTop: '2px' }}>
              Traffic authority notified via Email & Telegram with live snapshot.
            </div>
          </div>
        </div>
      )}

      {/* Page Header */}
      <div className="page-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
        <div>
          <h1 className="page-title">Live Vision Feed</h1>
          <p className="page-description">Real-time video analytics, vehicle counting, and automated jam detection console.</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span className="status-pill success" style={{ padding: '6px 14px', fontSize: '13px' }}>
            <span className="live-dot" /> SYSTEM ONLINE
          </span>
        </div>
      </div>

      {/* Top KPI Cards Row (4 Cards) */}
      <div className="grid-4">
        <StatCard
          title="Traffic Flow State"
          value={isCongested ? "Congested" : "Normal Flow"}
          subtitle={isCongested ? "Heavy jam detected at intersection" : "Flowing within normal parameters"}
          icon={isCongested ? AlertTriangle : CheckCircle2}
          variant={isCongested ? 'alert' : 'success'}
        />
        <StatCard
          title="Live Vehicle Count"
          value={vehicleCount}
          subtitle={`Alert Threshold: ${settings.threshold} vehicles`}
          icon={Activity}
          variant={isCongested ? 'alert' : 'normal'}
        />
        <StatCard
          title="Congestion Level"
          value={severityInfo.text}
          subtitle="Based on live vehicle density"
          icon={Radio}
          variant={severityInfo.variant}
        />
        <StatCard
          title="Camera Stream"
          value={isCameraActive ? "LIVE · 30 FPS" : "OFFLINE"}
          subtitle={isCameraActive ? "Laptop Cam #0 (Active)" : "Stream currently paused"}
          icon={Camera}
          variant={isCameraActive ? 'success' : 'normal'}
        />
      </div>

      {/* Main Console Area: 70% Video Feed + 30% Live Detection Panel */}
      <div style={{ display: 'grid', gridTemplateColumns: '3fr 1fr', gap: '24px', marginBottom: '32px' }}>
        {/* Left: Video Feed Console */}
        <div className="card-glass" style={{ padding: '0', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
          {/* Header Bar */}
          <div style={{
            padding: '14px 20px',
            backgroundColor: 'var(--surface-dark)',
            borderBottom: '1px solid var(--surface-border)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
              <div style={{ display: 'flex', gap: '6px' }}>
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--error)' }} />
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--warning)' }} />
                <div style={{ width: '12px', height: '12px', borderRadius: '50%', backgroundColor: 'var(--success)' }} />
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontFamily: 'var(--font-code)', fontSize: '13px', fontWeight: 600, color: 'var(--ink)' }}>
                  CAMERA 1 — LAPTOP (INDEX #0)
                </span>
                {isCameraActive ? (
                  <span className="badge badge-success">
                    <span className="live-dot" style={{ marginRight: '6px' }} /> LIVE
                  </span>
                ) : (
                  <span className="badge badge-danger">OFFLINE</span>
                )}
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <button
                onClick={handleToggleCamera}
                className="btn-secondary"
                style={{
                  borderColor: isCameraActive ? 'var(--surface-border)' : 'var(--error)',
                  color: isCameraActive ? 'var(--ink)' : 'var(--error)',
                  height: '36px',
                  padding: '0 14px',
                  fontSize: '13px'
                }}
              >
                <Power size={14} />
                {isCameraActive ? 'Pause Stream' : 'Start Camera'}
              </button>
            </div>
          </div>

          {/* Video Container */}
          <div style={{
            flex: 1,
            minHeight: '480px',
            backgroundColor: '#050505',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            position: 'relative'
          }}>
            <img
              src="http://localhost:5001/video_feed"
              alt="Live Camera Stream"
              style={{ width: '100%', height: '480px', objectFit: 'contain' }}
              onError={(e) => {
                e.currentTarget.style.display = 'none';
                const errElem = document.getElementById('video-err-fallback');
                if (errElem) errElem.style.display = 'flex';
              }}
              onLoad={(e) => {
                e.currentTarget.style.display = 'block';
                const errElem = document.getElementById('video-err-fallback');
                if (errElem) errElem.style.display = 'none';
              }}
            />

            <div
              id="video-err-fallback"
              style={{
                display: 'none',
                flexDirection: 'column',
                alignItems: 'center',
                color: 'var(--muted)',
                gap: '12px'
              }}
            >
              <Camera size={48} style={{ opacity: 0.4 }} />
              <p style={{ fontFamily: 'var(--font-code)', fontSize: '14px' }}>
                {cameraError || 'Camera feed offline or disconnected.'}
              </p>
              <p style={{ fontSize: '12px', color: 'var(--muted-soft)' }}>
                Click "Start Camera" above to initiate device video capture.
              </p>
            </div>
          </div>

          {/* Camera Status Bar */}
          <div className="cam-status-bar">
            <div>
              <span style={{ color: 'var(--ink)', fontWeight: 600 }}>FPS:</span> {isCameraActive ? '30' : '0'} &nbsp;│&nbsp;
              <span style={{ color: 'var(--ink)', fontWeight: 600 }}>LATENCY:</span> {isCameraActive ? '~33ms' : '0ms'} &nbsp;│&nbsp;
              <span style={{ color: 'var(--ink)', fontWeight: 600 }}>RES:</span> 1280x720
            </div>
            <div>
              <span style={{ color: 'var(--ink)', fontWeight: 600 }}>MODEL:</span> YOLOv8n &nbsp;│&nbsp;
              <span style={{ color: 'var(--ink)', fontWeight: 600 }}>TRACKER:</span> ByteTrack
            </div>
          </div>
        </div>

        {/* Right: Live Detection Panel */}
        <div className="card-glass" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', paddingBottom: '12px', borderBottom: '1px solid var(--surface-border)' }}>
            <Activity size={20} style={{ color: 'var(--primary)' }} />
            <h3 style={{ fontSize: '16px', fontWeight: 700 }}>Live Detection</h3>
          </div>

          {/* Vehicle Count Box */}
          <div style={{ background: 'var(--surface-dark)', padding: '16px', borderRadius: 'var(--rounded-md)', border: '1px solid var(--surface-border)' }}>
            <div style={{ fontSize: '12px', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600 }}>
              Vehicles Counted
            </div>
            <div style={{ fontSize: '40px', fontWeight: 800, fontFamily: 'var(--font-display)', color: isCongested ? 'var(--error)' : 'var(--ink)', marginTop: '4px' }}>
              {vehicleCount}
            </div>
            <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '2px' }}>
              Alert Limit: <strong style={{ color: 'var(--ink)' }}>{settings.threshold} vehicles</strong>
            </div>
          </div>

          {/* Severity Badge Box */}
          <div style={{ background: 'var(--surface-dark)', padding: '16px', borderRadius: 'var(--rounded-md)', border: '1px solid var(--surface-border)' }}>
            <div style={{ fontSize: '12px', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: '8px' }}>
              Severity Level
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <span className={`badge ${severityInfo.badgeClass}`} style={{ fontSize: '13px', padding: '6px 14px' }}>
                {severityInfo.text}
              </span>
              <span style={{ fontSize: '12px', color: 'var(--muted)' }}>
                {isCongested ? 'Jam Threshold Exceeded' : 'Normal Traffic'}
              </span>
            </div>
          </div>

          {/* Performance Box */}
          <div style={{ background: 'var(--surface-dark)', padding: '16px', borderRadius: 'var(--rounded-md)', border: '1px solid var(--surface-border)' }}>
            <div style={{ fontSize: '12px', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: '10px' }}>
              Engine Metrics
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', fontSize: '13px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--muted)' }}>Processing Speed:</span>
                <span style={{ fontFamily: 'var(--font-code)', fontWeight: 600 }}>{isCameraActive ? '30 FPS' : '0 FPS'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--muted)' }}>Frame Latency:</span>
                <span style={{ fontFamily: 'var(--font-code)', fontWeight: 600 }}>{isCameraActive ? '33 ms' : 'N/A'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--muted)' }}>Detection Model:</span>
                <span style={{ fontWeight: 600, color: 'var(--primary)' }}>YOLOv8 Nano</span>
              </div>
            </div>
          </div>

          {/* Location Box */}
          <div style={{ background: 'var(--surface-dark)', padding: '16px', borderRadius: 'var(--rounded-md)', border: '1px solid var(--surface-border)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 600, marginBottom: '8px' }}>
              <MapPin size={14} style={{ color: 'var(--primary)' }} /> Camera Location
            </div>
            <div style={{ fontFamily: 'var(--font-code)', fontSize: '13px', color: 'var(--ink)' }}>
              Lat: {formatCoord(location.lat)}, Lon: {formatCoord(location.lon)}
            </div>
            <div style={{ fontSize: '11px', color: 'var(--muted-soft)', marginTop: '4px' }}>
              IP Geolocated Intersection Point
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Grid: Camera Sources (Left) + System Info (Right) */}
      <div className="grid-2">
        {/* Camera Sources Selector */}
        <div className="card-glass">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <Camera size={20} style={{ color: 'var(--primary)' }} />
            <h3 style={{ fontSize: '18px', fontWeight: 700 }}>Camera Sources</h3>
          </div>

          <div className="cam-source-item active-source">
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontWeight: 600, color: 'var(--ink)' }}>Laptop Camera (Index #0)</span>
                <span className="badge badge-success">PRIMARY</span>
              </div>
              <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '2px' }}>
                Built-in OpenCV VideoCapture(0) device stream
              </div>
            </div>
            <button
              onClick={handleToggleCamera}
              className="btn-secondary"
              style={{ height: '34px', fontSize: '12px', padding: '0 12px' }}
            >
              {isCameraActive ? 'Pause' : 'Start'}
            </button>
          </div>

          <div className="cam-source-item" style={{ opacity: 0.65 }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontWeight: 600, color: 'var(--ink)' }}>IP Phone Camera (RTSP / HTTP)</span>
                <span className="badge badge-info">OPTIONAL</span>
              </div>
              <div style={{ fontSize: '12px', color: 'var(--muted)', marginTop: '2px' }}>
                Mobile phone camera stream integration (Scaffolding UI)
              </div>
            </div>
            <span style={{ fontSize: '12px', color: 'var(--muted)', fontFamily: 'var(--font-code)' }}>
              ○ READY
            </span>
          </div>
        </div>

        {/* Live System Info Card */}
        <div className="card-glass">
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '20px' }}>
            <ShieldCheck size={20} style={{ color: 'var(--primary)' }} />
            <h3 style={{ fontSize: '18px', fontWeight: 700 }}>Live System Info</h3>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
            <div style={{ background: 'var(--surface-dark)', padding: '14px', borderRadius: 'var(--rounded-md)', border: '1px solid var(--surface-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--muted)', fontWeight: 600 }}>
                <Cpu size={14} style={{ color: 'var(--primary)' }} /> INFERENCE ENGINE
              </div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--ink)', marginTop: '4px' }}>
                YOLOv8 Nano (COCO)
              </div>
            </div>

            <div style={{ background: 'var(--surface-dark)', padding: '14px', borderRadius: 'var(--rounded-md)', border: '1px solid var(--surface-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--muted)', fontWeight: 600 }}>
                <Layers size={14} style={{ color: 'var(--accent-teal)' }} /> TRACKING ALGORITHM
              </div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--ink)', marginTop: '4px' }}>
                ByteTrack Algorithm
              </div>
            </div>

            <div style={{ background: 'var(--surface-dark)', padding: '14px', borderRadius: 'var(--rounded-md)', border: '1px solid var(--surface-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--muted)', fontWeight: 600 }}>
                <Smartphone size={14} style={{ color: 'var(--accent-amber)' }} /> DISPATCH CHANNELS
              </div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--ink)', marginTop: '4px' }}>
                Email (SMTP) + Telegram Bot
              </div>
            </div>

            <div style={{ background: 'var(--surface-dark)', padding: '14px', borderRadius: 'var(--rounded-md)', border: '1px solid var(--surface-border)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px', color: 'var(--muted)', fontWeight: 600 }}>
                <Radio size={14} style={{ color: 'var(--success)' }} /> BACKEND REFRESH
              </div>
              <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--ink)', marginTop: '4px' }}>
                1000ms Polling Loop
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

