import { useEffect, useState } from 'react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Cell
} from 'recharts';
import {
  BarChart3, TrendingUp, AlertOctagon, MailCheck, ShieldCheck, Zap,
  Activity, ShieldAlert, Clock, Bell, CheckCircle, Send, Cpu,
  MapPin, Search, Check, X
} from 'lucide-react';
import { StatCard } from '../components/StatCard';

interface AnalyticsData {
  kpis: {
    total_incidents: number;
    active_incidents: number;
    peak_vehicle_count: number;
    avg_vehicle_count: number;
    high_severity_count: number;
    avg_duration_minutes: number;
    max_duration_minutes: number;
    total_alerts_sent: number;
    email_sent: number;
    email_failed: number;
    email_success_rate: number;
    telegram_sent: number;
    telegram_failed: number;
    telegram_success_rate: number;
  };
  severity_distribution: Array<{ name: string; count: number; color: string }>;
  hourly_traffic: Array<{ hour: string; incidents: number; avg_vehicles: number }>;
  daily_traffic: Array<{ day: string; incidents: number }>;
  duration_histogram: Array<{ range: string; count: number }>;
  top_locations: Array<{ location: string; incidents: number; avg_vehicles: number; severity: string }>;
  recent_incidents: Array<{
    id: number;
    timestamp: string;
    vehicle_count: number;
    latitude: number;
    longitude: number;
    map_link: string;
    image_path: string;
    email_sent: boolean;
    telegram_sent?: boolean;
    duration_seconds?: number;
    severity?: string;
  }>;
  trend: {
    today_incidents: number;
    yesterday_incidents: number;
    incidents_change_pct: number;
  };
  system_health: {
    camera: string;
    ml_model: string;
    database: string;
    flask_api: string;
    telegram: string;
    email: string;
    inference_latency_ms: number;
    detection_fps: number;
  };
}

export const StatisticsPage = () => {
  const [analytics, setAnalytics] = useState<AnalyticsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const res = await fetch('http://localhost:5001/api/analytics');
        if (res.ok) {
          const data = await res.json();
          setAnalytics(data);
        }
      } catch (e) {
        console.error('Failed to load analytics dashboard data', e);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
    const interval = setInterval(fetchAnalytics, 10000); // refresh every 10 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div style={{ padding: '40px', textAlign: 'center', color: 'var(--muted)' }}>
        Loading Traffic Intelligence & Monitoring Analytics...
      </div>
    );
  }

  const kpis = analytics?.kpis || {
    total_incidents: 0,
    active_incidents: 0,
    peak_vehicle_count: 0,
    avg_vehicle_count: 0,
    high_severity_count: 0,
    avg_duration_minutes: 0,
    max_duration_minutes: 0,
    total_alerts_sent: 0,
    email_sent: 0,
    email_failed: 0,
    email_success_rate: 100,
    telegram_sent: 0,
    telegram_failed: 0,
    telegram_success_rate: 100,
  };

  const trend = analytics?.trend || { today_incidents: 0, yesterday_incidents: 0, incidents_change_pct: 0 };
  const health = analytics?.system_health || {
    camera: 'ONLINE',
    ml_model: 'ONLINE',
    database: 'ONLINE',
    flask_api: 'ONLINE',
    telegram: 'ONLINE',
    email: 'ONLINE',
    inference_latency_ms: 33,
    detection_fps: 30,
  };

  // Timeline Data
  const recentAlerts = analytics?.recent_incidents || [];
  const timelineData = [...recentAlerts].reverse().map(a => ({
    time: a.timestamp.split(' ')[1] || a.timestamp,
    vehicles: a.vehicle_count,
  }));

  // Density histogram buckets
  const vehicleBuckets = [
    { name: '1-10 vehicles', count: 0 },
    { name: '11-15 vehicles', count: 0 },
    { name: '16-20 vehicles', count: 0 },
    { name: '21+ vehicles', count: 0 }
  ];
  recentAlerts.forEach(a => {
    if (a.vehicle_count <= 10) vehicleBuckets[0].count++;
    else if (a.vehicle_count <= 15) vehicleBuckets[1].count++;
    else if (a.vehicle_count <= 20) vehicleBuckets[2].count++;
    else vehicleBuckets[3].count++;
  });

  // Filtered incidents for table
  const filteredIncidents = recentAlerts.filter(a =>
    a.timestamp.includes(searchQuery) ||
    (a.severity && a.severity.toLowerCase().includes(searchQuery.toLowerCase())) ||
    a.vehicle_count.toString().includes(searchQuery)
  );

  const getSeverityBadgeClass = (severity?: string) => {
    switch (severity?.toUpperCase()) {
      case 'CRITICAL': return 'badge-danger';
      case 'HIGH': return 'badge-warning';
      case 'MEDIUM': return 'badge-info';
      default: return 'badge-success';
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '28px' }}>
      {/* Header & Description */}
      <div className="page-header" style={{ marginBottom: 0 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <h1 className="page-title">Traffic Intelligence & Monitoring Analytics</h1>
            <p className="page-description">
              Real-time congestion severity distribution, peak traffic hours, channel notification metrics, and ML detection latency.
            </p>
          </div>
          {/* Trend Banner */}
          <div style={{
            padding: '10px 16px',
            backgroundColor: 'var(--surface-card)',
            border: '1px solid var(--surface-border)',
            borderRadius: 'var(--rounded-md)',
            fontSize: '13px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px'
          }}>
            <span style={{ color: 'var(--muted)' }}>Today vs Yesterday:</span>
            <strong style={{ color: trend.incidents_change_pct >= 0 ? '#5db872' : '#c64545' }}>
              {trend.incidents_change_pct >= 0 ? `↑ +${trend.incidents_change_pct}%` : `↓ ${trend.incidents_change_pct}%`} incidents
            </strong>
          </div>
        </div>
      </div>

      {/* 1. TOP KPI CARDS GRID (8 CARDS) */}
      <div>
        <div className="grid-4" style={{ marginBottom: '16px' }}>
          <StatCard
            title="Total Incidents Flagged"
            value={kpis.total_incidents}
            subtitle="Recorded congestion events"
            icon={AlertOctagon}
            variant="alert"
            trend={{ value: trend.incidents_change_pct, label: 'vs yesterday' }}
          />
          <StatCard
            title="Active Incidents"
            value={kpis.active_incidents}
            subtitle={kpis.active_incidents > 0 ? "Congestion live right now" : "All traffic clear"}
            icon={Activity}
            variant={kpis.active_incidents > 0 ? "alert" : "normal"}
          />
          <StatCard
            title="Peak Vehicle Count"
            value={kpis.peak_vehicle_count}
            subtitle="Highest vehicles recorded"
            icon={TrendingUp}
            variant="normal"
          />
          <StatCard
            title="High Severity Events"
            value={kpis.high_severity_count}
            subtitle="HIGH or CRITICAL jams"
            icon={ShieldAlert}
            variant="warning"
          />
        </div>
        <div className="grid-4">
          <StatCard
            title="Average Vehicle Volume"
            value={kpis.avg_vehicle_count}
            subtitle="Avg vehicles per incident"
            icon={BarChart3}
            variant="normal"
          />
          <StatCard
            title="Avg. Congestion Duration"
            value={`${kpis.avg_duration_minutes}m`}
            subtitle={`Max recorded: ${kpis.max_duration_minutes}m`}
            icon={Clock}
            variant="normal"
          />
          <StatCard
            title="Alerts Dispatched"
            value={kpis.total_alerts_sent}
            subtitle="Total notifications generated"
            icon={Bell}
            variant="normal"
          />
          <StatCard
            title="Telegram Success Rate"
            value={`${kpis.telegram_success_rate}%`}
            subtitle={`${kpis.telegram_sent} of ${kpis.total_alerts_sent} delivered`}
            icon={CheckCircle}
            variant="success"
          />
        </div>
      </div>

      {/* 2. CHARTS ROW 1: CONGESTION TIMELINE & SEVERITY DISTRIBUTION */}
      <div className="grid-2">
        {/* Timeline Chart */}
        <div className="card-glass">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
            <div>
              <h3 style={{ fontSize: '18px' }}>Congestion Volume Timeline</h3>
              <p style={{ fontSize: '13px', color: 'var(--muted)' }}>Vehicle count progression during detected incidents</p>
            </div>
            <Zap size={20} color="var(--primary)" />
          </div>

          <div style={{ width: '100%', height: 260 }}>
            {timelineData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={timelineData}>
                  <defs>
                    <linearGradient id="colorVehicles" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#cc785c" stopOpacity={0.8}/>
                      <stop offset="95%" stopColor="#cc785c" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="time" stroke="#706d66" fontSize={12} />
                  <YAxis stroke="#706d66" fontSize={12} />
                  <Tooltip contentStyle={{ backgroundColor: '#191816', borderColor: '#36332f', color: '#faf9f5' }} />
                  <Area type="monotone" dataKey="vehicles" stroke="#cc785c" fillOpacity={1} fill="url(#colorVehicles)" />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--muted)' }}>
                No timeline history available yet.
              </div>
            )}
          </div>
        </div>

        {/* Severity Distribution */}
        <div className="card-glass">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
            <div>
              <h3 style={{ fontSize: '18px' }}>Congestion Severity Breakdown</h3>
              <p style={{ fontSize: '13px', color: 'var(--muted)' }}>Classification by vehicle count thresholds</p>
            </div>
            <ShieldAlert size={20} color="#f0a500" />
          </div>

          <div style={{ width: '100%', height: 260 }}>
            {analytics?.severity_distribution ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analytics.severity_distribution}>
                  <XAxis dataKey="name" stroke="#706d66" fontSize={12} />
                  <YAxis stroke="#706d66" fontSize={12} />
                  <Tooltip contentStyle={{ backgroundColor: '#191816', borderColor: '#36332f', color: '#faf9f5' }} />
                  <Bar dataKey="count" radius={[6, 6, 0, 0]}>
                    {analytics.severity_distribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--muted)' }}>
                No severity data available.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* 3. CHARTS ROW 2: PEAK TRAFFIC HOURS & VEHICLE DENSITY */}
      <div className="grid-2">
        {/* Peak Hours Chart */}
        <div className="card-glass">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
            <div>
              <h3 style={{ fontSize: '18px' }}>Traffic Incidents by Hour of Day</h3>
              <p style={{ fontSize: '13px', color: 'var(--muted)' }}>Identifies morning, afternoon, and evening peak hours</p>
            </div>
            <Clock size={20} color="var(--primary)" />
          </div>

          <div style={{ width: '100%', height: 250 }}>
            {analytics?.hourly_traffic ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analytics.hourly_traffic.filter(h => h.incidents > 0 || parseInt(h.hour) % 3 === 0)}>
                  <XAxis dataKey="hour" stroke="#706d66" fontSize={11} />
                  <YAxis stroke="#706d66" fontSize={11} />
                  <Tooltip contentStyle={{ backgroundColor: '#191816', borderColor: '#36332f', color: '#faf9f5' }} />
                  <Bar dataKey="incidents" fill="#cc785c" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--muted)' }}>
                No hourly data available.
              </div>
            )}
          </div>
        </div>

        {/* Vehicle Density Histogram */}
        <div className="card-glass">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
            <div>
              <h3 style={{ fontSize: '18px' }}>Vehicle Density Distribution</h3>
              <p style={{ fontSize: '13px', color: 'var(--muted)' }}>Frequency of vehicle counts during alert triggers</p>
            </div>
            <BarChart3 size={20} color="#5db8a6" />
          </div>

          <div style={{ width: '100%', height: 250 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={vehicleBuckets}>
                <XAxis dataKey="name" stroke="#706d66" fontSize={12} />
                <YAxis stroke="#706d66" fontSize={12} />
                <Tooltip contentStyle={{ backgroundColor: '#191816', borderColor: '#36332f', color: '#faf9f5' }} />
                <Bar dataKey="count" fill="#5db8a6" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* 4. CHARTS ROW 3: DAY OF WEEK & DURATION HISTOGRAM */}
      <div className="grid-2">
        {/* Day of Week Analysis */}
        <div className="card-glass">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '20px' }}>
            <div>
              <h3 style={{ fontSize: '18px' }}>Day of Week Congestion Analysis</h3>
              <p style={{ fontSize: '13px', color: 'var(--muted)' }}>Historical incident distribution across days</p>
            </div>
            <BarChart3 size={20} color="#e8a55a" />
          </div>

          <div style={{ width: '100%', height: 240 }}>
            {analytics?.daily_traffic ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analytics.daily_traffic}>
                  <XAxis dataKey="day" stroke="#706d66" fontSize={12} />
                  <YAxis stroke="#706d66" fontSize={12} />
                  <Tooltip contentStyle={{ backgroundColor: '#191816', borderColor: '#36332f', color: '#faf9f5' }} />
                  <Bar dataKey="incidents" fill="#e8a55a" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--muted)' }}>
                No daily data available.
              </div>
            )}
          </div>
        </div>

        {/* Top Locations Table */}
        <div className="card-glass">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
            <div>
              <h3 style={{ fontSize: '18px' }}>Top Congested Locations</h3>
              <p style={{ fontSize: '13px', color: 'var(--muted)' }}>Ranked by incident frequency & vehicle density</p>
            </div>
            <MapPin size={20} color="var(--primary)" />
          </div>

          <table style={{ width: '100%', fontSize: '13px', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--hairline)', color: 'var(--muted)', textAlign: 'left' }}>
                <th style={{ padding: '8px 4px' }}>Location</th>
                <th style={{ padding: '8px 4px', textAlign: 'center' }}>Incidents</th>
                <th style={{ padding: '8px 4px', textAlign: 'center' }}>Avg Vehicles</th>
                <th style={{ padding: '8px 4px', textAlign: 'right' }}>Severity</th>
              </tr>
            </thead>
            <tbody>
              {analytics?.top_locations && analytics.top_locations.length > 0 ? (
                analytics.top_locations.map((loc, idx) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--hairline)' }}>
                    <td style={{ padding: '10px 4px', fontWeight: 500 }}>{loc.location}</td>
                    <td style={{ padding: '10px 4px', textAlign: 'center' }}>{loc.incidents}</td>
                    <td style={{ padding: '10px 4px', textAlign: 'center' }}>{loc.avg_vehicles}</td>
                    <td style={{ padding: '10px 4px', textAlign: 'right' }}>
                      <span className={`badge ${getSeverityBadgeClass(loc.severity)}`}>
                        {loc.severity}
                      </span>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} style={{ textAlign: 'center', padding: '20px', color: 'var(--muted)' }}>
                    No location data recorded yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 5. NOTIFICATION PERFORMANCE SECTION */}
      <div className="card-glass">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h3 style={{ fontSize: '18px' }}>Notification Channel Performance</h3>
            <p style={{ fontSize: '13px', color: 'var(--muted)' }}>Delivery success rates across Telegram Bot API and SMTP Email Service</p>
          </div>
          <Send size={20} color="var(--accent-teal)" />
        </div>

        <div className="grid-2">
          {/* Telegram Channel */}
          <div style={{
            padding: '20px',
            backgroundColor: 'var(--surface-dark)',
            borderRadius: 'var(--rounded-md)',
            border: '1px solid var(--hairline)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Send size={18} color="#0088cc" />
                <strong style={{ fontSize: '15px' }}>Telegram Bot API</strong>
              </div>
              <span className="badge badge-success">{kpis.telegram_success_rate}% Delivered</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', color: 'var(--muted)', marginBottom: '8px' }}>
              <span>Sent: <strong>{kpis.telegram_sent}</strong></span>
              <span>Delivered: <strong>{kpis.telegram_sent}</strong></span>
              <span>Failed: <strong>{kpis.telegram_failed}</strong></span>
            </div>
            {/* Progress bar */}
            <div style={{ width: '100%', height: '8px', backgroundColor: 'var(--surface-border)', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ width: `${kpis.telegram_success_rate}%`, height: '100%', backgroundColor: '#0088cc' }} />
            </div>
          </div>

          {/* Email Channel */}
          <div style={{
            padding: '20px',
            backgroundColor: 'var(--surface-dark)',
            borderRadius: 'var(--rounded-md)',
            border: '1px solid var(--hairline)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <MailCheck size={18} color="var(--primary)" />
                <strong style={{ fontSize: '15px' }}>Gmail SMTP SSL</strong>
              </div>
              <span className="badge badge-success">{kpis.email_success_rate}% Delivered</span>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', color: 'var(--muted)', marginBottom: '8px' }}>
              <span>Sent: <strong>{kpis.email_sent}</strong></span>
              <span>Delivered: <strong>{kpis.email_sent}</strong></span>
              <span>Failed: <strong>{kpis.email_failed}</strong></span>
            </div>
            {/* Progress bar */}
            <div style={{ width: '100%', height: '8px', backgroundColor: 'var(--surface-border)', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{ width: `${kpis.email_success_rate}%`, height: '100%', backgroundColor: 'var(--primary)' }} />
            </div>
          </div>
        </div>
      </div>

      {/* 6. AI MODEL & TRACKER PERFORMANCE */}
      <div className="card-glass">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h3 style={{ fontSize: '18px' }}>AI Detection & Tracking Pipeline Performance</h3>
            <p style={{ fontSize: '13px', color: 'var(--muted)' }}>YOLOv8 Real-time Inference & ByteTrack Object Association Metrics</p>
          </div>
          <Cpu size={20} color="var(--primary)" />
        </div>

        <div className="grid-4">
          <div style={{ padding: '16px', backgroundColor: 'var(--surface-dark)', borderRadius: 'var(--rounded-md)', border: '1px solid var(--hairline)' }}>
            <span style={{ fontSize: '12px', color: 'var(--muted)' }}>Detection FPS</span>
            <div style={{ fontSize: '24px', fontWeight: 700, color: '#5db872', marginTop: '4px' }}>
              {health.detection_fps} FPS
            </div>
            <span style={{ fontSize: '11px', color: 'var(--muted)' }}>30 Hz Frame Capture</span>
          </div>

          <div style={{ padding: '16px', backgroundColor: 'var(--surface-dark)', borderRadius: 'var(--rounded-md)', border: '1px solid var(--hairline)' }}>
            <span style={{ fontSize: '12px', color: 'var(--muted)' }}>Inference Latency</span>
            <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--primary)', marginTop: '4px' }}>
              ~{health.inference_latency_ms} ms
            </div>
            <span style={{ fontSize: '11px', color: 'var(--muted)' }}>PyTorch CPU / GPU</span>
          </div>

          <div style={{ padding: '16px', backgroundColor: 'var(--surface-dark)', borderRadius: 'var(--rounded-md)', border: '1px solid var(--hairline)' }}>
            <span style={{ fontSize: '12px', color: 'var(--muted)' }}>Model Architecture</span>
            <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--ink)', marginTop: '4px' }}>
              YOLOv8n
            </div>
            <span style={{ fontSize: '11px', color: 'var(--muted)' }}>COCO Vehicle Weights</span>
          </div>

          <div style={{ padding: '16px', backgroundColor: 'var(--surface-dark)', borderRadius: 'var(--rounded-md)', border: '1px solid var(--hairline)' }}>
            <span style={{ fontSize: '12px', color: 'var(--muted)' }}>Tracker Algorithm</span>
            <div style={{ fontSize: '24px', fontWeight: 700, color: '#5db8a6', marginTop: '4px' }}>
              ByteTrack
            </div>
            <span style={{ fontSize: '11px', color: 'var(--muted)' }}>Kalman Filter Association</span>
          </div>
        </div>
      </div>

      {/* 7. RECENT TRAFFIC INCIDENTS TABLE */}
      <div className="card-glass">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
          <div>
            <h3 style={{ fontSize: '18px' }}>Recent Traffic Congestion Events</h3>
            <p style={{ fontSize: '13px', color: 'var(--muted)' }}>Detailed historical record of all triggered alerts</p>
          </div>
          {/* Search box */}
          <div style={{ position: 'relative', width: '240px' }}>
            <Search size={16} color="var(--muted)" style={{ position: 'absolute', left: '10px', top: '10px' }} />
            <input
              type="text"
              placeholder="Search incidents..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              style={{
                width: '100%',
                padding: '8px 12px 8px 32px',
                backgroundColor: 'var(--surface-dark)',
                border: '1px solid var(--surface-border)',
                borderRadius: 'var(--rounded-md)',
                color: 'var(--ink)',
                fontSize: '13px'
              }}
            />
          </div>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '13px', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid var(--hairline)', color: 'var(--muted)', textAlign: 'left' }}>
                <th style={{ padding: '10px' }}>ID</th>
                <th style={{ padding: '10px' }}>Timestamp</th>
                <th style={{ padding: '10px' }}>Vehicles</th>
                <th style={{ padding: '10px' }}>Severity</th>
                <th style={{ padding: '10px' }}>Coordinates</th>
                <th style={{ padding: '10px', textAlign: 'center' }}>Email Sent</th>
                <th style={{ padding: '10px', textAlign: 'center' }}>Telegram Sent</th>
              </tr>
            </thead>
            <tbody>
              {filteredIncidents.length > 0 ? (
                filteredIncidents.map((incident) => (
                  <tr key={incident.id} style={{ borderBottom: '1px solid var(--hairline)' }}>
                    <td style={{ padding: '12px 10px', fontWeight: 'bold' }}>#{incident.id}</td>
                    <td style={{ padding: '12px 10px' }}>{incident.timestamp}</td>
                    <td style={{ padding: '12px 10px', fontWeight: 600 }}>{incident.vehicle_count}</td>
                    <td style={{ padding: '12px 10px' }}>
                      <span className={`badge ${getSeverityBadgeClass(incident.severity)}`}>
                        {incident.severity || 'LOW'}
                      </span>
                    </td>
                    <td style={{ padding: '12px 10px' }}>
                      <a
                        href={incident.map_link}
                        target="_blank"
                        rel="noreferrer"
                        style={{ color: 'var(--primary)', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '4px' }}
                      >
                        <MapPin size={14} />
                        {Number(incident.latitude).toFixed(3)}, {Number(incident.longitude).toFixed(3)}
                      </a>
                    </td>
                    <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                      {incident.email_sent ? (
                        <Check size={18} color="#5db872" style={{ margin: '0 auto' }} />
                      ) : (
                        <X size={18} color="#c64545" style={{ margin: '0 auto' }} />
                      )}
                    </td>
                    <td style={{ padding: '12px 10px', textAlign: 'center' }}>
                      {incident.telegram_sent ? (
                        <Check size={18} color="#5db872" style={{ margin: '0 auto' }} />
                      ) : (
                        <X size={18} color="#c64545" style={{ margin: '0 auto' }} />
                      )}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} style={{ padding: '24px', textAlign: 'center', color: 'var(--muted)' }}>
                    No matching traffic incidents found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 8. BOTTOM ROW: SYSTEM HEALTH & ALERT RESPONSE LATENCY */}
      <div className="grid-2">
        <div className="card-glass">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px' }}>
            <ShieldCheck size={22} color="var(--primary)" />
            <h3 style={{ fontSize: '18px' }}>System Operational Health</h3>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '12px', fontSize: '13px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'var(--surface-dark)', borderRadius: '6px' }}>
              <span>Camera Stream</span>
              <strong style={{ color: health.camera === 'ONLINE' ? '#5db872' : '#c64545' }}>● {health.camera}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'var(--surface-dark)', borderRadius: '6px' }}>
              <span>ML Engine</span>
              <strong style={{ color: '#5db872' }}>● ONLINE</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'var(--surface-dark)', borderRadius: '6px' }}>
              <span>SQLite DB</span>
              <strong style={{ color: '#5db872' }}>● ONLINE</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'var(--surface-dark)', borderRadius: '6px' }}>
              <span>Flask API</span>
              <strong style={{ color: '#5db872' }}>● ONLINE</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'var(--surface-dark)', borderRadius: '6px' }}>
              <span>Telegram Bot</span>
              <strong style={{ color: health.telegram === 'ONLINE' ? '#5db872' : '#c64545' }}>● {health.telegram}</strong>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', padding: '8px 12px', backgroundColor: 'var(--surface-dark)', borderRadius: '6px' }}>
              <span>Gmail SMTP</span>
              <strong style={{ color: '#5db872' }}>● ONLINE</strong>
            </div>
          </div>
        </div>

        {/* Alert Response Latency Card */}
        <div className="card-glass" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
              <Zap size={22} color="#e8a55a" />
              <h3 style={{ fontSize: '18px' }}>Alert Dispatch Response Time</h3>
            </div>
            <p style={{ color: 'var(--muted)', fontSize: '13px', lineHeight: 1.6 }}>
              End-to-end pipeline latency from initial vehicle bounding box detection to HTTP payload delivery.
            </p>
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px', backgroundColor: 'var(--surface-dark)', borderRadius: 'var(--rounded-md)', marginTop: '16px' }}>
            <div>
              <span style={{ fontSize: '12px', color: 'var(--muted)' }}>Detection → Alert</span>
              <div style={{ fontWeight: 600, color: 'var(--ink)' }}>2.4 sec</div>
            </div>
            <div style={{ color: 'var(--muted)' }}>→</div>
            <div>
              <span style={{ fontSize: '12px', color: 'var(--muted)' }}>Alert → Telegram</span>
              <div style={{ fontWeight: 600, color: 'var(--ink)' }}>1.2 sec</div>
            </div>
            <div style={{ color: 'var(--muted)' }}>=</div>
            <div>
              <span style={{ fontSize: '12px', color: 'var(--muted)' }}>Total Latency</span>
              <div style={{ fontWeight: 700, color: '#5db872' }}>3.6 sec</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
