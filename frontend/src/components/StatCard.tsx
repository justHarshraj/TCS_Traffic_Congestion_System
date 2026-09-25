import type { LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  variant?: 'normal' | 'alert' | 'success' | 'warning';
  trend?: {
    value: number;
    label?: string;
  };
}

export const StatCard = ({ title, value, subtitle, icon: Icon, variant = 'normal', trend }: StatCardProps) => {
  return (
    <div className={`stat-card ${variant}`}>
      <div className="stat-header">
        <span className="stat-title">{title}</span>
        <div className="stat-icon">
          <Icon size={20} />
        </div>
      </div>
      <div className="stat-value" style={{ display: 'flex', alignItems: 'baseline', gap: '8px' }}>
        <span>{value}</span>
        {trend !== undefined && (
          <span
            style={{
              fontSize: '12px',
              padding: '2px 6px',
              borderRadius: '12px',
              fontWeight: 600,
              backgroundColor: trend.value >= 0 ? 'rgba(93, 184, 114, 0.15)' : 'rgba(198, 69, 69, 0.15)',
              color: trend.value >= 0 ? '#5db872' : '#c64545',
            }}
          >
            {trend.value >= 0 ? `↑ ${trend.value}%` : `↓ ${Math.abs(trend.value)}%`}
          </span>
        )}
      </div>
      {subtitle && <div className="stat-subtitle">{subtitle}</div>}
    </div>
  );
};

