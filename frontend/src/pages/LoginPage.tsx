import React, { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { Lock, Mail, AlertCircle, ArrowRight, ShieldCheck, KeyRound, RefreshCw, CheckCircle2, ArrowLeft } from 'lucide-react';

export const LoginPage = () => {
  const { login, verifyOtp, resendOtp } = useAuth();
  
  // Step state: 'credentials' | 'otp'
  const [step, setStep] = useState<'credentials' | 'otp'>('credentials');
  
  // Form fields
  const [email, setEmail] = useState('admin@tcs.local');
  const [password, setPassword] = useState('Admin@TCS123');
  const [otpCode, setOtpCode] = useState('');
  
  // Status and messages
  const [targetEmail, setTargetEmail] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [infoMessage, setInfoMessage] = useState<string | null>(null);
  const [devOtpHint, setDevOtpHint] = useState<string | null>(null);
  
  // Loading states
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isResending, setIsResending] = useState(false);
  
  // Timers
  const [timerSeconds, setTimerSeconds] = useState(300); // 5 min OTP expiry
  const [resendCooldown, setResendCooldown] = useState(0);

  // OTP Countdown Timer Effect
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    if (step === 'otp' && timerSeconds > 0) {
      interval = setInterval(() => {
        setTimerSeconds((prev) => prev - 1);
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [step, timerSeconds]);

  // Resend Cooldown Timer Effect
  useEffect(() => {
    let interval: ReturnType<typeof setInterval> | null = null;
    if (resendCooldown > 0) {
      interval = setInterval(() => {
        setResendCooldown((prev) => prev - 1);
      }, 1000);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [resendCooldown]);

  const handleCredentialsSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setInfoMessage(null);
    setIsSubmitting(true);

    const res = await login(email, password);
    setIsSubmitting(false);

    if (res.success) {
      if (res.otp_required) {
        setStep('otp');
        setTargetEmail(res.sent_to || res.email || email);
        setInfoMessage(res.message || 'OTP code sent via Gmail SMTP.');
        if (res.dev_otp) {
          setDevOtpHint(res.dev_otp);
        }
        setTimerSeconds(300);
        setResendCooldown(30);
      }
    } else {
      setError(res.error || 'Invalid email or password');
    }
  };

  const handleOtpSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!otpCode || otpCode.trim().length !== 6) {
      setError('Please enter the 6-digit OTP code sent to your Gmail.');
      return;
    }

    setError(null);
    setIsSubmitting(true);

    const res = await verifyOtp(email, otpCode.trim());
    setIsSubmitting(false);

    if (!res.success) {
      setError(res.error || 'Invalid or expired OTP code.');
    }
  };

  const handleResendOtp = async () => {
    if (resendCooldown > 0 || isResending) return;
    
    setError(null);
    setIsResending(true);
    
    const res = await resendOtp(email);
    setIsResending(false);

    if (res.success) {
      setInfoMessage(res.message || 'A new OTP code has been sent via Gmail SMTP.');
      if (res.dev_otp) {
        setDevOtpHint(res.dev_otp);
      }
      setTimerSeconds(300);
      setResendCooldown(30);
    } else {
      setError(res.error || 'Failed to resend OTP.');
    }
  };

  const formatTimer = (secs: number) => {
    const mins = Math.floor(secs / 60);
    const remainderSecs = secs % 60;
    return `${mins.toString().padStart(2, '0')}:${remainderSecs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="login-container">
      {/* Left Visual Panel */}
      <div className="login-left">
        <div className="radar-wrapper">
          <div className="radar-ring" />
          <div className="radar-ring" />
          <div className="radar-ring" />
        </div>

        <div className="login-brand" style={{ display: 'flex', alignItems: 'center', gap: '14px' }}>
          <img src="/logo.png" alt="TCS Logo" style={{ height: '56px', width: 'auto', filter: 'drop-shadow(0 4px 12px rgba(0, 0, 0, 0.4))' }} />
          <div>
            <h3 style={{ fontSize: '22px', fontWeight: 700, margin: 0, letterSpacing: '0.5px' }}>TCS AI Vision</h3>
            <span style={{ fontSize: '13px', color: 'var(--muted)' }}>Traffic Congestion System</span>
          </div>
        </div>

        <div className="login-hero-text">
          <h1>Intelligent Real-Time Traffic Management</h1>
          <p>
            Powered by YOLOv8 deep learning and ByteTrack object identification. 
            Monitors vehicle density, detects traffic jams instantly, and dispatches multi-channel alerts.
          </p>
        </div>

        <div className="login-footer-info">
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <ShieldCheck size={16} color="var(--primary)" />
            <span>Gmail SMTP OTP Authentication</span>
          </div>
          <div>•</div>
          <div>System v2.4 Active</div>
        </div>
      </div>

      {/* Right Card Panel */}
      <div className="login-right">
        <div className="login-card">
          
          {step === 'credentials' ? (
            /* STEP 1: Credentials Form */
            <>
              <h2>Sign In to TCS</h2>
              <p>Enter your system operator credentials to trigger Gmail SMTP OTP confirmation.</p>

              {error && (
                <div style={{
                  backgroundColor: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid #ef4444',
                  color: '#f87171',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  marginBottom: '20px',
                  fontSize: '14px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px'
                }}>
                  <AlertCircle size={18} />
                  <span>{error}</span>
                </div>
              )}

              <form onSubmit={handleCredentialsSubmit}>
                <div className="form-group">
                  <label className="form-label">Email Address</label>
                  <div style={{ position: 'relative' }}>
                    <input
                      type="email"
                      className="input-field"
                      style={{ paddingLeft: '42px' }}
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="operator@tcs.local"
                      required
                    />
                    <Mail size={18} color="var(--muted)" style={{ position: 'absolute', left: '14px', top: '14px' }} />
                  </div>
                </div>

                <div className="form-group" style={{ marginBottom: '28px' }}>
                  <label className="form-label">Password</label>
                  <div style={{ position: 'relative' }}>
                    <input
                      type="password"
                      className="input-field"
                      style={{ paddingLeft: '42px' }}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      placeholder="••••••••"
                      required
                    />
                    <Lock size={18} color="var(--muted)" style={{ position: 'absolute', left: '14px', top: '14px' }} />
                  </div>
                </div>

                <button type="submit" className="btn-primary" disabled={isSubmitting} style={{ width: '100%' }}>
                  {isSubmitting ? 'Verifying Credentials...' : 'Continue to Gmail OTP'}
                  {!isSubmitting && <ArrowRight size={18} />}
                </button>
              </form>

              <div style={{
                marginTop: '24px',
                paddingTop: '20px',
                borderTop: '1px solid var(--hairline)',
                fontSize: '12px',
                color: 'var(--muted)',
                textAlign: 'center'
              }}>
                Default Admin Login: <strong style={{ color: 'var(--ink)' }}>admin@tcs.local</strong> / <strong style={{ color: 'var(--ink)' }}>Admin@TCS123</strong>
              </div>
            </>
          ) : (
            /* STEP 2: Gmail SMTP OTP Verification Screen */
            <>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                <span style={{
                  background: 'rgba(56, 189, 248, 0.15)',
                  color: '#38bdf8',
                  padding: '4px 10px',
                  borderRadius: '12px',
                  fontSize: '11px',
                  fontWeight: 600,
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '5px'
                }}>
                  <ShieldCheck size={13} /> Gmail SMTP 2FA
                </span>
              </div>

              <h2>Enter Verification Code</h2>
              <p>We've sent a 6-digit OTP code to <strong style={{ color: '#38bdf8' }}>{targetEmail}</strong> via Gmail SMTP.</p>

              {infoMessage && (
                <div style={{
                  backgroundColor: 'rgba(16, 185, 129, 0.15)',
                  border: '1px solid #10b981',
                  color: '#34d399',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  marginBottom: '18px',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px'
                }}>
                  <CheckCircle2 size={18} />
                  <span>{infoMessage}</span>
                </div>
              )}

              {error && (
                <div style={{
                  backgroundColor: 'rgba(239, 68, 68, 0.15)',
                  border: '1px solid #ef4444',
                  color: '#f87171',
                  padding: '12px 16px',
                  borderRadius: '8px',
                  marginBottom: '18px',
                  fontSize: '13px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '10px'
                }}>
                  <AlertCircle size={18} />
                  <span>{error}</span>
                </div>
              )}

              {devOtpHint && (
                <div style={{
                  backgroundColor: 'rgba(245, 158, 11, 0.15)',
                  border: '1px dashed #f59e0b',
                  color: '#fbbf24',
                  padding: '10px 14px',
                  borderRadius: '8px',
                  marginBottom: '20px',
                  fontSize: '13px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <span>💡 <strong>Dev Quick OTP:</strong></span>
                  <code style={{ fontSize: '16px', fontWeight: 800, letterSpacing: '2px', background: '#1e293b', padding: '2px 8px', borderRadius: '4px' }}>
                    {devOtpHint}
                  </code>
                </div>
              )}

              <form onSubmit={handleOtpSubmit}>
                <div className="form-group" style={{ marginBottom: '24px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <label className="form-label" style={{ margin: 0 }}>6-Digit OTP Code</label>
                    <span style={{ fontSize: '12px', color: timerSeconds < 60 ? '#f87171' : 'var(--muted)', fontWeight: 600 }}>
                      Expires in: {formatTimer(timerSeconds)}
                    </span>
                  </div>
                  
                  <div style={{ position: 'relative' }}>
                    <input
                      type="text"
                      className="input-field"
                      style={{
                        paddingLeft: '44px',
                        letterSpacing: '8px',
                        fontSize: '22px',
                        fontWeight: 700,
                        textAlign: 'center',
                        color: '#38bdf8'
                      }}
                      maxLength={6}
                      value={otpCode}
                      onChange={(e) => setOtpCode(e.target.value.replace(/\D/g, ''))}
                      placeholder="••••••"
                      autoFocus
                      required
                    />
                    <KeyRound size={20} color="#0284c7" style={{ position: 'absolute', left: '14px', top: '14px' }} />
                  </div>
                </div>

                <button
                  type="submit"
                  className="btn-primary"
                  disabled={isSubmitting || otpCode.length !== 6 || timerSeconds === 0}
                  style={{ width: '100%', marginBottom: '16px' }}
                >
                  {isSubmitting ? 'Verifying OTP Code...' : 'Verify OTP & Access Dashboard'}
                  {!isSubmitting && <CheckCircle2 size={18} />}
                </button>
              </form>

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '16px', paddingTop: '16px', borderTop: '1px solid var(--hairline)' }}>
                <button
                  type="button"
                  onClick={() => {
                    setStep('credentials');
                    setError(null);
                    setInfoMessage(null);
                  }}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: 'var(--muted)',
                    cursor: 'pointer',
                    fontSize: '13px',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '5px'
                  }}
                >
                  <ArrowLeft size={14} /> Back to Sign In
                </button>

                <button
                  type="button"
                  onClick={handleResendOtp}
                  disabled={resendCooldown > 0 || isResending}
                  style={{
                    background: 'none',
                    border: 'none',
                    color: resendCooldown > 0 ? 'var(--muted)' : '#38bdf8',
                    cursor: resendCooldown > 0 ? 'not-allowed' : 'pointer',
                    fontSize: '13px',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '5px',
                    fontWeight: 600
                  }}
                >
                  <RefreshCw size={14} className={isResending ? 'spin' : ''} />
                  {resendCooldown > 0 ? `Resend code in ${resendCooldown}s` : 'Resend Code'}
                </button>
              </div>
            </>
          )}

        </div>
      </div>
    </div>
  );
};
