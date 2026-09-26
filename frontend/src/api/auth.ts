import { fetchWithAuth } from './client';

export interface User {
  id: number;
  username: string;
  email: string;
  role: string;
}

export interface LoginResponse {
  success: boolean;
  otp_required?: boolean;
  email?: string;
  sent_to?: string;
  smtp_sent?: boolean;
  message?: string;
  token?: string;
  user?: User;
  error?: string;
  dev_otp?: string;
}

export async function loginApi(email: string, password: string): Promise<LoginResponse> {
  try {
    const res = await fetch('http://localhost:5001/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    return await res.json();
  } catch {
    return { success: false, error: 'Network error. Is backend running?' };
  }
}

export async function verifyOtpApi(email: string, otp: string): Promise<LoginResponse> {
  try {
    const res = await fetch('http://localhost:5001/api/auth/verify-otp', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, otp }),
    });
    return await res.json();
  } catch {
    return { success: false, error: 'Network error. Could not connect to verification server.' };
  }
}

export async function resendOtpApi(email: string): Promise<{ success: boolean; message?: string; error?: string; dev_otp?: string }> {
  try {
    const res = await fetch('http://localhost:5001/api/auth/resend-otp', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email }),
    });
    return await res.json();
  } catch {
    return { success: false, error: 'Network error while requesting new OTP.' };
  }
}

export async function getMeApi(): Promise<{ success: boolean; user?: User; error?: string }> {
  try {
    const res = await fetchWithAuth('/api/auth/me');
    if (!res.ok) return { success: false };
    return await res.json();
  } catch {
    return { success: false };
  }
}

export async function registerUserApi(data: { username: string; email: string; password: string; role: string }) {
  const res = await fetchWithAuth('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify(data),
  });
  return await res.json();
}

