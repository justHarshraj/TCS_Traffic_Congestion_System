import { createContext, useContext, useState, useEffect } from 'react';
import type { ReactNode } from 'react';
import { getMeApi, loginApi, verifyOtpApi, resendOtpApi } from '../api/auth';
import type { User, LoginResponse } from '../api/auth';

interface AuthContextType {
  user: User | null;
  token: string | null;
  isLoading: boolean;
  login: (email: string, pass: string) => Promise<LoginResponse>;
  verifyOtp: (email: string, otp: string) => Promise<LoginResponse>;
  resendOtp: (email: string) => Promise<{ success: boolean; message?: string; error?: string; dev_otp?: string }>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(localStorage.getItem('tcs_token'));
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const initAuth = async () => {
      const storedToken = localStorage.getItem('tcs_token');
      if (storedToken) {
        const res = await getMeApi();
        if (res.success && res.user) {
          setUser(res.user);
          setToken(storedToken);
        } else {
          localStorage.removeItem('tcs_token');
          setToken(null);
          setUser(null);
        }
      }
      setIsLoading(false);
    };

    initAuth();
  }, []);

  const login = async (email: string, pass: string): Promise<LoginResponse> => {
    const res = await loginApi(email, pass);
    if (res.success && res.token && res.user) {
      localStorage.setItem('tcs_token', res.token);
      setToken(res.token);
      setUser(res.user);
    }
    return res;
  };

  const verifyOtp = async (email: string, otp: string): Promise<LoginResponse> => {
    const res = await verifyOtpApi(email, otp);
    if (res.success && res.token && res.user) {
      localStorage.setItem('tcs_token', res.token);
      setToken(res.token);
      setUser(res.user);
    }
    return res;
  };

  const resendOtp = async (email: string) => {
    return await resendOtpApi(email);
  };

  const logout = () => {
    localStorage.removeItem('tcs_token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{ user, token, isLoading, login, verifyOtp, resendOtp, logout }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

