import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Bot, ArrowLeft } from 'lucide-react';
import { useAuth } from '../context/AuthContext';

export default function Auth() {
  const location = useLocation();
  const navigate = useNavigate();
  const isLogin = location.pathname === '/login';
  
  const { login, register } = useAuth();
  
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    
    try {
      if (isLogin) {
        await login(email, password);
      } else {
        await register(name, email, password);
      }
      navigate('/chat');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="bg-blobs"></div>
      
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '24px' }}>
        
        <Link to="/" style={{ position: 'absolute', top: '32px', left: '32px', display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--text-muted)', textDecoration: 'none' }}>
          <ArrowLeft size={20} /> Back to Home
        </Link>

        <div className="glass-panel" style={{ width: '100%', maxWidth: '420px', padding: '40px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '16px', textAlign: 'center' }}>
            <div style={{ width: '56px', height: '56px', borderRadius: '16px', background: 'var(--bg-darker)', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 0 20px var(--primary-glow)' }}>
              <Bot size={32} color="var(--primary)" />
            </div>
            <div>
              <h2 style={{ fontSize: '1.75rem', fontWeight: '700' }}>{isLogin ? 'Welcome back' : 'Create an account'}</h2>
              <p style={{ color: 'var(--text-muted)', marginTop: '4px' }}>
                {isLogin ? 'Enter your details to access your workspace.' : 'Start analyzing HR data intelligently.'}
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column' }}>
            
            {error && (
              <div style={{ padding: '12px', background: 'rgba(239, 68, 68, 0.1)', border: '1px solid rgba(239, 68, 68, 0.5)', borderRadius: '8px', color: '#ef4444', fontSize: '0.875rem', marginBottom: '16px' }}>
                {error}
              </div>
            )}

            {!isLogin && (
              <div className="input-group">
                <label>Full Name</label>
                <input type="text" value={name} onChange={e => setName(e.target.value)} placeholder="Jane Doe" required />
              </div>
            )}
            
            <div className="input-group">
              <label>Email address</label>
              <input type="email" value={email} onChange={e => setEmail(e.target.value)} placeholder="jane@company.com" required />
            </div>
            
            <div className="input-group">
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <label>Password</label>
                {isLogin && <a href="#" style={{ fontSize: '0.875rem', color: 'var(--primary)', textDecoration: 'none' }}>Forgot password?</a>}
              </div>
              <input type="password" value={password} onChange={e => setPassword(e.target.value)} placeholder="••••••••" required />
            </div>

            <button type="submit" disabled={loading} className="btn btn-primary" style={{ width: '100%', marginTop: '16px', padding: '14px', opacity: loading ? 0.7 : 1 }}>
              {loading ? 'Processing...' : (isLogin ? 'Sign In' : 'Create Account')}
            </button>
          </form>

          <div style={{ textAlign: 'center', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
            {isLogin ? "Don't have an account? " : "Already have an account? "}
            <Link to={isLogin ? '/signup' : '/login'} style={{ color: 'var(--primary)', textDecoration: 'none', fontWeight: '500' }}>
              {isLogin ? 'Sign up' : 'Log in'}
            </Link>
          </div>
          
        </div>
      </div>
    </>
  );
}
