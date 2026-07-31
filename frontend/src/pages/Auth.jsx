import { useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Bot, ArrowLeft } from 'lucide-react';

export default function Auth() {
  const location = useLocation();
  const navigate = useNavigate();
  const isLogin = location.pathname === '/login';
  
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    // Mock authentication for Phase 7
    navigate('/chat');
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
            
            {!isLogin && (
              <div className="input-group">
                <label>Full Name</label>
                <input type="text" placeholder="Jane Doe" required />
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

            <button type="submit" className="btn btn-primary" style={{ width: '100%', marginTop: '16px', padding: '14px' }}>
              {isLogin ? 'Sign In' : 'Create Account'}
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
