import { useState, useRef, useEffect } from 'react';
import { Bot, User, Send, Settings, History, PlusCircle, LogOut, Paperclip, X } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import React, { Component } from 'react';

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  render() {
    if (this.state.hasError) {
      return <div style={{ color: 'red', padding: '20px' }}>
        <h1>Frontend Crash!</h1>
        <pre>{this.state.error?.toString()}</pre>
        <pre>{this.state.error?.stack}</pre>
      </div>;
    }
    return this.props.children;
  }
}

export default function Chat() {
  const { token, user, logout } = useAuth();
  const navigate = useNavigate();

  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am your AI HR Assistant. You can ask me about policies, employee data, or upload contracts for review.' }
  ]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [uploadedFilePath, setUploadedFilePath] = useState(null);
  const [uploading, setUploading] = useState(false);
  
  const endOfMessagesRef = useRef(null);
  const fileInputRef = useRef(null);

  const [sessionId] = useState(() => Math.random().toString(36).substring(7));

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`
        },
        body: formData
      });
      if (!res.ok) throw new Error('Upload failed');
      const data = await res.json();
      setUploadedFilePath(data.file_path);
    } catch (err) {
      console.error(err);
      alert('Failed to upload file.');
    } finally {
      setUploading(false);
      e.target.value = null; // reset input
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (isTyping) return;
    if (!input.trim() && !uploadedFilePath) return;
    
    const currentInput = input || (uploadedFilePath ? 'Please analyze this document.' : '');
    setMessages(prev => [...prev, { 
      role: 'user', 
      content: currentInput,
      hasAttachment: !!uploadedFilePath 
    }]);
    
    setInput('');
    setIsTyping(true);
    const pathToSend = uploadedFilePath;
    setUploadedFilePath(null); // clear for next message

    // Initialize assistant response
    setMessages(prev => [...prev, { role: 'assistant', content: '' }]);

    try {
      const response = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ 
          query: currentInput, 
          session_id: sessionId,
          uploaded_file_path: pathToSend
        }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      setIsTyping(false);
      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.replace('data: ', '').trim();
            if (dataStr === '[DONE]') break;
            
            try {
              const data = JSON.parse(dataStr);
              if (data.token) {
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastIndex = newMessages.length - 1;
                  newMessages[lastIndex] = {
                    ...newMessages[lastIndex],
                    content: newMessages[lastIndex].content + data.token
                  };
                  return newMessages;
                });
              } else if (data.error) {
                setMessages(prev => {
                  const newMessages = [...prev];
                  const lastIndex = newMessages.length - 1;
                  newMessages[lastIndex] = {
                    ...newMessages[lastIndex],
                    content: newMessages[lastIndex].content + `\n\n[Error: ${data.error}]`
                  };
                  return newMessages;
                });
              }
            } catch (err) {
              console.error('Error parsing SSE JSON:', err);
            }
          }
        }
      }
    } catch (error) {
      console.error('Fetch error:', error);
      setIsTyping(false);
      setMessages(prev => {
        const newMessages = [...prev];
        const lastIndex = newMessages.length - 1;
        newMessages[lastIndex] = {
          ...newMessages[lastIndex],
          content: newMessages[lastIndex].content + '\n\n[Failed to connect to the server. Please try again.]'
        };
        return newMessages;
      });
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <ErrorBoundary>
    <div style={{ display: 'flex', height: '100vh', width: '100vw', overflow: 'hidden', background: 'var(--bg-dark)' }}>
      
      {/* Sidebar */}
      <aside style={{ width: '280px', background: 'var(--bg-darker)', borderRight: '1px solid var(--border)', display: 'flex', flexDirection: 'column', padding: '16px' }}>
        
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '12px', fontSize: '1.25rem', fontWeight: '700', marginBottom: '24px' }}>
          <Bot size={24} color="var(--primary)" />
          <span>HrMind</span>
        </div>

        <button className="btn btn-outline" style={{ display: 'flex', justifyContent: 'flex-start', background: 'var(--bg-panel)', padding: '12px' }}>
          <PlusCircle size={18} /> New Chat
        </button>

        <div style={{ marginTop: '24px', flex: 1, overflowY: 'auto' }}>
          <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--text-muted)', fontWeight: '600', marginBottom: '12px', padding: '0 12px' }}>Recent Chats</div>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            {['Leave Policies', 'Engineering Salary Data', 'Onboarding Process'].map((chat, i) => (
              <div key={i} style={{ padding: '10px 12px', borderRadius: '8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--text-muted)', fontSize: '0.875rem' }}
                   onMouseEnter={e => e.currentTarget.style.background = 'var(--bg-panel)'}
                   onMouseLeave={e => e.currentTarget.style.background = 'transparent'}>
                <History size={16} />
                <span style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{chat}</span>
              </div>
            ))}
          </div>
        </div>

        <div style={{ marginTop: 'auto', borderTop: '1px solid var(--border)', paddingTop: '16px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <div style={{ padding: '10px 12px', borderRadius: '8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
            <Settings size={18} /> Settings
          </div>
          <button onClick={handleLogout} style={{ background: 'transparent', border: 'none', textAlign: 'left', padding: '10px 12px', borderRadius: '8px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '12px', color: 'var(--text-muted)', fontSize: '0.875rem' }}>
            <LogOut size={18} /> Log out
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', position: 'relative' }}>
        <div className="bg-blobs" style={{ position: 'absolute', opacity: 0.5 }}></div>

        {/* Header */}
        <header style={{ height: '64px', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0 24px', background: 'rgba(10, 11, 14, 0.8)', backdropFilter: 'blur(12px)' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: '500' }}>Current Session</h2>
          <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
            Logged in as <span style={{ color: 'white', fontWeight: '500' }}>{user?.name || 'User'}</span>
          </div>
        </header>

        {/* Messages */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
          {messages.map((m, i) => (
            <div key={i} style={{ display: 'flex', gap: '16px', maxWidth: '800px', margin: '0 auto', width: '100%', flexDirection: m.role === 'user' ? 'row-reverse' : 'row' }}>
              
              <div style={{ width: '40px', height: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                            background: m.role === 'user' ? 'var(--primary)' : 'var(--bg-panel)',
                            boxShadow: m.role === 'user' ? '0 0 15px var(--primary-glow)' : 'none',
                            border: m.role === 'assistant' ? '1px solid var(--border)' : 'none' }}>
                {m.role === 'user' ? <User size={20} color="white" /> : <Bot size={20} color="var(--primary)" />}
              </div>
              
              <div style={{ background: m.role === 'user' ? 'var(--bg-panel)' : 'transparent',
                            padding: m.role === 'user' ? '12px 16px' : '8px 0',
                            borderRadius: '16px', borderTopRightRadius: m.role === 'user' ? 0 : '16px',
                            borderTopLeftRadius: m.role === 'assistant' ? 0 : '16px',
                            color: 'var(--text-main)', fontSize: '1rem', lineHeight: '1.6' }}>
                {m.role === 'assistant' ? (
                  <div className="markdown-body">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {m.content}
                    </ReactMarkdown>
                  </div>
                ) : (
                  <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>
                )}
              </div>
            </div>
          ))}
          
          {isTyping && (
            <div style={{ display: 'flex', gap: '16px', maxWidth: '800px', margin: '0 auto', width: '100%' }}>
              <div className="bot-pulsate" style={{ width: '40px', height: '40px', borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-panel)', border: '1px solid var(--primary)', boxShadow: '0 0 10px var(--primary-glow)' }}>
                <Bot size={20} color="var(--primary)" />
              </div>
              <div style={{ padding: '8px 0', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                <span className="dot" style={{ animation: 'blink 1.4s infinite both' }}>•</span>
                <span className="dot" style={{ animation: 'blink 1.4s infinite both 0.2s' }}>•</span>
                <span className="dot" style={{ animation: 'blink 1.4s infinite both 0.4s' }}>•</span>
              </div>
            </div>
          )}
          <div ref={endOfMessagesRef} />
        </div>

        {/* Input Area */}
        <div style={{ padding: '24px', background: 'transparent' }}>
          
          {uploadedFilePath && (
            <div style={{ maxWidth: '800px', margin: '0 auto 8px auto', padding: '8px 16px', background: 'var(--bg-panel)', borderRadius: '8px', border: '1px solid var(--primary)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '0.875rem' }}>
                <Paperclip size={14} color="var(--primary)" />
                <span style={{ color: 'var(--text-main)' }}>{uploadedFilePath.split('/').pop().split('\\').pop()}</span>
              </div>
              <button onClick={() => setUploadedFilePath(null)} style={{ background: 'transparent', border: 'none', cursor: 'pointer', color: 'var(--text-muted)' }}>
                <X size={16} />
              </button>
            </div>
          )}

          <form onSubmit={handleSubmit} style={{ maxWidth: '800px', margin: '0 auto', position: 'relative', display: 'flex', gap: '8px' }}>
            
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileUpload} 
              style={{ display: 'none' }} 
              accept=".pdf,.docx" 
            />
            
            <button 
              type="button" 
              onClick={() => fileInputRef.current?.click()} 
              disabled={uploading || isTyping}
              style={{ width: '56px', borderRadius: '24px', background: 'var(--bg-panel)', border: '1px solid var(--border)', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: (uploading || isTyping) ? 'not-allowed' : 'pointer', color: 'var(--text-muted)' }}
            >
              {uploading ? <div className="dot" style={{ animation: 'blink 1s infinite' }}>•</div> : <Paperclip size={20} />}
            </button>

            <div style={{ flex: 1, position: 'relative' }}>
              <input 
                value={input}
                onChange={e => setInput(e.target.value)}
                placeholder="Ask HrMind a question..."
                style={{ width: '100%', padding: '16px 64px 16px 24px', borderRadius: '24px', border: '1px solid var(--border)', background: 'var(--bg-panel)', backdropFilter: 'blur(12px)', color: 'white', fontSize: '1rem', outline: 'none', boxShadow: 'var(--shadow-lg)' }}
              />
              <button type="submit" disabled={(!input.trim() && !uploadedFilePath) || isTyping} style={{ position: 'absolute', right: '8px', top: '8px', bottom: '8px', width: '40px', borderRadius: '50%', background: (input.trim() || uploadedFilePath) ? 'var(--primary)' : 'rgba(255,255,255,0.1)', border: 'none', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: (input.trim() || uploadedFilePath) ? 'pointer' : 'not-allowed', color: 'white', transition: 'all 0.2s' }}>
                <Send size={18} style={{ transform: 'translateX(1px) translateY(1px)' }} />
              </button>
            </div>
          </form>
          <div style={{ textAlign: 'center', marginTop: '12px', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            HrMind uses AI and may generate inaccurate information. Please verify important HR policies.
          </div>
        </div>

      </main>

      <style dangerouslySetInnerHTML={{__html: `
        @keyframes blink {
          0% { opacity: 0.2; }
          20% { opacity: 1; }
          100% { opacity: 0.2; }
        }
        @keyframes bot-pulse {
          0% { transform: scale(1); opacity: 0.8; box-shadow: 0 0 5px var(--primary-glow); }
          50% { transform: scale(1.1); opacity: 1; box-shadow: 0 0 20px var(--primary-glow); }
          100% { transform: scale(1); opacity: 0.8; box-shadow: 0 0 5px var(--primary-glow); }
        }
        .bot-pulsate {
          animation: bot-pulse 1.5s cubic-bezier(0.4, 0, 0.2, 1) infinite;
        }
        
        .markdown-body {
          font-family: inherit;
        }
        .markdown-body p { margin-bottom: 1em; }
        .markdown-body p:last-child { margin-bottom: 0; }
        .markdown-body ul, .markdown-body ol { margin-left: 1.5em; margin-bottom: 1em; }
        .markdown-body li { margin-bottom: 0.5em; }
        .markdown-body code { background: rgba(255,255,255,0.1); padding: 0.2em 0.4em; border-radius: 4px; font-size: 0.9em; font-family: monospace; white-space: pre-wrap; word-break: break-word; }
        .markdown-body pre { background: rgba(0,0,0,0.3); padding: 1em; border-radius: 8px; overflow-x: hidden; margin-bottom: 1em; white-space: pre-wrap; word-break: break-word; }
        .markdown-body pre code { background: transparent; padding: 0; white-space: pre-wrap; }
        .markdown-body strong { color: white; font-weight: 600; }
        .markdown-body table { width: 100%; border-collapse: collapse; margin-bottom: 1em; }
        .markdown-body th, .markdown-body td { border: 1px solid var(--border); padding: 8px 12px; text-align: left; }
        .markdown-body th { background: rgba(255,255,255,0.05); color: white; font-weight: 600; }
        .markdown-body a { color: var(--primary); text-decoration: none; }
        .markdown-body a:hover { text-decoration: underline; }
      `}} />
    </div>
    </ErrorBoundary>
  );
}
