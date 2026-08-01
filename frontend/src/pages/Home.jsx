import { Link } from 'react-router-dom';
import { Bot, FileText, Database, ArrowRight, ShieldCheck } from 'lucide-react';

export default function Home() {
  return (
    <>
      <div className="bg-blobs"></div>
      
      <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 24px', minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        
        {/* Navigation */}
        <nav style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '24px 0' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '1.25rem', fontWeight: '700' }}>
            <Bot size={28} color="var(--primary)" />
            <span>HrMind</span>
          </div>
          <div style={{ display: 'flex', gap: '16px' }}>
            <Link to="/login" className="btn btn-outline">Log in</Link>
            <Link to="/signup" className="btn btn-primary">Sign up</Link>
          </div>
        </nav>

        {/* Hero Section */}
        <main style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', textAlign: 'center', gap: '32px', padding: '64px 0' }}>
          <h1 style={{ fontSize: '4rem', fontWeight: '800', lineHeight: '1.1', maxWidth: '800px' }}>
            The Future of <span className="text-gradient">HR Intelligence</span> is Here.
          </h1>
          <p style={{ fontSize: '1.25rem', color: 'var(--text-muted)', maxWidth: '600px' }}>
            Multi-agent orchestration powered by LangGraph. Query policies, analyze contracts, and explore HR databases with a single conversational interface.
          </p>
          
          <div style={{ display: 'flex', gap: '16px', marginTop: '16px' }}>
            <Link to="/signup" className="btn btn-primary" style={{ padding: '16px 32px', fontSize: '1.125rem' }}>
              Get Started <ArrowRight size={20} />
            </Link>
          </div>

          {/* Feature Cards */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '24px', width: '100%', marginTop: '64px', textAlign: 'left' }}>
            <FeatureCard 
              icon={<Database size={24} color="var(--accent)" />}
              title="SQL Agent"
              desc="Natural language to SQL via sqlglot security layer. Queries the HR employee database seamlessly."
            />
            <FeatureCard 
              icon={<ShieldCheck size={24} color="var(--primary)" />}
              title="RAG Agent"
              desc="Hybrid BM25 and Dense vector search with Reciprocal Rank Fusion for perfect policy retrieval."
            />
            <FeatureCard 
              icon={<FileText size={24} color="var(--secondary)" />}
              title="Doc Parser"
              desc="Asynchronous OCR pipeline extracting structured entities from PDF and DOCX contracts."
            />
          </div>
        </main>
      </div>
    </>
  );
}

function FeatureCard({ icon, title, desc }) {
  return (
    <div className="glass-panel" style={{ padding: '32px', display: 'flex', flexDirection: 'column', gap: '16px', transition: 'transform 0.3s ease' }}
         onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-5px)'}
         onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}>
      <div style={{ width: '48px', height: '48px', borderRadius: '12px', background: 'rgba(255,255,255,0.05)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        {icon}
      </div>
      <h3 style={{ fontSize: '1.25rem', fontWeight: '600' }}>{title}</h3>
      <p style={{ color: 'var(--text-muted)' }}>{desc}</p>
    </div>
  );
}
