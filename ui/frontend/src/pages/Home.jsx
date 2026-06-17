import { Link } from 'react-router-dom'
import './Home.css'

const FEATURES = [
  { icon:'🔍', title:'Instant Analysis', desc:'Upload any pulmonary CT scan and receive a complete AI-powered assessment report in seconds.' },
  { icon:'🌡', title:'Visual Explanations', desc:'Three independent visual methods show exactly which regions of the scan drove the AI decision.' },
  { icon:'📊', title:'Confidence Metrics', desc:'Every result includes a probability breakdown and confidence score for transparent, interpretable outputs.' },
  { icon:'⊞',  title:'Side-by-Side Comparison', desc:'Compare all three explanation views simultaneously to validate findings and build diagnostic confidence.' },
  { icon:'📋', title:'Detailed Reports', desc:'Each analysis produces a structured report with decision details, timings and model information.' },
  { icon:'🛡', title:'Clinical Grade Training', desc:'Trained on a curated dataset annotated by multiple radiologists, ensuring high-quality reliable results.' },
]

const STEPS = [
  { n:'01', title:'Upload your scan', desc:'Drag & drop or browse a PNG/JPG CT image from your computer.' },
  { n:'02', title:'AI processes it',  desc:'The model analyzes the image and generates a prediction with probability score.' },
  { n:'03', title:'Review explanations', desc:'Three visual methods highlight the regions that influenced the result.' },
  { n:'04', title:'Save & share',    desc:'Download the annotated images or copy the analysis report.' },
]

export default function Home() {
  return (
    <main className="home">

      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero-bg">
          <div className="blob b1"/><div className="blob b2"/>
          <div className="dots"/>
        </div>
        <div className="container hero-body">
          <span className="badge badge-blue hero-badge">
            <span className="pulse-dot"/>
            AI-Powered · Pulmonary Imaging
          </span>
          <h1 className="display">
            Smarter lung scan<br/>analysis, <span>explained.</span>
          </h1>
          <p className="subtext hero-sub">
            MedVision helps clinicians and researchers analyze pulmonary CT scans with
            AI-driven detection and transparent visual explanations —
            so you always understand <em>why</em>.
          </p>
          <div className="hero-cta">
            <Link to="/analyze" className="btn btn-primary btn-lg">
              <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
              Analyze a scan
            </Link>
            <Link to="/dashboard" className="btn btn-outline btn-lg">View Dashboard</Link>
          </div>
          <div className="hero-stats">
            {[['91.2%','AUC-ROC score'],['4,200+','Training scans'],['3','Explanation methods']].map(([v,l])=>(
              <div key={l} className="hstat">
                <b>{v}</b><span>{l}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Mock UI card */}
        <div className="container">
          <div className="mock-card">
            <div className="mock-bar">
              <div style={{display:'flex',gap:'6px'}}>
                {['#ef4444','#f59e0b','#10b981'].map(c=><div key={c} style={{width:11,height:11,borderRadius:'50%',background:c}}/>)}
              </div>
              <span className="mock-title">medvision — scan analysis</span>
              <div/>
            </div>
            <div className="mock-body">
              <div className="mock-scan">
                <svg viewBox="0 0 200 200" fill="none" style={{width:'100%',height:'100%'}}>
                  <rect width="200" height="200" rx="6" fill="#0a0f1e"/>
                  <ellipse cx="100" cy="100" rx="72" ry="80" fill="none" stroke="#1e2d4a" strokeWidth="1"/>
                  <ellipse cx="78"  cy="105" rx="20" ry="30" fill="#0e1d35"/>
                  <ellipse cx="122" cy="105" rx="20" ry="30" fill="#0e1d35"/>
                  <ellipse cx="78"  cy="105" rx="14" ry="23" fill="#162a4a"/>
                  <ellipse cx="122" cy="105" rx="14" ry="23" fill="#162a4a"/>
                  <circle  cx="116" cy="97"  r="8"  fill="rgba(239,68,68,.2)"/>
                  <circle  cx="116" cy="97"  r="5"  fill="rgba(239,68,68,.5)"/>
                  <circle  cx="116" cy="97"  r="3"  fill="#ef4444"/>
                  <ellipse cx="116" cy="97"  rx="18" ry="15" fill="url(#hg)" opacity=".45"/>
                  <defs>
                    <radialGradient id="hg" cx="50%" cy="50%" r="50%">
                      <stop offset="0%"   stopColor="#ef4444" stopOpacity=".9"/>
                      <stop offset="55%"  stopColor="#f59e0b" stopOpacity=".5"/>
                      <stop offset="100%" stopColor="#10b981" stopOpacity="0"/>
                    </radialGradient>
                  </defs>
                  <line x1="116" y1="85" x2="116" y2="109" stroke="#ef4444" strokeWidth=".8" strokeDasharray="2,2" opacity=".7"/>
                  <line x1="104" y1="97" x2="128" y2="97" stroke="#ef4444" strokeWidth=".8" strokeDasharray="2,2" opacity=".7"/>
                </svg>
              </div>
              <div className="mock-info">
                <span className="badge badge-red" style={{fontSize:13,padding:'6px 14px'}}>⚠ Nodule Detected</span>
                <div className="mock-conf">
                  <div style={{fontSize:12,color:'var(--g500)',marginBottom:5}}>Confidence</div>
                  <div style={{display:'flex',alignItems:'center',gap:8}}>
                    <div style={{flex:1,height:6,background:'var(--g200)',borderRadius:3,overflow:'hidden'}}>
                      <div style={{width:'87%',height:'100%',background:'var(--danger)',borderRadius:3}}/>
                    </div>
                    <b style={{fontFamily:'var(--mono)',color:'var(--danger)'}}>87%</b>
                  </div>
                </div>
                <div className="mock-xai">
                  {['Heatmap','Boundaries','Attribution'].map((m,i)=>(
                    <div key={m} className="mock-xai-item">
                      <div className="mock-xai-thumb" style={{
                        background:'#0a0f1e',
                        filter:['sepia(.8) saturate(4) hue-rotate(330deg)','sepia(.6) saturate(3) hue-rotate(80deg)','sepia(.6) saturate(3) hue-rotate(200deg)'][i],
                      }}>
                        <svg viewBox="0 0 60 60" style={{width:'100%',height:'100%'}}>
                          <rect width="60" height="60" fill="#0a0f1e"/>
                          <ellipse cx="30" cy="30" rx="18" ry="22" fill="#0e1d35"/>
                          <circle cx="32" cy="28" r="7" fill="rgba(239,68,68,.35)"/>
                        </svg>
                      </div>
                      <span>{m}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Features ── */}
      <section className="section">
        <div className="container">
          <div className="sec-head">
            <p className="label">What MedVision offers</p>
            <h2 className="headline" style={{marginTop:'8px'}}>Everything you need for<br/>confident analysis</h2>
          </div>
          <div className="feat-grid">
            {FEATURES.map(f=>(
              <div key={f.title} className="feat-card">
                <div className="feat-icon">{f.icon}</div>
                <h3 className="feat-title">{f.title}</h3>
                <p className="feat-desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section className="section" style={{background:'var(--g50)'}}>
        <div className="container">
          <div className="sec-head">
            <p className="label">How it works</p>
            <h2 className="headline" style={{marginTop:'8px'}}>From upload to insight<br/>in four steps</h2>
          </div>
          <div className="steps-grid">
            {STEPS.map(s=>(
              <div key={s.n} className="step-item">
                <div className="step-num">{s.n}</div>
                <h3 className="step-title">{s.title}</h3>
                <p className="step-desc">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="section">
        <div className="container">
          <div className="cta-block">
            <div className="cta-glow"/>
            <p className="label" style={{color:'rgba(255,255,255,.5)'}}>Get started</p>
            <h2 className="headline" style={{color:'#fff',marginTop:8}}>Ready to analyze your first scan?</h2>
            <p style={{color:'rgba(255,255,255,.65)',fontSize:16,marginTop:8,maxWidth:440}}>
              Upload a CT image and receive a full AI-powered analysis with visual explanations in seconds.
            </p>
            <div style={{display:'flex',gap:12,marginTop:28,flexWrap:'wrap'}}>
              <Link to="/analyze" className="btn btn-lg" style={{background:'#fff',color:'var(--primary)',fontWeight:700}}>
                Start for free
              </Link>
              <Link to="/dashboard" className="btn btn-lg btn-ghost" style={{color:'rgba(255,255,255,.8)',border:'1.5px solid rgba(255,255,255,.2)'}}>
                View metrics
              </Link>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer">
        <div className="container footer-inner">
          <div style={{display:'flex',alignItems:'center',gap:8}}>
            <svg width="24" height="24" viewBox="0 0 32 32" fill="none">
              <rect width="32" height="32" rx="9" fill="#0057FF"/>
              <path d="M9 23L13.5 10L16 17L18.5 13L22 23" stroke="white" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="22.5" cy="10.5" r="2" fill="#60AFFF"/>
            </svg>
            <b style={{fontSize:16,letterSpacing:'-.03em'}}>MedVision</b>
          </div>
          <p style={{color:'var(--g400)',fontSize:13}}>© 2025 MedVision · Research use only</p>
        </div>
      </footer>
    </main>
  )
}
