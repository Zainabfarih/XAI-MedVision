import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import './Navbar.css'

export default function Navbar() {
  const { pathname } = useLocation()
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)

  useEffect(() => {
    const fn = () => setScrolled(window.scrollY > 10)
    window.addEventListener('scroll', fn)
    return () => window.removeEventListener('scroll', fn)
  }, [])

  useEffect(() => setOpen(false), [pathname])

  const links = [
    { to: '/',          label: 'Home' },
    { to: '/analyze',   label: 'Analyze' },
    { to: '/dashboard', label: 'Dashboard' },
  ]

  return (
    <header className={`nav${scrolled ? ' nav--up' : ''}`}>
      <div className="nav-inner">
        <Link to="/" className="nav-logo">
          <svg width="30" height="30" viewBox="0 0 32 32" fill="none">
            <rect width="32" height="32" rx="9" fill="#0057FF"/>
            <path d="M9 23L13.5 10L16 17L18.5 13L22 23" stroke="white"
              strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round"/>
            <circle cx="22.5" cy="10.5" r="2" fill="#60AFFF"/>
          </svg>
          MedVision
        </Link>

        <nav className="nav-links">
          {links.map(l => (
            <Link key={l.to} to={l.to}
              className={`nav-link${pathname === l.to ? ' active' : ''}`}>
              {l.label}
            </Link>
          ))}
        </nav>

        <div className="nav-right">
          <Link to="/analyze" className="btn btn-primary btn-sm">Start Analysis</Link>
          <button className="hamburger" onClick={() => setOpen(o => !o)}>
            <span className={open ? 'x' : ''}/>
          </button>
        </div>
      </div>

      {open && (
        <div className="nav-drawer">
          {links.map(l => (
            <Link key={l.to} to={l.to} className="drawer-link">{l.label}</Link>
          ))}
          <Link to="/analyze" className="btn btn-primary" style={{justifyContent:'center',marginTop:'8px'}}>
            Start Analysis
          </Link>
        </div>
      )}
    </header>
  )
}
