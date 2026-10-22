import { ArrowUpRight, Gear, List, TennisBall, X } from '@phosphor-icons/react';
import { useRef, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';

export function AppShell() {
  const { pathname } = useLocation();
  const [menuPath, setMenuPath] = useState<string | null>(null);
  const menuOpen = menuPath === pathname;
  const menuToggle = useRef<HTMLButtonElement>(null);
  const closeMenu = () => setMenuPath(null);
  return (
    <div className="app-shell astra-shell product-shell society-shell">
      <a className="skip-link" href="#main-content">Skip to content</a>
      <div className="app-column">
        <header className="topbar product-topbar" onKeyDown={event => { if (event.key === 'Escape' && menuOpen) { closeMenu(); menuToggle.current?.focus(); } }}>
          <Link className="product-brand" to="/overview" aria-label="Astra Tennis home" onClick={closeMenu}>
            <TennisBall size={30} weight="regular" aria-hidden="true" />
            <span>Astra Tennis<span className="society-brand-dot">®</span></span>
          </Link>
          <button ref={menuToggle} className="society-menu-toggle" type="button" aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen} aria-controls="society-navigation" onClick={() => setMenuPath(menuOpen ? null : pathname)}>{menuOpen ? <X size={23} aria-hidden="true" /> : <List size={23} aria-hidden="true" />}</button>
          <nav id="society-navigation" className={`product-primary-nav${menuOpen ? ' is-open' : ''}`} aria-label="Primary">
            <NavLink to="/overview" end onClick={closeMenu}>Home</NavLink>
            <NavLink to="/match-center" onClick={closeMenu}>My analyses</NavLink>
            <NavLink to="/examples" onClick={closeMenu}>Examples</NavLink>
          </nav>
          <div className="product-header-actions">
            <Link className="product-settings-link" to="/settings" aria-label="Settings" onClick={closeMenu}><Gear size={21} aria-hidden="true" /></Link>
            <Link className="button button-primary product-app-cta" to="/upload" onClick={closeMenu}>Upload video<ArrowUpRight size={18} aria-hidden="true" /></Link>
          </div>
        </header>
        <main id="main-content" className="main-content" tabIndex={-1}>
          <Outlet />
        </main>
        <footer className="society-footer"><div className="society-footer-top"><span>TENNIS. SEEN DIFFERENTLY.</span><Link to="/settings">Service and replay settings <ArrowUpRight size={16} aria-hidden="true" /></Link></div><div className="society-footer-word" aria-hidden="true">ASTRA</div><div className="society-footer-bottom"><span>Astra Tennis · Video analysis</span><span>[ BUILT AROUND THE EVIDENCE ]</span><Link to="/upload">Your next match ↗</Link></div></footer>
      </div>
    </div>
  );
}
