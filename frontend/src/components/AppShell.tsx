import { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { getHealth } from "../api/client";

const NAV = [
  { to: "/", label: "Overview", end: true },
  { to: "/models", label: "Models" },
  { to: "/matches", label: "Matches" },
  { to: "/predict", label: "Predict" },
  { to: "/statistics", label: "Statistics" },
];

export function AppShell() {
  const [apiOk, setApiOk] = useState<boolean | null>(null);

  useEffect(() => {
    getHealth()
      .then(() => setApiOk(true))
      .catch(() => setApiOk(false));
  }, []);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand__mark">FIL</span>
          <div>
            <div className="brand__name">Football Intelligence Lab</div>
            <div className="brand__sub">xG · inference · StatsBomb</div>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) => `nav__link${isActive ? " nav__link--active" : ""}`}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className={`api-pill${apiOk === false ? " api-pill--down" : ""}`}>
          <span className="api-pill__dot" />
          {apiOk === null ? "Checking API" : apiOk ? "API connected" : "API unreachable"}
        </div>
        <p className="sidebar__note">
          Figures come from the local FastAPI service. Nothing here is hardcoded.
        </p>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
