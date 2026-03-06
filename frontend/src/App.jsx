import { useEffect, useRef, useState } from "react";
import SourcePanel  from "./components/SourcePanel";
import AlertBanner  from "./components/AlertBanner";
import VLMPanel     from "./components/VLMPanel";
import LogPanel     from "./components/LogPanel";

const API = "http://localhost:8000";

export default function App() {
  const [status,    setStatus]    = useState(null);
  const [alerts,    setAlerts]    = useState([]);
  const [logs,      setLogs]      = useState([]);
  const [persons,   setPersons]   = useState([]);
  const [weapons,   setWeapons]   = useState([]);   // latest weapon detections
  const [activeTab, setActiveTab] = useState("vlm");
  const imgRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const [s, a, l, p] = await Promise.all([
          fetch(`${API}/status`).then(r => r.json()).catch(() => null),
          fetch(`${API}/alerts`).then(r => r.json()).catch(() => []),
          fetch(`${API}/logs`).then(r => r.json()).catch(() => []),
          fetch(`${API}/persons`).then(r => r.json()).catch(() => []),
        ]);
        if (s) {
          setStatus(s);
          // Weapon detections live inside /status
          setWeapons(Array.isArray(s.weapon_detections) ? s.weapon_detections : []);
        }
        setAlerts(Array.isArray(a)  ? a.slice(-30).reverse() : []);
        setLogs(Array.isArray(l)    ? l.slice(-30).reverse() : []);
        setPersons(Array.isArray(p) ? p.slice(-20).reverse() : []);
      } catch (_) {}
    }, 800);
    return () => clearInterval(interval);
  }, []);

  const alertColor = {
    CLEAR:  "border-green-500  bg-green-950",
    YELLOW: "border-yellow-400 bg-yellow-950",
    RED:    "border-red-500    bg-red-950",
  }[status?.alert ?? "CLEAR"] ?? "border-gray-500 bg-gray-900";

  const hasWeapons = weapons.length > 0;

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden">

      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 py-3 bg-gray-900 border-b border-gray-800 shrink-0">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold tracking-wide">
            🎯 YOLO26 <span className="text-blue-400">+</span> VLM Surveillance
          </h1>
          {/* Model badges */}
          <span className="text-xs px-2 py-0.5 rounded bg-blue-900 text-blue-300 font-mono">
            YOLO26n
          </span>
          <span className="text-xs px-2 py-0.5 rounded bg-purple-900 text-purple-300 font-mono">
            SmolVLM2-2.2B
          </span>
          <span className="text-xs px-2 py-0.5 rounded bg-orange-900 text-orange-300 font-mono">
            WeaponDetect
          </span>
        </div>

        <div className="flex items-center gap-3">
          {/* Weapon alert badge — shows when weapon model fires */}
          {hasWeapons && (
            <span className="flex items-center gap-1.5 text-xs px-3 py-1 rounded-full font-semibold
                             border border-red-500 bg-red-950 text-red-300 animate-pulse">
              🔪 {weapons.length} Weapon{weapons.length > 1 ? "s" : ""} Detected
            </span>
          )}

          {/* Person count badge */}
          {status?.running && (
            <span className="text-xs px-3 py-1 rounded-full font-semibold
                             border border-gray-600 bg-gray-800 text-gray-300">
              👥 {status?.person_count ?? 0} People
            </span>
          )}

          {/* Alert level badge */}
          <span className={`text-xs px-3 py-1 rounded-full font-semibold border ${alertColor}`}>
            {status?.alert ?? "IDLE"}
          </span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* ── Left Sidebar — Source Control ── */}
        <aside className="w-64 shrink-0 bg-gray-900 border-r border-gray-800 flex flex-col p-4 gap-4 overflow-y-auto">
          <SourcePanel api={API} status={status} />

          {/* Live weapon list in sidebar */}
          {hasWeapons && (
            <div className="mt-2">
              <p className="text-xs text-red-400 font-semibold uppercase tracking-wider mb-1">
                🔪 Weapon Detections
              </p>
              <div className="flex flex-col gap-1">
                {weapons.map((w, i) => (
                  <div key={i}
                       className="flex items-center justify-between text-xs px-2 py-1
                                  rounded bg-red-950 border border-red-800">
                    <span className="text-red-300 font-semibold capitalize">{w.label}</span>
                    <span className="text-red-400">{Math.round(w.confidence * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </aside>

        {/* ── Main — Video Feed ── */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          <AlertBanner status={status} />

          <div className="flex-1 flex items-center justify-center bg-gray-950 overflow-hidden relative">
            {status?.running ? (
              <>
                <img
                  ref={imgRef}
                  src={`${API}/video_feed`}
                  className="max-h-full max-w-full object-contain rounded"
                  alt="Live Feed"
                />
                {/* Detection summary overlay */}
                {status?.detection_summary && (
                  <div className="absolute bottom-2 left-2 right-2 text-center">
                    <span className="text-xs px-2 py-1 rounded bg-black/60 text-green-300">
                      {status.detection_summary}
                    </span>
                  </div>
                )}
              </>
            ) : (
              <div className="text-gray-600 text-lg flex flex-col items-center gap-3 select-none">
                <span className="text-6xl">📷</span>
                <p>Select a source and press Start</p>
                <p className="text-sm text-gray-700">
                  YOLO26n tracks people · WeaponDetect finds axe/knife/gun · Qwen2.5-VL confirms threats
                </p>
              </div>
            )}
          </div>
        </main>

        {/* ── Right Sidebar — Panels ── */}
        <aside className="w-80 shrink-0 bg-gray-900 border-l border-gray-800 flex flex-col overflow-hidden">
          <div className="flex shrink-0 border-b border-gray-800">
            {["vlm", "log"].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-2 text-sm font-semibold uppercase tracking-wider transition-colors
                  ${activeTab === tab
                    ? "bg-gray-800 text-white border-b-2 border-blue-500"
                    : "text-gray-500 hover:text-gray-300"}`}
              >
                {tab === "vlm" ? "🤖 VLM Alerts" : "📋 Entry / Exit"}
              </button>
            ))}
          </div>
          <div className="flex-1 overflow-y-auto p-3">
            {activeTab === "vlm"
              ? <VLMPanel status={status} alerts={alerts} persons={persons} />
              : <LogPanel logs={logs} />}
          </div>
        </aside>

      </div>
    </div>
  );
}
