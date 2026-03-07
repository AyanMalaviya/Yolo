import { useEffect, useRef, useState } from "react";
import SourcePanel from "./components/SourcePanel";
import AlertBanner from "./components/AlertBanner";
import VLMPanel    from "./components/VLMPanel";
import LogPanel    from "./components/LogPanel";

const API = "http://localhost:8000";

export default function App() {
  const [status,    setStatus]    = useState(null);
  const [alerts,    setAlerts]    = useState([]);
  const [logs,      setLogs]      = useState([]);
  const [persons,   setPersons]   = useState([]);
  const [weapons,   setWeapons]   = useState([]);
  const [vram,      setVram]      = useState(null);
  const [activeTab, setActiveTab] = useState("vlm");
  const [vlmToggling, setVlmToggling] = useState(false);   // loading state for button
  const imgRef = useRef(null);

  // ── Polling ───────────────────────────────────────────────────────────────
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const [s, a, l, p, v] = await Promise.all([
          fetch(`${API}/status`).then(r => r.json()).catch(() => null),
          fetch(`${API}/alerts`).then(r => r.json()).catch(() => []),
          fetch(`${API}/logs`).then(r => r.json()).catch(() => []),
          fetch(`${API}/persons`).then(r => r.json()).catch(() => []),
          fetch(`${API}/vram`).then(r => r.json()).catch(() => null),
        ]);
        if (s) {
          setStatus(s);
          setWeapons(Array.isArray(s.weapon_detections) ? s.weapon_detections : []);
        }
        setAlerts(Array.isArray(a)  ? a.slice(-30).reverse() : []);
        setLogs(Array.isArray(l)    ? l.slice(-30).reverse() : []);
        setPersons(Array.isArray(p) ? p.slice(-20).reverse() : []);
        if (v && !v.error) setVram(v);
      } catch (_) {}
    }, 800);
    return () => clearInterval(interval);
  }, []);

  // ── VLM Toggle ────────────────────────────────────────────────────────────
  const toggleVlm = async () => {
    if (vlmToggling) return;
    setVlmToggling(true);
    const endpoint = status?.vlm_enabled ? "disable" : "enable";
    try {
      await fetch(`${API}/vlm/${endpoint}`, { method: "POST" });
    } catch (_) {}
    finally {
      setVlmToggling(false);
    }
  };

  // ── Derived state ─────────────────────────────────────────────────────────
  const alertLevel  = status?.alert ?? "CLEAR";
  const hasWeapons  = weapons.length > 0;
  const vlmEnabled  = status?.vlm_enabled ?? true;
  const isRunning   = status?.running ?? false;

  const alertColor = {
    CLEAR:  "border-green-500  bg-green-950  text-green-300",
    YELLOW: "border-yellow-400 bg-yellow-950 text-yellow-300",
    RED:    "border-red-500    bg-red-950    text-red-300",
  }[alertLevel] ?? "border-gray-500 bg-gray-900 text-gray-300";

  const vramPct = vram?.usage_pct ?? 0;
  const vramColor =
    vramPct > 85 ? "text-red-400" :
    vramPct > 65 ? "text-yellow-400" :
                   "text-green-400";

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden">

      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 py-3 bg-gray-900
                         border-b border-gray-800 shrink-0 gap-4">

        {/* Left — title + model badges */}
        <div className="flex items-center gap-3 min-w-0">
          <h1 className="text-xl font-bold tracking-wide whitespace-nowrap">
            🎯 YOLO26 <span className="text-blue-400">+</span> VLM Surveillance
          </h1>
          <span className="text-xs px-2 py-0.5 rounded bg-blue-900   text-blue-300   font-mono whitespace-nowrap">YOLO26n</span>
          <span className="text-xs px-2 py-0.5 rounded bg-orange-900 text-orange-300 font-mono whitespace-nowrap hidden sm:inline">WeaponDetect</span>
          <span className="text-xs px-2 py-0.5 rounded bg-purple-900 text-purple-300 font-mono whitespace-nowrap hidden md:inline">SmolVLM2-2.2B</span>
        </div>

        {/* Right — controls + status badges */}
        <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">

          {/* VRAM meter — only when GPU available */}
          {vram && (
            <div className="flex items-center gap-1.5 text-xs px-3 py-1 rounded-full
                            border border-gray-700 bg-gray-800">
              <span className="text-gray-400">GPU</span>
              <span className={`font-semibold ${vramColor}`}>{vramPct}%</span>
              <span className="text-gray-500">{vram.free_gb?.toFixed(1)}GB free</span>
            </div>
          )}

          {/* Source FPS */}
          {isRunning && status?.source_fps > 0 && (
            <span className="text-xs px-3 py-1 rounded-full font-semibold
                             border border-gray-600 bg-gray-800 text-gray-300">
              📹 {status.source_fps}fps
            </span>
          )}

          {/* Person count */}
          {isRunning && (
            <span className="text-xs px-3 py-1 rounded-full font-semibold
                             border border-gray-600 bg-gray-800 text-gray-300">
              👥 {status?.person_count ?? 0}
            </span>
          )}

          {/* Weapon alert badge */}
          {hasWeapons && (
            <span className="flex items-center gap-1.5 text-xs px-3 py-1 rounded-full
                             font-semibold border border-red-500 bg-red-950
                             text-red-300 animate-pulse">
              🔪 {weapons.length} Weapon{weapons.length > 1 ? "s" : ""}
            </span>
          )}

          {/* VLM toggle button */}
          {isRunning && (
            <button
              onClick={toggleVlm}
              disabled={vlmToggling}
              className={`flex items-center gap-1.5 text-xs px-3 py-1 rounded-full
                          font-semibold border transition-all duration-200 cursor-pointer
                          disabled:opacity-50 disabled:cursor-not-allowed
                          ${vlmEnabled
                            ? "border-purple-500 bg-purple-950 text-purple-300 hover:bg-purple-900"
                            : "border-gray-600  bg-gray-800  text-gray-400  hover:bg-gray-700"
                          }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full transition-colors
                ${vlmEnabled ? "bg-purple-400" : "bg-gray-500"}`}
              />
              {vlmToggling ? "..." : vlmEnabled ? "🧠 VLM On" : "🧠 VLM Off"}
            </button>
          )}

          {/* Alert level badge */}
          <span className={`text-xs px-3 py-1 rounded-full font-semibold border ${alertColor}`}>
            {alertLevel}
          </span>

        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* ── Left Sidebar — Source Control ── */}
        <aside className="w-64 shrink-0 bg-gray-900 border-r border-gray-800
                          flex flex-col p-4 gap-4 overflow-y-auto">
          <SourcePanel api={API} status={status} />

          {/* Weapon list */}
          {hasWeapons && (
            <div>
              <p className="text-xs text-red-400 font-semibold uppercase tracking-wider mb-2">
                🔪 Detected Weapons
              </p>
              <div className="flex flex-col gap-1">
                {weapons.map((w, i) => (
                  <div key={i}
                       className="flex items-center justify-between text-xs px-2 py-1.5
                                  rounded bg-red-950 border border-red-800">
                    <span className="text-red-300 font-semibold capitalize">{w.label}</span>
                    <span className="text-red-400 font-mono">{Math.round(w.confidence * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* VRAM detail card */}
          {vram && (
            <div className="mt-auto">
              <p className="text-xs text-gray-500 font-semibold uppercase tracking-wider mb-2">
                ⚡ GPU Memory
              </p>
              <div className="bg-gray-800 rounded p-2 flex flex-col gap-1.5">
                {/* Progress bar */}
                <div className="w-full bg-gray-700 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full transition-all duration-500
                      ${vramPct > 85 ? "bg-red-500" : vramPct > 65 ? "bg-yellow-400" : "bg-green-500"}`}
                    style={{ width: `${Math.min(vramPct, 100)}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Used: <span className={vramColor}>{vram.reserved_gb?.toFixed(1)}GB</span></span>
                  <span className="text-gray-400">Total: {vram.total_gb?.toFixed(1)}GB</span>
                </div>
                {vramPct > 85 && (
                  <p className="text-xs text-red-400 mt-1">
                    ⚠️ High VRAM — consider disabling VLM
                  </p>
                )}
              </div>
            </div>
          )}
        </aside>

        {/* ── Main — Video Feed ── */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          <AlertBanner status={status} />

          <div className="flex-1 flex items-center justify-center bg-gray-950
                          overflow-hidden relative">
            {isRunning ? (
              <>
                <img
                  ref={imgRef}
                  src={`${API}/video_feed`}
                  className="max-h-full max-w-full object-contain rounded"
                  alt="Live Feed"
                />

                {/* Detection summary overlay */}
                <div className="absolute bottom-2 left-2 right-2 flex items-center
                                justify-center gap-2 flex-wrap">
                  {status?.detection_summary && (
                    <span className="text-xs px-2 py-1 rounded bg-black/70 text-green-300">
                      {status.detection_summary}
                    </span>
                  )}
                  {!vlmEnabled && (
                    <span className="text-xs px-2 py-1 rounded bg-black/70 text-gray-500">
                      🧠 VLM disabled — YOLO detection only
                    </span>
                  )}
                </div>

                {/* VLM status pill — top-left of video */}
                <div className="absolute top-2 left-2">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-semibold
                    ${vlmEnabled
                      ? "bg-purple-900/80 text-purple-300 border border-purple-700"
                      : "bg-gray-800/80  text-gray-500  border border-gray-700"}`}>
                    {vlmEnabled ? "🧠 VLM Active" : "🧠 VLM Off"}
                  </span>
                </div>
              </>
            ) : (
              <div className="text-gray-600 text-lg flex flex-col items-center gap-3 select-none">
                <span className="text-6xl">📷</span>
                <p>Select a source and press Start</p>
                <p className="text-sm text-gray-700 text-center max-w-xs">
                  YOLO26n tracks people · WeaponDetect finds weapons · SmolVLM2 confirms threats
                </p>
              </div>
            )}
          </div>
        </main>

        {/* ── Right Sidebar — VLM / Log Panels ── */}
        <aside className="w-80 shrink-0 bg-gray-900 border-l border-gray-800
                          flex flex-col overflow-hidden">

          {/* Tab bar */}
          <div className="flex shrink-0 border-b border-gray-800">
            {["vlm", "log"].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-2 text-sm font-semibold uppercase tracking-wider
                            transition-colors
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
