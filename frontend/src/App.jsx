import { useEffect, useRef, useState } from "react";
import SourcePanel from "./components/SourcePanel";
import AlertBanner from "./components/AlertBanner";
import VLMPanel    from "./components/VLMPanel";

const API = "http://localhost:8000";

const MODE_CONFIG = {
  yolo_only: {
    label:  "YOLO Only",
    icon:   "🎯",
    color:  "border-blue-500   bg-blue-950   text-blue-300   hover:bg-blue-900",
    active: "border-blue-400   bg-blue-800   text-white",
    desc:   "YOLO26n + Weapon detection. No VLM. Lowest GPU usage.",
  },
  vlm_only: {
    label:  "VLM Only",
    icon:   "🧠",
    color:  "border-purple-500 bg-purple-950 text-purple-300 hover:bg-purple-900",
    active: "border-purple-400 bg-purple-800 text-white",
    desc:   "Person tracking + passive VLM scene analysis. No weapon/proximity triggers.",
  },
  both: {
    label:  "Both",
    icon:   "⚡",
    color:  "border-green-500  bg-green-950  text-green-300  hover:bg-green-900",
    active: "border-green-400  bg-green-800  text-white",
    desc:   "Full pipeline — YOLO triggers + VLM confirms. Highest accuracy.",
  },
};

export default function App() {
  const [status,      setStatus]      = useState(null);
  const [alerts,      setAlerts]      = useState([]);
  const [persons,     setPersons]     = useState([]);
  const [weapons,     setWeapons]     = useState([]);
  const [vram,        setVram]        = useState(null);
  const [activeTab,   setActiveTab]   = useState("vlm");
  const [vlmToggling, setVlmToggling] = useState(false);
  const [intervalVal, setIntervalVal] = useState(15);     // local slider state
  const [intervalSaving, setIntervalSaving] = useState(false);
  const imgRef = useRef(null);

  // ── Polling ───────────────────────────────────────────────────────────────
  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const [s, a, p, v] = await Promise.all([
          fetch(`${API}/status`).then(r => r.json()).catch(() => null),
          fetch(`${API}/alerts`).then(r => r.json()).catch(() => []),
          fetch(`${API}/persons`).then(r => r.json()).catch(() => []),
          fetch(`${API}/vram`).then(r => r.json()).catch(() => null),
        ]);
        if (s) {
          setStatus(s);
          setWeapons(Array.isArray(s.weapon_detections) ? s.weapon_detections : []);
          // Sync slider only if not currently editing
          setIntervalVal(prev => intervalSaving ? prev : Math.round(s.vlm_interval ?? 15));
        }
        setAlerts(Array.isArray(a)  ? a.slice(-30).reverse() : []);
        setPersons(Array.isArray(p) ? p.slice(-20).reverse() : []);
        if (v && !v.error) setVram(v);
      } catch (_) {}
    }, 800);
    return () => clearInterval(interval);
  }, [intervalSaving]);

  // ── VLM Toggle ────────────────────────────────────────────────────────────
  const toggleVlm = async () => {
    if (vlmToggling) return;
    setVlmToggling(true);
    try {
      await fetch(`${API}/vlm/${status?.vlm_enabled ? "disable" : "enable"}`,
                  { method: "POST" });
    } finally {
      setVlmToggling(false);
    }
  };

  // ── Mode Switch ───────────────────────────────────────────────────────────
  const setMode = async (mode) => {
    try {
      await fetch(`${API}/mode/${mode}`, { method: "POST" });
    } catch (_) {}
  };

  // ── VLM Interval Save ─────────────────────────────────────────────────────
  const saveInterval = async (val) => {
    setIntervalSaving(true);
    try {
      await fetch(`${API}/vlm/interval?seconds=${val}`, { method: "POST" });
    } finally {
      setIntervalSaving(false);
    }
  };

  // ── Derived ───────────────────────────────────────────────────────────────
  const alertLevel = status?.alert    ?? "CLEAR";
  const vlmEnabled = status?.vlm_enabled ?? true;
  const currentMode = status?.detection_mode ?? "both";
  const isRunning  = status?.running  ?? false;
  const hasWeapons = weapons.length > 0;

  const alertColor = {
    CLEAR:  "border-green-500  bg-green-950  text-green-300",
    YELLOW: "border-yellow-400 bg-yellow-950 text-yellow-300",
    RED:    "border-red-500    bg-red-950    text-red-300",
  }[alertLevel] ?? "border-gray-500 bg-gray-900 text-gray-300";

  const vramPct   = vram?.usage_pct ?? 0;
  const vramColor = vramPct > 85 ? "text-red-400" : vramPct > 65 ? "text-yellow-400" : "text-green-400";

  return (
    <div className="flex flex-col h-screen bg-gray-950 text-white overflow-hidden">

      {/* ── Header ── */}
      <header className="flex items-center justify-between px-6 py-3 bg-gray-900
                         border-b border-gray-800 shrink-0 gap-4">
        <div className="flex items-center gap-3 min-w-0">
          <h1 className="text-xl font-bold tracking-wide whitespace-nowrap">
            🎯 YOLO26 <span className="text-blue-400">+</span> VLM Surveillance
          </h1>
          <span className="text-xs px-2 py-0.5 rounded bg-blue-900   text-blue-300   font-mono hidden sm:inline">YOLO26n</span>
          <span className="text-xs px-2 py-0.5 rounded bg-orange-900 text-orange-300 font-mono hidden md:inline">WeaponDetect</span>
          <span className="text-xs px-2 py-0.5 rounded bg-purple-900 text-purple-300 font-mono hidden lg:inline">SmolVLM2-2.2B</span>
        </div>

        <div className="flex items-center gap-2 shrink-0 flex-wrap justify-end">
          {vram && (
            <div className="flex items-center gap-1.5 text-xs px-3 py-1 rounded-full
                            border border-gray-700 bg-gray-800">
              <span className="text-gray-400">GPU</span>
              <span className={`font-semibold ${vramColor}`}>{vramPct}%</span>
              <span className="text-gray-500 hidden sm:inline">{vram.free_gb?.toFixed(1)}GB free</span>
            </div>
          )}

          {isRunning && status?.source_fps > 0 && (
            <span className="text-xs px-3 py-1 rounded-full font-semibold
                             border border-gray-600 bg-gray-800 text-gray-300">
              📹 {status.source_fps}fps
            </span>
          )}

          {isRunning && (
            <span className="text-xs px-3 py-1 rounded-full font-semibold
                             border border-gray-600 bg-gray-800 text-gray-300">
              👥 {status?.person_count ?? 0}
            </span>
          )}

          {hasWeapons && (
            <span className="flex items-center gap-1.5 text-xs px-3 py-1 rounded-full
                             font-semibold border border-red-500 bg-red-950 text-red-300 animate-pulse">
              🔪 {weapons.length} Weapon{weapons.length > 1 ? "s" : ""}
            </span>
          )}

          {isRunning && currentMode !== "yolo_only" && (
            <button
              onClick={toggleVlm}
              disabled={vlmToggling}
              className={`flex items-center gap-1.5 text-xs px-3 py-1 rounded-full
                          font-semibold border transition-all duration-200
                          disabled:opacity-50 disabled:cursor-not-allowed
                          ${vlmEnabled
                            ? "border-purple-500 bg-purple-950 text-purple-300 hover:bg-purple-900"
                            : "border-gray-600  bg-gray-800  text-gray-400  hover:bg-gray-700"}`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${vlmEnabled ? "bg-purple-400" : "bg-gray-500"}`} />
              {vlmToggling ? "..." : vlmEnabled ? "🧠 VLM On" : "🧠 VLM Off"}
            </button>
          )}

          <span className={`text-xs px-3 py-1 rounded-full font-semibold border ${alertColor}`}>
            {alertLevel}
          </span>
        </div>
      </header>

      <div className="flex flex-1 overflow-hidden">

        {/* ── Left Sidebar ── */}
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
                  <div key={i} className="flex items-center justify-between text-xs px-2 py-1.5
                                          rounded bg-red-950 border border-red-800">
                    <span className="text-red-300 font-semibold capitalize">{w.label}</span>
                    <span className="text-red-400 font-mono">{Math.round(w.confidence * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* VRAM bar */}
          {vram && (
            <div className="mt-auto">
              <p className="text-xs text-gray-500 font-semibold uppercase tracking-wider mb-2">
                ⚡ GPU Memory
              </p>
              <div className="bg-gray-800 rounded p-2 flex flex-col gap-1.5">
                <div className="w-full bg-gray-700 rounded-full h-1.5">
                  <div
                    className={`h-1.5 rounded-full transition-all duration-500
                      ${vramPct > 85 ? "bg-red-500" : vramPct > 65 ? "bg-yellow-400" : "bg-green-500"}`}
                    style={{ width: `${Math.min(vramPct, 100)}%` }}
                  />
                </div>
                <div className="flex justify-between text-xs">
                  <span className="text-gray-400">Used: <span className={vramColor}>{vram.reserved_gb?.toFixed(1)}GB</span></span>
                  <span className="text-gray-400">/{vram.total_gb?.toFixed(1)}GB</span>
                </div>
                {vramPct > 85 && (
                  <p className="text-xs text-red-400">⚠️ High VRAM — disable VLM</p>
                )}
              </div>
            </div>
          )}
        </aside>

        {/* ── Main — Video Feed ── */}
        <main className="flex-1 flex flex-col overflow-hidden min-w-0">
          <AlertBanner status={status} />

          <div className="flex-1 flex items-center justify-center bg-gray-950 overflow-hidden relative">
            {isRunning ? (
              <>
                <img
                  ref={imgRef}
                  src={`${API}/video_feed`}
                  className="max-h-full max-w-full object-contain rounded"
                  alt="Live Feed"
                />

                {/* Bottom overlay */}
                <div className="absolute bottom-2 left-2 right-2 flex items-center
                                justify-center gap-2 flex-wrap">
                  {status?.detection_summary && (
                    <span className="text-xs px-2 py-1 rounded bg-black/70 text-green-300">
                      {status.detection_summary}
                    </span>
                  )}
                  {(!vlmEnabled || currentMode === "yolo_only") && (
                    <span className="text-xs px-2 py-1 rounded bg-black/70 text-gray-500">
                      🧠 VLM inactive
                    </span>
                  )}
                </div>

                {/* Mode pill — top-left */}
                <div className="absolute top-2 left-2 flex items-center gap-1.5">
                  <span className={`text-xs px-2 py-0.5 rounded-full font-semibold border
                    ${currentMode === "both"      ? "bg-green-900/80  text-green-300  border-green-700"  :
                      currentMode === "yolo_only" ? "bg-blue-900/80   text-blue-300   border-blue-700"   :
                                                    "bg-purple-900/80 text-purple-300 border-purple-700"}`}>
                    {MODE_CONFIG[currentMode]?.icon} {MODE_CONFIG[currentMode]?.label}
                  </span>
                </div>
              </>
            ) : (
              <div className="text-gray-600 text-lg flex flex-col items-center gap-3 select-none">
                <span className="text-6xl">📷</span>
                <p>Select a source and press Start</p>
                <p className="text-sm text-gray-700 text-center max-w-xs">
                  YOLO26n · WeaponDetect · SmolVLM2-2.2B
                </p>
              </div>
            )}
          </div>
        </main>

        {/* ── Right Sidebar ── */}
        <aside className="w-80 shrink-0 bg-gray-900 border-l border-gray-800
                          flex flex-col overflow-hidden">

          {/* Tabs */}
          <div className="flex shrink-0 border-b border-gray-800">
            {["vlm", "settings"].map(tab => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-2 text-sm font-semibold uppercase tracking-wider transition-colors
                  ${activeTab === tab
                    ? "bg-gray-800 text-white border-b-2 border-blue-500"
                    : "text-gray-500 hover:text-gray-300"}`}
              >
                {tab === "vlm" ? "🤖 VLM Alerts" : "⚙️ Settings"}
              </button>
            ))}
          </div>

          <div className="flex-1 overflow-y-auto p-3">
            {activeTab === "vlm" ? (
              <VLMPanel status={status} alerts={alerts} persons={persons} />
            ) : (

              /* ── Settings Panel ── */
              <div className="flex flex-col gap-6">

                {/* Detection Mode */}
                <div>
                  <p className="text-xs text-gray-400 font-semibold uppercase tracking-wider mb-3">
                    🔍 Detection Mode
                  </p>
                  <div className="flex flex-col gap-2">
                    {Object.entries(MODE_CONFIG).map(([key, cfg]) => (
                      <button
                        key={key}
                        onClick={() => setMode(key)}
                        className={`w-full text-left px-3 py-2.5 rounded border text-sm
                                    font-semibold transition-all duration-150
                                    ${currentMode === key ? cfg.active : cfg.color}`}
                      >
                        <div className="flex items-center gap-2">
                          <span>{cfg.icon}</span>
                          <span>{cfg.label}</span>
                          {currentMode === key && (
                            <span className="ml-auto text-xs opacity-70">● Active</span>
                          )}
                        </div>
                        <p className="text-xs font-normal opacity-60 mt-0.5 pl-6">
                          {cfg.desc}
                        </p>
                      </button>
                    ))}
                  </div>
                </div>

                {/* VLM Interval */}
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <p className="text-xs text-gray-400 font-semibold uppercase tracking-wider">
                      ⏱ VLM Scene Interval
                    </p>
                    <span className="text-sm font-mono font-bold text-purple-300">
                      {intervalVal}s
                    </span>
                  </div>

                  <input
                    type="range"
                    min={5} max={120} step={5}
                    value={intervalVal}
                    onChange={e => setIntervalVal(Number(e.target.value))}
                    onMouseUp={e  => saveInterval(Number(e.target.value))}
                    onTouchEnd={e => saveInterval(Number(e.target.value))}
                    className="w-full accent-purple-500 cursor-pointer"
                  />

                  <div className="flex justify-between text-xs text-gray-600 mt-1">
                    <span>5s (frequent)</span>
                    <span>120s (rare)</span>
                  </div>

                  <p className="text-xs text-gray-500 mt-2">
                    How often VLM passively describes the scene when no threat is detected.
                    {intervalSaving && <span className="text-purple-400 ml-1">Saving...</span>}
                  </p>
                </div>

                {/* VLM Toggle (also accessible here) */}
                {currentMode !== "yolo_only" && (
                  <div>
                    <p className="text-xs text-gray-400 font-semibold uppercase tracking-wider mb-3">
                      🧠 VLM Toggle
                    </p>
                    <button
                      onClick={toggleVlm}
                      disabled={vlmToggling}
                      className={`w-full py-2 rounded border font-semibold text-sm
                                  transition-all disabled:opacity-50
                                  ${vlmEnabled
                                    ? "border-purple-500 bg-purple-950 text-purple-300 hover:bg-purple-900"
                                    : "border-gray-600  bg-gray-800  text-gray-400  hover:bg-gray-700"}`}
                    >
                      {vlmToggling ? "..." : vlmEnabled ? "🧠 VLM Enabled — Click to Disable" : "🧠 VLM Disabled — Click to Enable"}
                    </button>
                  </div>
                )}

                {/* Performance guide */}
                <div className="bg-gray-800 rounded p-3">
                  <p className="text-xs text-gray-400 font-semibold uppercase tracking-wider mb-2">
                    💡 RTX 3060 Tips
                  </p>
                  <ul className="text-xs text-gray-500 flex flex-col gap-1.5">
                    <li>🎯 YOLO Only → ~25% GPU, best for stable scenes</li>
                    <li>🧠 VLM Only → ~60% GPU, best for description tasks</li>
                    <li>⚡ Both → ~85% GPU, full threat detection</li>
                    <li>⏱ Set interval to 30s+ to reduce VLM calls</li>
                    <li>🧠 VLM Off → drops GPU by ~50% instantly</li>
                  </ul>
                </div>

              </div>
            )}
          </div>
        </aside>

      </div>
    </div>
  );
}
