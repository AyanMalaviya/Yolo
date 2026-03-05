const CHIP = {
  RED:    "bg-red-900    text-red-300    border border-red-700",
  YELLOW: "bg-yellow-900 text-yellow-300 border border-yellow-700",
  CLEAR:  "bg-green-900  text-green-300  border border-green-700",
};

export default function VLMPanel({ status, alerts, persons, vram }) {
  return (
    <div className="flex flex-col gap-3">

      {/* Live scene description — always visible */}
      <div className="bg-gray-800 rounded-xl p-3 border border-gray-700">
        <p className="text-xs text-green-400 font-bold uppercase mb-1">
          🔍 Live Scene (every 5s)
        </p>
        <p className="text-sm text-gray-200 leading-relaxed">
          {status?.scene_description || "Waiting for first analysis..."}
        </p>
      </div>

      {/* YOLO detection summary */}
      {status?.detection_summary && (
        <div className="bg-gray-800 rounded-lg p-2 border border-gray-700">
          <p className="text-xs text-gray-500 font-bold uppercase mb-1">YOLO Detections</p>
          <p className="text-xs text-blue-300">{status.detection_summary}</p>
        </div>
      )}

      {/* Threat alert from VLM */}
      {status?.description && status?.threat_type !== "none" && (
        <div className="bg-gray-800 rounded-xl p-3 border border-red-800">
          <p className="text-xs text-red-400 font-bold uppercase mb-1">⚠️ Threat Analysis</p>
          <p className="text-sm text-gray-200 leading-relaxed">{status.description}</p>
          <span className="mt-2 inline-block text-xs px-2 py-0.5 rounded bg-red-800 text-red-300">
            {status.threat_type}
          </span>
        </div>
      )}
      {/* New person detections */}
      {persons?.length > 0 && (
        <>
          <p className="text-xs text-gray-500 uppercase font-bold tracking-widest">
            👤 People Detected
          </p>
          {persons.map((p, i) => (
            <div key={i} className="bg-gray-800 rounded-lg p-3 flex flex-col gap-1">
              <div className="flex justify-between items-center">
                <span className="text-xs font-bold text-purple-400">ID #{p.track_id}</span>
                <span className="text-xs text-gray-500">{p.time}</span>
              </div>
              <p className="text-xs text-gray-300 leading-snug">{p.description}</p>
            </div>
          ))}
        </>
      )}


      <p className="text-xs text-gray-500 uppercase font-bold tracking-widest">Alert History</p>

      {alerts.length === 0 && (
        <p className="text-xs text-gray-600">No YELLOW/RED alerts triggered yet.</p>
      )}

      {alerts.map((a, i) => (
        <div key={i} className="bg-gray-800 rounded-lg p-3 flex flex-col gap-1.5">
          <div className="flex items-center justify-between gap-2">
            <span className={`text-xs px-2 py-0.5 rounded-full font-bold whitespace-nowrap
              ${a.alert === "RED"
                ? "bg-red-900 text-red-300 border border-red-700"
                : "bg-yellow-900 text-yellow-300 border border-yellow-700"}`}>
              {a.alert}
            </span>
            <span className="text-xs text-gray-500 shrink-0">{a.time}</span>
          </div>
          <p className="text-xs text-gray-400 leading-snug">{a.reason}</p>
          {a.vlm && <p className="text-xs text-blue-200 italic">"{a.vlm}"</p>}
        </div>
      ))}
    </div>
  );
}
