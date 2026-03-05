export default function LogPanel({ logs }) {
  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-gray-500 uppercase font-bold tracking-widest">
        People Entry / Exit
      </p>

      {logs.length === 0 && (
        <p className="text-xs text-gray-600">
          No entries yet. Cross the yellow line to log.
        </p>
      )}

      {logs.map((l, i) => (
        <div
          key={i}
          className={`rounded-lg p-3 flex items-center justify-between gap-2
            ${l.event === "ENTER"
              ? "bg-green-950 border border-green-800"
              : "bg-orange-950 border border-orange-800"}`}
        >
          <div className="min-w-0">
            <span className={`font-bold text-sm block
              ${l.event === "ENTER" ? "text-green-400" : "text-orange-400"}`}>
              {l.event === "ENTER" ? "→ ENTER" : "← EXIT"}
            </span>
            <p className="text-xs text-gray-400 mt-0.5">
              ID #{l.track_id} · {(l.confidence * 100).toFixed(0)}% conf
            </p>
          </div>
          <div className="text-right shrink-0">
            <p className="text-xs text-gray-400">{l.timestamp?.split(" ")[1]}</p>
            <span className={`text-xs px-1.5 py-0.5 rounded ${
              l.alert_state === "RED"    ? "bg-red-900 text-red-400" :
              l.alert_state === "YELLOW" ? "bg-yellow-900 text-yellow-400" :
                                           "bg-gray-800 text-gray-500"}`}>
              {l.alert_state}
            </span>
          </div>
        </div>
      ))}
    </div>
  );
}
