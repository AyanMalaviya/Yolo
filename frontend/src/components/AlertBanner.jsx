const BG = {
  CLEAR:  "bg-green-900  text-green-200  border-green-700",
  YELLOW: "bg-yellow-900 text-yellow-200 border-yellow-600",
  RED:    "bg-red-900    text-red-200    border-red-600 animate-pulse",
};

const ICON = { CLEAR: "✅", YELLOW: "⚠️", RED: "🚨" };

export default function AlertBanner({ status }) {
  if (!status || !status.running) return null;
  const alert = status.alert ?? "CLEAR";
  return (
    <div className={`shrink-0 px-4 py-2 border-b text-sm font-medium flex gap-3 items-center ${BG[alert]}`}>
      <span className="font-bold text-base whitespace-nowrap">
        {ICON[alert]} {alert}
      </span>
      <span className="opacity-80 truncate">
        {status.reason || "No active trigger"}
      </span>
    </div>
  );
}
