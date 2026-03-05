import { useRef, useState } from "react";

export default function SourcePanel({ api, status }) {
  const [mode, setMode]           = useState("camera");
  const [camIndex, setCamIndex]   = useState(0);
  const [videoPath, setVideoPath] = useState("");
  const [loading, setLoading]     = useState(false);
  const [vram, setVram]           = useState(null);
  const fileRef = useRef(null);

  async function start() {
    setLoading(true);
    try {
      if (mode === "camera") {
        const res = await fetch(`${api}/start/camera?index=${camIndex}`, { method: "POST" });
        if (!res.ok) throw new Error("Backend error");
      } else if (mode === "path") {
        const res = await fetch(`${api}/start/path?path=${encodeURIComponent(videoPath)}`, { method: "POST" });
        if (!res.ok) throw new Error("Backend error");
      } else if (mode === "upload" && fileRef.current?.files?.[0]) {
        const fd = new FormData();
        fd.append("file", fileRef.current.files[0]);
        const res = await fetch(`${api}/start/video`, { method: "POST", body: fd });
        if (!res.ok) throw new Error("Backend error");
      }
    } catch (err) {
      alert(`Failed to start: ${err.message}. Is the backend running on port 8000?`);
    } finally {
      setLoading(false);
    }
  }


  async function stop() {
    await fetch(`${api}/stop`, { method: "POST" });
  }

  return (
    <>
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-widest">Source</h2>

      {/* Mode selector */}
      <div className="flex rounded-lg overflow-hidden border border-gray-700">
        {["camera", "upload", "path"].map(m => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`flex-1 py-1.5 text-xs font-medium capitalize transition-colors
              ${mode === m
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
          >
            {m === "camera" ? "📷" : m === "upload" ? "📁" : "🗂️"} {m}
          </button>
        ))}
      </div>

      {/* Camera index */}
      {mode === "camera" && (
        <div className="flex flex-col gap-2">
          <label className="text-xs text-gray-400">Camera Index</label>
          <select
            value={camIndex}
            onChange={e => setCamIndex(Number(e.target.value))}
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500"
          >
            {[0, 1, 2, 3].map(i => (
              <option key={i} value={i}>Camera {i}</option>
            ))}
          </select>
        </div>
      )}

      {/* Upload */}
      {mode === "upload" && (
        <div className="flex flex-col gap-2">
          <label className="text-xs text-gray-400">Upload Video File</label>
          <input
            ref={fileRef}
            type="file"
            accept="video/*"
            className="text-xs text-gray-300 file:bg-gray-700 file:border-0
              file:text-gray-200 file:rounded file:px-2 file:py-1 file:mr-2 file:cursor-pointer
              file:text-xs cursor-pointer"
          />
        </div>
      )}

      {/* Path */}
      {mode === "path" && (
        <div className="flex flex-col gap-2">
          <label className="text-xs text-gray-400">Video File Path</label>
          <input
            type="text"
            value={videoPath}
            onChange={e => setVideoPath(e.target.value)}
            placeholder="C:\Videos\clip.mp4"
            className="bg-gray-800 border border-gray-700 rounded px-2 py-1.5 text-sm text-white
              placeholder-gray-600 focus:outline-none focus:border-blue-500"
          />
        </div>
      )}

      {/* Start */}
      <button
        onClick={start}
        disabled={loading || !!status?.running}
        className="w-full py-2 rounded-lg bg-blue-600 hover:bg-blue-500
          disabled:opacity-40 disabled:cursor-not-allowed font-semibold text-sm transition-colors"
      >
        {loading ? "Starting..." : "▶  Start"}
      </button>

      {/* Stop */}
      <button
        onClick={stop}
        disabled={!status?.running}
        className="w-full py-2 rounded-lg bg-red-700 hover:bg-red-600
          disabled:opacity-40 disabled:cursor-not-allowed font-semibold text-sm transition-colors"
      >
        ■  Stop
      </button>

      {/* Live badge */}
      {status?.running && (
        <div className="bg-gray-800 rounded-lg p-3 text-xs text-gray-400 break-all">
          <span className="text-green-400 font-bold">● LIVE</span>
          <p className="mt-1">{status.source}</p>
        </div>
      )}
      {status?.running && vram && (
      <div className="bg-gray-800 rounded-lg p-3 text-xs">
        <p className="text-gray-400 font-bold mb-1">GPU VRAM</p>
        <div className="w-full bg-gray-700 rounded-full h-2 mb-1">
          <div
            className={`h-2 rounded-full transition-all ${
              vram.usage_pct > 90 ? "bg-red-500" :
              vram.usage_pct > 70 ? "bg-yellow-400" : "bg-green-500"}`}
            style={{ width: `${vram.usage_pct}%` }}
          />
        </div>
        <p className="text-gray-400">
          {vram.allocated_gb}GB used / {vram.total_gb}GB total ({vram.usage_pct}%)
        </p>
      </div>
    )}

    </>
  );
}
