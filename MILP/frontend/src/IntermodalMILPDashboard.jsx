
import React, { useMemo, useState, useEffect } from "react";
import { Play, Settings, AlertCircle, Loader2, Route, Info, Brain, Zap } from "lucide-react";
import { MapContainer, TileLayer, Marker, Popup, Polyline } from "react-leaflet";
import { Layers, Activity, Navigation } from "lucide-react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// ----------------------------
// Data & Helpers
// ----------------------------
const TERMINALS = {
  AARA: { name: "Aarau", lat: 47.39, lon: 8.05 },
  BASE: { name: "Basel", lat: 47.55, lon: 7.59 },
  BERN: { name: "Bern", lat: 46.94, lon: 7.44 },
  LAUS: { name: "Lausanne", lat: 46.52, lon: 6.63 },
  LUZE: { name: "Luzern", lat: 47.05, lon: 8.31 },
  SCHA: { name: "Schaffhausen", lat: 47.69, lon: 8.63 },
  STAB: { name: "Stabio", lat: 45.85, lon: 8.94 },
  VISP: { name: "Visp", lat: 46.29, lon: 7.88 },
  WINT: { name: "Winterthur", lat: 47.50, lon: 8.72 }
};

const MAP_CENTER = [46.82, 8.23];

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function formatCHF(x) { return new Intl.NumberFormat("en-CH").format(Math.round(x || 0)); }
function formatKCHF(x) { return `${((x || 0) / 1000).toFixed(0)}k`; }

export default function IntermodalMILPDashboard() {
  const [mnlParams, setMnlParams] = useState([]);
  const [operationalOffsets, setOperationalOffsets] = useState({
    time_multiplier: 0,
    price_multiplier: 0,
    freq_multiplier: 0,
  });

  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [executionTime, setExecutionTime] = useState(0);
  const [minFlow, setMinFlow] = useState(10);
  const [sortKey, setSortKey] = useState("flow");

  const summary = results?.summary;

  // Use Memo to handle the map and table data filtering
  const displayData = useMemo(() => {
    if (!results?.routes) return { intermodal: [], road: [], totalCount: 0 };
    
    const all = results.routes;
    // Filter intermodal routes above the flow threshold
    const intermodal = all.filter(r => 
        r.mode.toLowerCase().includes("intermodal") && (r.flow ?? 0) >= minFlow
    );
    // Filter road routes above the flow threshold
    const road = all.filter(r => 
        r.mode.toLowerCase().includes("road") && (r.flow ?? 0) >= minFlow
    );

    return { 
        intermodal, 
        road, 
        totalActive: intermodal.length + road.length 
    };
}, [results, minFlow]);

  useEffect(() => {
    const fetchParams = async () => {
      try {
        const res = await fetch("http://localhost:8000/current-parameters");
        const data = await res.json();
        if (!data.error) {
          // Filter out ASC_Other and Reliability as requested
          const filtered = data.filter(p => 
            !["ASC_Other", "reliab"].includes(p.parameter)
          );
          setMnlParams(filtered);
        }
      } catch (err) {
        console.error("Failed to fetch MNL params:", err);
      }
    };
    fetchParams();
  }, []);

  const handleOffsetChange = (param, value) => {
    setOperationalOffsets((prev) => ({ ...prev, [param]: parseFloat(value) }));
  };

  const runModel = async () => {
    setIsRunning(true);
    setError(null);
    const startTime = Date.now();
    try {
      const response = await fetch("http://localhost:8000/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(operationalOffsets),
      });
      if (!response.ok) throw new Error(await response.text() || "Backend error");
      const data = await response.json();
      setResults(data);
      setExecutionTime((Date.now() - startTime) / 1000);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsRunning(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => runModel(), 800);
    return () => clearTimeout(timer);
  }, [operationalOffsets]);

  const activeRoutes = useMemo(() => {
    if (!results?.routes) return [];
    return results.routes
      .filter((r) => (r.flow ?? 0) >= minFlow)
      .sort((a, b) => {
        if (sortKey === "frequency") return (b.frequency ?? 0) - (a.frequency ?? 0);
        if (sortKey === "revenue") return (b.revenue ?? 0) - (a.revenue ?? 0);
        return (b.flow ?? 0) - (a.flow ?? 0);
      });
  }, [results, minFlow, sortKey]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50 to-slate-50">
      <div className="mx-auto max-w-7xl px-4 py-8">
        {/* ... Header stays the same ... */}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <aside className="lg:col-span-4 space-y-6">
            
            {/* 1. Behavioral Parameters (Shifted to top and reformatted) */}
            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Brain className="h-5 w-5 text-indigo-600" />
                  <h2 className="text-lg font-bold text-slate-900">Behavioral Profile</h2>
                </div>
                <div className="group relative">
                  <Info className="h-4 w-4 text-slate-400 cursor-help" />
                  <div className="absolute right-0 bottom-full mb-2 hidden group-hover:block w-48 p-2 bg-slate-800 text-white text-[10px] rounded shadow-lg z-50">
                    Live coefficients estimated from survey data using a Multinomial Logit model.
                  </div>
                </div>
              </div>
              <div className="rounded-2xl border border-slate-200 p-4 space-y-3">
                {mnlParams.length > 0 ? mnlParams.map((p, idx) => (
                  <div key={idx} className="flex justify-between items-center text-sm">
                    <span className="text-slate-600 font-medium capitalize">
                      {p.parameter.replace('_', ' ')}
                    </span>
                    <span className="font-mono font-bold text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded">
                      {typeof p.coef === 'number' ? p.coef.toFixed(4) : p.coef}
                    </span>
                  </div>
                )) : <div className="text-xs text-slate-400 italic text-center">Loading parameters...</div>}
              </div>
            </section>

            {/* 2. Intermodal Sensitivity Sliders */}
            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Zap className="h-5 w-5 text-indigo-600" />
                  <h2 className="text-lg font-bold text-slate-900">Intermodal Sensitivity</h2>
                </div>
                <div className="group relative">
                  <Info className="h-4 w-4 text-slate-400 cursor-help" />
                  <div className="absolute right-0 bottom-full mb-2 hidden group-hover:block w-48 p-2 bg-slate-800 text-white text-[10px] rounded shadow-lg z-50">
                    Adjust the supply-side attributes (time, price, frequency) specifically for Intermodal services.
                  </div>
                </div>
              </div>
              <ParamBlock
                title="Operational Multipliers"
                rows={[
                  { label: "Time (t_hours)", value: operationalOffsets.time_multiplier, min: -0.5, max: 0.5, step: 0.05, onChange: (v) => handleOffsetChange("time_multiplier", v) },
                  { label: "Price (p_max)", value: operationalOffsets.price_multiplier, min: -0.5, max: 0.5, step: 0.05, onChange: (v) => handleOffsetChange("price_multiplier", v) },
                  { label: "Frequency (f_max)", value: operationalOffsets.freq_multiplier, min: -0.5, max: 0.5, step: 0.05, onChange: (v) => handleOffsetChange("freq_multiplier", v) },
                ]}
              />
            </section>

            {/* 3. Route Filters */}
            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-2">
                  <Route className="h-5 w-5 text-indigo-600" />
                  <h2 className="text-lg font-bold text-slate-900">Route Filters</h2>
                </div>
                <div className="group relative">
                  <Info className="h-4 w-4 text-slate-400 cursor-help" />
                  <div className="absolute right-0 bottom-full mb-2 hidden group-hover:block w-48 p-2 bg-slate-800 text-white text-[10px] rounded shadow-lg z-50">
                    Filter and sort the displayed Intermodal routes on the map and table based on volume.
                  </div>
                </div>
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex justify-between text-sm text-slate-600 mb-2">
                    <span>Min Flow</span>
                    <span className="font-mono">{minFlow} TEU</span>
                  </div>
                  <input type="range" min={0} max={200} step={5} value={minFlow} onChange={(e) => setMinFlow(parseInt(e.target.value))} className="w-full accent-indigo-600" />
                </div>
                <div>
                  <div className="text-sm text-slate-600 mb-1">Sort by</div>
                  <select value={sortKey} onChange={(e) => setSortKey(e.target.value)} className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm">
                    <option value="flow">Flow</option>
                    <option value="frequency">Frequency</option>
                    <option value="revenue">Revenue</option>
                  </select>
                </div>
              </div>
            </section>
          </aside>

          <main className="lg:col-span-8 space-y-6">
            {/* 1. Updated Intermodal KPIs */}
            <section className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <KPI 
                label="Intermodal Flow" 
                value={summary ? `${Number(summary.totalFlow).toFixed(0)} TEU` : "—"} 
                icon={<Activity className="h-4 w-4 text-emerald-600" />}
                colorClass="text-emerald-700 bg-emerald-50 ring-emerald-100"
              />
              <KPI 
                label="Intermodal Revenue" 
                value={summary ? `${formatKCHF(summary.totalRevenue)} CHF` : "—"} 
                icon={<Zap className="h-4 w-4 text-emerald-600" />}
                colorClass="text-emerald-700 bg-emerald-50 ring-emerald-100"
              />
              <KPI 
                label="Active Network Coverage" 
                value={results ? `${displayData.intermodal.length} / ${displayData.totalActive}` : "—"} 
                sublabel="Intermodal / Total Active Routes"
                icon={<Layers className="h-4 w-4 text-emerald-600" />}
                colorClass="text-emerald-700 bg-emerald-50 ring-emerald-100"
              />
            </section>

            {/* 2. Map with Flow Differentiation and Legend */}
            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 overflow-hidden">
              <div className="p-6 pb-4 flex justify-between items-center">
                <div className="flex flex-col">
                  <h2 className="text-lg font-bold text-slate-900">Intermodal Network Connectivity</h2>
                  <p className="text-xs text-slate-500">Visualization of optimized modal split and route density</p>
                </div>
                <div className="flex gap-4 text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  <div className="flex items-center gap-1">
                    <span className="w-3 h-1 bg-emerald-500 rounded-full"></span> Intermodal
                  </div>
                  <div className="flex items-center gap-1">
                    <span className="w-3 h-1 bg-slate-300 rounded-full"></span> Road
                  </div>
                </div>
              </div>
              
              <div className="h-[500px]">
                <MapContainer center={MAP_CENTER} zoom={8} className="h-full w-full">
                  <TileLayer 
                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" 
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                  />
                  
                  {/* 1. Road Flows - Light dashed lines for context */}
                  {displayData.road.map((r, idx) => {
                    const o = TERMINALS[r.origin]; 
                    const d = TERMINALS[r.destination];
                    if (!o || !d) return null;
                    return (
                      <Polyline 
                        key={`road-${idx}`} 
                        positions={[[o.lat, o.lon], [d.lat, d.lon]]} 
                        pathOptions={{ 
                          color: "#cbd5e1", 
                          weight: 1.5, 
                          opacity: 0.3, 
                          dashArray: '5, 10' 
                        }} 
                      />
                    );
                  })}

                  {/* 2. Intermodal Flows - Strong solid emerald lines scaled by frequency */}
                  {displayData.intermodal.map((r, idx) => {
                    const o = TERMINALS[r.origin]; 
                    const d = TERMINALS[r.destination];
                    if (!o || !d) return null;
                    return (
                      <Polyline 
                        key={`inter-${idx}`} 
                        positions={[[o.lat, o.lon], [d.lat, d.lon]]} 
                        pathOptions={{ 
                          color: "#10b981", 
                          weight: clamp(2 + (r.frequency || 0) * 1.5, 3, 12), 
                          opacity: 0.8 
                        }} 
                      />
                    );
                  })}
                  
                  {/* 3. Terminal Markers - Re-enabled for all defined nodes */}
                  {Object.entries(TERMINALS).map(([code, t]) => (
                    <Marker key={code} position={[t.lat, t.lon]}>
                      <Popup>
                        <div className="text-sm">
                          <div className="font-bold border-b border-slate-100 mb-1">{code} - {t.name}</div>
                          <div className="text-[10px] text-slate-500">Terminal Node</div>
                        </div>
                      </Popup>
                    </Marker>
                  ))}
                </MapContainer>
              </div>
            </section>

            {/* 3. Simplified Result Table */}
            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <h2 className="text-lg font-bold text-slate-900 mb-4">Intermodal Route Details</h2>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="border-b border-slate-200 text-slate-600">
                      <th className="text-left py-2 font-semibold">Origin</th>
                      <th className="text-left py-2 font-semibold">Destination</th>
                      <th className="text-right py-2 font-semibold">Intermodal Flow (TEU)</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayData.intermodal.map((r, idx) => (
                      <tr key={idx} className="border-b border-slate-100 hover:bg-emerald-50/30 transition-colors">
                        <td className="py-3 font-medium">{r.origin_name || r.origin}</td>
                        <td className="py-3 font-medium">{r.destination_name || r.destination}</td>
                        <td className="py-3 text-right text-emerald-700 font-bold">{Number(r.flow).toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </main>
        </div>
      </div>
    </div>
  );
}

// Updated KPI Component with custom coloring
function KPI({ label, value, sublabel, icon, colorClass }) {
  return (
    <div className={`rounded-2xl shadow-sm ring-1 p-4 ${colorClass}`}>
      <div className="flex items-center justify-between mb-1">
        <div className="text-[10px] font-bold uppercase tracking-wider opacity-70">{label}</div>
        {icon}
      </div>
      <div className="text-xl font-black">{value}</div>
      {sublabel && <div className="text-[10px] mt-1 opacity-60 font-medium">{sublabel}</div>}
    </div>
  );
}

function ParamBlock({ title, rows }) {
  return (
    <div className="rounded-2xl border border-slate-200 p-4">
      <div className="text-sm font-semibold text-slate-800 mb-3">{title}</div>
      <div className="space-y-4">
        {rows.map((row, i) => (
          <div key={i}>
            <div className="flex items-center justify-between text-xs text-slate-600 mb-1">
              <span>{row.label}</span>
              <span className={`font-mono font-bold ${row.value > 0 ? "text-emerald-600" : row.value < 0 ? "text-rose-600" : "text-slate-400"}`}>
                {row.value > 0 ? "+" : ""}{(row.value * 100).toFixed(0)}%
              </span>
            </div>
            <input type="range" min={row.min} max={row.max} step={row.step} value={row.value} onChange={(e) => row.onChange(e.target.value)} className="w-full accent-indigo-600" />
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mt-6 rounded-2xl border border-dashed border-slate-300 p-10 text-center">
      <div className="mx-auto inline-flex h-12 w-12 items-center justify-center rounded-full bg-indigo-50 ring-1 ring-indigo-100 mb-4"><Play className="h-6 w-6 text-indigo-600" /></div>
      <div className="text-base font-semibold text-slate-900">Calculating...</div>
      <div className="mt-1 text-sm text-slate-600">Optimization is in progress. Adjust multipliers to see new results.</div>
    </div>
  );
}