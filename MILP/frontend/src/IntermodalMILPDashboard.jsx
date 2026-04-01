import React, { useMemo, useState, useEffect } from "react";
import { Play, Settings, AlertCircle, Loader2, Route } from "lucide-react";
import { MapContainer, TileLayer, Marker, Popup, Polyline } from "react-leaflet";
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
        <header className="rounded-2xl bg-white/80 backdrop-blur shadow-sm ring-1 ring-slate-200 p-6 mb-6">
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div>
              <h1 className="text-2xl font-bold text-slate-900">Intermodal Network Design</h1>
              <p className="text-slate-600 mt-1">Operational Sensitivity Analysis (Percentage Change)</p>
            </div>
            <div className="flex items-center gap-3">
              {isRunning && <span className="text-sm font-medium text-indigo-600 animate-pulse">Solving MILP...</span>}
              <button onClick={runModel} disabled={isRunning} className="inline-flex items-center gap-2 rounded-xl px-5 py-3 font-semibold text-white transition-all bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-400">
                {isRunning ? <Loader2 className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />}
                Run
              </button>
            </div>
          </div>
          {executionTime > 0 && <div className="mt-3 text-sm text-slate-500">Last run: {executionTime.toFixed(2)}s</div>}
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          <aside className="lg:col-span-4 space-y-6">
            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Settings className="h-5 w-5 text-indigo-600" />
                <h2 className="text-lg font-bold text-slate-900">Sensitivity Sliders</h2>
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

            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Route className="h-5 w-5 text-indigo-600" />
                <h2 className="text-lg font-bold text-slate-900">Route Filters</h2>
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
            {error && (
              <div className="rounded-2xl bg-rose-50 ring-1 ring-rose-200 p-4 flex gap-3">
                <AlertCircle className="h-5 w-5 text-rose-600 mt-0.5" />
                <div>
                  <div className="font-semibold text-rose-800">Error</div>
                  <div className="text-sm text-rose-700">{error}</div>
                </div>
              </div>
            )}

            <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <KPI label="Total Flow" value={summary ? `${Number(summary.totalFlow).toFixed(0)} TEU` : "—"} />
              <KPI label="Revenue" value={summary ? `${formatKCHF(summary.totalRevenue)} CHF` : "—"} />
              <KPI label="Profit" value={summary ? `${formatKCHF(summary.profit)} CHF` : "—"} />
              <KPI label="Routes" value={results ? activeRoutes.length : "—"} />
            </section>

            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 overflow-hidden">
              <div className="p-6 pb-4"><h2 className="text-lg font-bold text-slate-900">Network Map</h2></div>
              <div className="h-[500px]">
                <MapContainer center={MAP_CENTER} zoom={8} className="h-full w-full">
                  <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  {Object.entries(TERMINALS).map(([code, t]) => (
                    <Marker key={code} position={[t.lat, t.lon]}>
                      <Popup><div className="text-sm font-semibold">{code} - {t.name}</div></Popup>
                    </Marker>
                  ))}
                  {activeRoutes.map((r, idx) => {
                    const o = TERMINALS[r.origin];
                    const d = TERMINALS[r.destination];
                    if (!o || !d) return null;
                    return (
                      <Polyline key={idx} positions={[[o.lat, o.lon], [d.lat, d.lon]]} 
                        pathOptions={{ color: "#4f46e5", weight: clamp(1.5 + (r.frequency || 0) * 1.4, 2, 10), opacity: clamp((r.flow || 0) / 200, 0.2, 0.9) }} 
                      />
                    );
                  })}
                </MapContainer>
              </div>
            </section>

            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <h2 className="text-lg font-bold text-slate-900 mb-4">Active Routes Table</h2>
              {!results ? <EmptyState /> : (
                <div className="overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-600">
                        <th className="text-left py-2">Origin</th>
                        <th className="text-left py-2">Destination</th>
                        <th className="text-right py-2">Flow</th>
                        <th className="text-right py-2">Freq</th>
                        <th className="text-right py-2">Revenue</th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeRoutes.map((r, idx) => (
                        <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                          <td className="py-3 font-medium">{r.origin}</td>
                          <td className="py-3 font-medium">{r.destination}</td>
                          <td className="py-3 text-right">{Number(r.flow).toFixed(1)}</td>
                          <td className="py-3 text-right">{r.frequency}</td>
                          <td className="py-3 text-right text-indigo-700">{formatCHF(r.revenue)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </section>
          </main>
        </div>
      </div>
    </div>
  );
}

function KPI({ label, value }) {
  return (
    <div className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-4">
      <div className="text-xs text-slate-600 font-medium uppercase tracking-wider">{label}</div>
      <div className="mt-1 text-xl font-bold text-slate-900">{value}</div>
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