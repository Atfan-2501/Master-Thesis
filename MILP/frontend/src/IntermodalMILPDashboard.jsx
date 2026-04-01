import React, { useMemo, useState, useEffect } from "react"; // Added useEffect
import { Play, Settings, Info, AlertCircle, Loader2, MapPin, Route } from "lucide-react";
import { MapContainer, TileLayer, Marker, Popup, Polyline } from "react-leaflet";
import L from "leaflet";

// ----------------------------
// Data
// ----------------------------
// Reduced Terminal List based on uploaded OD pairs
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

// ----------------------------
// Helpers
// ----------------------------
function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

function formatCHF(x) {
  if (!Number.isFinite(x)) return "-";
  return new Intl.NumberFormat("en-CH").format(Math.round(x));
}

function formatKCHF(x) {
  if (!Number.isFinite(x)) return "-";
  return `${(x / 1000).toFixed(0)}k`;
}

function calcWTP({ time_mean, freq_mean, cost_mean }) {
  if (!cost_mean || cost_mean === 0) return { time: 0, frequency: 0 };
  return {
    time: -((time_mean / cost_mean) * 100),
    frequency: -((freq_mean / cost_mean) * 100),
  };
}

const DefaultIcon = L.icon({
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
  iconSize: [25, 41],
  iconAnchor: [12, 41],
});
L.Marker.prototype.options.icon = DefaultIcon;




export default function IntermodalMILPDashboard() {
  const [utilityParams, setUtilityParams] = useState({
    cost_mean: -0.2,
    cost_std: 0.1,
    time_mean: -0.05,
    time_std: 0.02,
    freq_mean: -0.15,
    freq_std: 0.05,
    asc_road: -0.3,
    asc_other: -4.0,
  });

  const [isRunning, setIsRunning] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [executionTime, setExecutionTime] = useState(0);
  const summary = results?.summary;

  const [minFlow, setMinFlow] = useState(10);
  const [sortKey, setSortKey] = useState("flow");

  const wtp = useMemo(() => calcWTP(utilityParams), [utilityParams]);

  const handleParamChange = (param, value) => {
    setUtilityParams((prev) => ({ ...prev, [param]: parseFloat(value) }));
  };

  const runModel = async () => {
    setIsRunning(true);
    setError(null);
    const startTime = Date.now();

    try {
      const response = await fetch("http://localhost:8000/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(utilityParams),
      });

      if (!response.ok) {
        const text = await response.text();
        throw new Error(text || "Backend error");
      }

      const data = await response.json();
      setResults(data);
      setExecutionTime((Date.now() - startTime) / 1000);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsRunning(false);
    }
  };

  // --- AUTO-RUN EFFECT WITH DEBOUNCE ---
  useEffect(() => {
    const timer = setTimeout(() => {
      runModel();
    }, 500); // Wait 500ms after last slider change before running
    return () => clearTimeout(timer);
  }, [utilityParams]); // Trigger run whenever any Utility Parameter changes

  const activeRoutes = useMemo(() => {
    if (!results?.routes) return [];
    const filtered = results.routes.filter((r) => (r.flow ?? 0) >= minFlow);
    const sorted = [...filtered].sort((a, b) => {
      if (sortKey === "frequency") return (b.frequency ?? 0) - (a.frequency ?? 0);
      if (sortKey === "revenue") return (b.revenue ?? 0) - (a.revenue ?? 0);
      return (b.flow ?? 0) - (a.flow ?? 0);
    });
    return sorted;
  }, [results, minFlow, sortKey]);

  const strokeForRoute = (route) => {
    const f = route.frequency ?? 0;
    return clamp(1.5 + f * 1.4, 2, 10);
  };

  const opacityForRoute = (route) => {
    const flow = route.flow ?? 0;
    return clamp(flow / 200, 0.15, 0.9);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-indigo-50 to-slate-50">
      <div className="mx-auto max-w-7xl px-4 py-8">
        <header className="rounded-2xl bg-white/80 backdrop-blur shadow-sm ring-1 ring-slate-200 p-6">
          <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
            <div>
              <h1 className="text-2xl md:text-3xl font-bold text-slate-900">
                Intermodal Network Design Dashboard
              </h1>
              <p className="text-slate-600 mt-1">
                The model now recalculates automatically when you adjust utility sliders.
              </p>
            </div>
            {/* Run Model button remains as a manual override/status indicator */}
            <div className="flex items-center gap-3">
               {isRunning && <span className="text-sm font-medium text-indigo-600 animate-pulse">Solving Optimization...</span>}
                <button
                onClick={runModel}
                disabled={isRunning}
                className={`inline-flex items-center justify-center gap-2 rounded-xl px-5 py-3 font-semibold text-white shadow-sm transition-all \
                    ${isRunning ? "bg-slate-400 cursor-not-allowed" : "bg-indigo-600 hover:bg-indigo-700"}`}
                >
                {isRunning ? <Loader2 className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />}
                {isRunning ? "Running…" : "Run Model"}
                </button>
            </div>
          </div>

          {executionTime > 0 && (
            <div className="mt-3 text-sm text-slate-500">Last run: {executionTime.toFixed(2)}s</div>
          )}
        </header>

        <div className="mt-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
          <aside className="lg:col-span-4 space-y-6">
            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Settings className="h-5 w-5 text-indigo-600" />
                <h2 className="text-lg font-bold text-slate-900">Utility Parameters</h2>
              </div>

              <div className="space-y-5">
                <ParamBlock
                  title="Cost (per 100 CHF)"
                  rows={[
                    { label: "Mean (μ)", value: utilityParams.cost_mean, min: -0.5, max: 0, step: 0.01, onChange: (v) => handleParamChange("cost_mean", v) },
                    { label: "Std Dev (σ)", value: utilityParams.cost_std, min: 0, max: 0.3, step: 0.01, onChange: (v) => handleParamChange("cost_std", v) },
                  ]}
                />

                <ParamBlock
                  title="Time (per hour)"
                  rows={[
                    { label: "Mean (μ)", value: utilityParams.time_mean, min: -0.15, max: 0, step: 0.005, onChange: (v) => handleParamChange("time_mean", v) },
                    { label: "Std Dev (σ)", value: utilityParams.time_std, min: 0, max: 0.05, step: 0.005, onChange: (v) => handleParamChange("time_std", v) },
                  ]}
                />

                <ParamBlock
                  title="Frequency (per departure)"
                  rows={[
                    { label: "Mean (μ)", value: utilityParams.freq_mean, min: -0.3, max: 0, step: 0.01, onChange: (v) => handleParamChange("freq_mean", v) },
                    { label: "Std Dev (σ)", value: utilityParams.freq_std, min: 0, max: 0.1, step: 0.005, onChange: (v) => handleParamChange("freq_std", v) },
                  ]}
                />

                <ParamBlock
                  title="Alternative-Specific Constants"
                  rows={[
                    { label: "ASC Road", value: utilityParams.asc_road, min: -1, max: 0, step: 0.05, onChange: (v) => handleParamChange("asc_road", v) },
                    { label: "ASC Other", value: utilityParams.asc_other, min: -6, max: -2, step: 0.1, onChange: (v) => handleParamChange("asc_other", v) },
                  ]}
                />
              </div>
            </section>

            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Info className="h-5 w-5 text-indigo-600" />
                <h2 className="text-lg font-bold text-slate-900">Willingness to Pay</h2>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <StatCard label="Time Value" value={`${wtp.time.toFixed(1)} CHF/hour`} />
                <StatCard label="Frequency Value" value={`${wtp.frequency.toFixed(1)} CHF/dep`} />
              </div>
              <p className="mt-4 text-xs text-slate-500 leading-relaxed">
                Computed as WTP = −β_attribute / β_cost.
              </p>
            </section>

            {/* Filters (Do not trigger model run) */}
            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Route className="h-5 w-5 text-indigo-600" />
                <h2 className="text-lg font-bold text-slate-900">Route Filters</h2>
              </div>
              <div className="space-y-4">
                <div>
                  <div className="flex items-center justify-between text-sm text-slate-600">
                    <span>Minimum flow</span>
                    <span className="font-mono">{minFlow.toFixed(0)} TEU</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={200}
                    step={5}
                    value={minFlow}
                    onChange={(e) => setMinFlow(parseFloat(e.target.value))}
                    className="mt-2 w-full accent-indigo-600"
                  />
                </div>
                <div>
                  <div className="text-sm text-slate-600 mb-1">Sort by</div>
                  <select
                    value={sortKey}
                    onChange={(e) => setSortKey(e.target.value)}
                    className="w-full rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm"
                  >
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
              <KPI label="Total Flow" value={summary ? `${Number(summary.totalFlow ?? 0).toFixed(0)} TEU` : "—"} />
              <KPI label="Revenue" value={summary ? `${formatKCHF(summary.totalRevenue ?? 0)} CHF` : "—"} />
              <KPI label="Profit" value={summary ? `${formatKCHF(summary.profit ?? 0)} CHF` : "—"} />
              <KPI label="Routes" value={results?.routes ? `${activeRoutes.length}` : "—"} />
            </section>

            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 overflow-hidden">
              <div className="p-6 pb-4">
                <h2 className="text-lg font-bold text-slate-900">Network Map</h2>
              </div>
              <div className="h-[540px]">
                <MapContainer center={MAP_CENTER} zoom={8} scrollWheelZoom className="h-full w-full">
                  <TileLayer url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />
                  {Object.entries(TERMINALS).map(([code, t]) => (
                    <Marker key={code} position={[t.lat, t.lon]}>
                      <Popup>
                        <div className="text-sm font-semibold">{code} - {t.name}</div>
                      </Popup>
                    </Marker>
                  ))}
                  {results && activeRoutes.map((r, idx) => {
                    const o = TERMINALS[r.origin];
                    const d = TERMINALS[r.destination];
                    if (!o || !d) return null;
                    return (
                      <Polyline
                        key={`${r.origin}-${r.destination}-${idx}`}
                        positions={[[o.lat, o.lon], [d.lat, d.lon]]}
                        pathOptions={{
                          color: "#4f46e5",
                          weight: strokeForRoute(r),
                          opacity: opacityForRoute(r),
                        }}
                      />
                    );
                  })}
                </MapContainer>
              </div>
            </section>

            <section className="rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 p-6">
              <h2 className="text-lg font-bold text-slate-900">Active Routes Table</h2>
              {!results ? (
                <EmptyState />
              ) : (
                <div className="mt-4 overflow-x-auto">
                  <table className="min-w-full text-sm">
                    <thead>
                      <tr className="border-b border-slate-200 text-slate-600">
                        <th className="text-left py-2 font-semibold">Origin</th>
                        <th className="text-left py-2 font-semibold">Destination</th>
                        <th className="text-right py-2 font-semibold">Flow</th>
                        <th className="text-right py-2 font-semibold">Freq</th>
                        <th className="text-right py-2 font-semibold">Revenue</th>
                      </tr>
                    </thead>
                    <tbody>
                      {activeRoutes.map((r, idx) => (
                        <tr key={idx} className="border-b border-slate-100 hover:bg-slate-50">
                          <td className="py-3 font-medium">{r.origin}</td>
                          <td className="py-3 font-medium">{r.destination}</td>
                          <td className="py-3 text-right">{(r.flow ?? 0).toFixed(1)}</td>
                          <td className="py-3 text-right">{r.frequency ?? 0}</td>
                          <td className="py-3 text-right text-indigo-700">{formatCHF(r.revenue ?? 0)}</td>
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
      <div className="text-xs text-slate-600">{label}</div>
      <div className="mt-1 text-xl font-bold text-slate-900">{value}</div>
    </div>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="rounded-2xl bg-slate-50 ring-1 ring-slate-200 p-4">
      <div className="text-xs text-slate-600">{label}</div>
      <div className="mt-1 text-base font-semibold text-slate-900">{value}</div>
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
            <div className="flex items-center justify-between text-xs text-slate-600">
              <span>{row.label}</span>
              <span className="font-mono">{Number(row.value).toFixed(3)}</span>
            </div>
            <input
              type="range"
              min={row.min}
              max={row.max}
              step={row.step}
              value={row.value}
              onChange={(e) => row.onChange(e.target.value)}
              className="mt-2 w-full accent-indigo-600"
            />
          </div>
        ))}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="mt-6 rounded-2xl border border-dashed border-slate-300 p-10 text-center">
      <div className="mx-auto inline-flex h-12 w-12 items-center justify-center rounded-full bg-indigo-50 ring-1 ring-indigo-100">
        <Play className="h-6 w-6 text-indigo-600" />
      </div>
      <div className="mt-4 text-base font-semibold text-slate-900">Calculating...</div>
      <div className="mt-1 text-sm text-slate-600">Adjust parameters to see the optimized network results.</div>
    </div>
  );
}