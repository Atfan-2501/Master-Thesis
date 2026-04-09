import { useState } from "react";
import { ChevronLeft, ChevronRight, Send, CheckCircle, Info } from "lucide-react";
import mobilityBg from './assets/project-mobility.jpg';

const FIREBASE_API_KEY = "AIzaSyD5KhzNL7kedVGj2IwIWbQNeytvV67r8RI";
const FIREBASE_PROJECT_ID = "survey-ea041";
const COLLECTION = "ch_intermodal_survey_rows";

const API_BASE_URL =  "http://localhost:8000";
// ─── SP Design Data (from Streamlit survey_schema, Section 7) ───────────────
const SP_TASKS = [
  { tNum: 1, road: { cost: 450, time: 12, reliability: 90, departures: 6 }, intermodal: { cost: 600, time: 18, reliability: 95, departures: 4 }, isCheck: false, checkType: null },
  { tNum: 2, road: { cost: 300, time: 8, reliability: 85, departures: 4 }, intermodal: { cost: 450, time: 12, reliability: 95, departures: 6 }, isCheck: false, checkType: null },
  { tNum: 3, road: { cost: 750, time: 24, reliability: 85, departures: 1 }, intermodal: { cost: 600, time: 18, reliability: 95, departures: 4 }, isCheck: false, checkType: null },
  { tNum: 4, road: { cost: 600, time: 18, reliability: 95, departures: 4 }, intermodal: { cost: 600, time: 24, reliability: 98, departures: 2 }, isCheck: false, checkType: null },
  { tNum: 5, road: { cost: 600, time: 24, reliability: 98, departures: 2 }, intermodal: { cost: 450, time: 12, reliability: 90, departures: 6 }, isCheck: false, checkType: null },
  { tNum: 6, road: { cost: 450, time: 12, reliability: 90, departures: 6 }, intermodal: { cost: 300, time: 8, reliability: 85, departures: 2 }, isCheck: false, checkType: null },
  { tNum: 7, road: { cost: 300, time: 8, reliability: 85, departures: 4 }, intermodal: { cost: 600, time: 24, reliability: 98, departures: 1 }, isCheck: false, checkType: null },
  { tNum: 8, road: { cost: 450, time: 12, reliability: 90, departures: 6 }, intermodal: { cost: 300, time: 8, reliability: 85, departures: 4 }, isCheck: false, checkType: null },
  { tNum: 9, road: { cost: 600, time: 18, reliability: 90, departures: 2 }, intermodal: { cost: 450, time: 12, reliability: 95, departures: 4 }, isCheck: true, checkType: "dominance" },
  { tNum: 10, road: { cost: 300, time: 8, reliability: 85, departures: 4 }, intermodal: { cost: 450, time: 12, reliability: 95, departures: 6 }, isCheck: true, checkType: "consistency" },
];

// ─── Survey Schema (mirrors Streamlit survey_schema exactly) ────────────────
const SURVEY_SCHEMA = [
  {
    title: "Section 1: Company Profile",
    items: [
      { id: "company_profile", label: "Company Profile (select all that apply)", type: "multiselect", options: ["Manufacturer", "Retailer/Wholesaler", "Freight Forwarder (3PL/4PL)", "Intermodal Operator (Rail, Terminal)", "Other"] },
      { id: "industry_sector", label: "Industry Sector", type: "multiselect", options: ["Automotive", "Chemicals", "Consumer Goods", "Industrial Equipment", "Dangerous goods", "Perishable goods", "Food & Beverage", "Other"] },
      { id: "annual_teu", label: "Annual Freight Volume (TEU per year)", type: "radio", options: ["< 100", "100 – 500", "500 – 1,000", "> 1,000"] },
      { id: "cost_per_teu", label: "Transport Cost per TEU (CHF per TEU)", type: "radio", options: ["< 500", "500 – 1,000", "1,000 – 2,000", "> 2,000"] },
      { id: "shipment_types", label: "Primary shipment types", type: "multiselect", options: ["Full Trailer Load (FTL)", "Containers", "Semitrailers", "Bulk Freight"] },
      { id: "distance_ch", label: "Typical transport distance within Switzerland", type: "radio", options: ["< 100 km", "100–200 km", "200–350 km", "> 350 km"] },
      { id: "mode_decider", label: "Who makes transport mode decisions?", type: "multiselect", options: ["Logistics / Supply Chain Manager", "CEO / Senior Executive", "Operations Manager", "Procurement / Purchasing", "Other"] },
    ],
  },
  {
    title: "Section 2: Current Transport Mode & Segmentation",
    items: [
      { id: "existing_mode", label: "Primary transport mode in Switzerland", type: "radio", options: ["Road", "Multimodal", "Other"] },
      { id: "secondary_modes", label: "Other modes used occasionally (select all that apply)", type: "multiselect", options: ["Road", "Multimodal", "None"] },
      { id: "reasons_current", label: "Reasons for current mode (select all that apply)", type: "multiselect", options: ["Cost efficiency", "Transit time / Speed", "Reliability (on-time)", "Flexibility", "Accessibility (terminal/first-last mile)", "Sustainability (CO₂)", "Risk avoidance", "Regulatory compliance", "Technological integration (tracking, digital booking)"] },
      { id: "use_intermodal_12m", label: "Used intermodal rail in last 12 months?", type: "radio", options: ["Yes", "No"] },
      { id: "intermodal_frequency", label: "If yes, how often do you use intermodal?", type: "radio", options: ["Occasionally (1–5/yr)", "Regularly (6+/yr)", "Always"] },
      { id: "nonuser_reasons", label: "If NOT using intermodal, why?", type: "multiselect", options: ["Cost is too high", "Rail schedules not flexible", "Transit time too long", "Terminal access inconvenient", "Need last-mile road", "Complicated booking", "Lack of tracking", "Delays", "Damage/loss", "Other"] },
      { id: "stop_using_reasons", label: "If no longer using intermodal, reasons", type: "multiselect", options: ["High costs", "Limited rail network coverage", "Long transit time", "Inflexible schedules", "Damage/loss concerns", "Complex coordination with rail operators", "Other"] },
      { id: "user_reasons", label: "If using intermodal, key reasons", type: "multiselect", options: ["Cost savings", "Appropriate transport time", "Customised service", "Environmental benefits", "Improved reliability", "Other"] },
    ],
  },
  {
    title: "Section 3: Mode Choice Factors & Preferences",
    items: [
      { id: "factor_importance", label: "Rate importance (1–5) for each factor", type: "matrix_likert5", rows: ["Cost", "Transport Time", "Service Frequency", "Punctuality (On-Time)", "Terminal Accessibility", "CO₂ Emissions / Sustainability", "Flexibility (Schedule)", "Cargo Security & Damage Risk", "Digital Tracking", "Booking Convenience"] },
      { id: "improvements", label: "What improvements would increase intermodal usage?", type: "multiselect", options: ["Lower costs", "Faster transport times", "More frequent rail schedules", "Better reliability", "Improved terminal access", "Digital tracking solutions", "Easy booking", "Improved transparency", "Other"] },
    ],
  },
  {
    title: "Section 4: Psychological / Behavioral Factors",
    items: [
      { id: "trust_overall", label: "Trust in intermodal rail vs trucking (1–5)", type: "likert5" },
      { id: "on_time_perf", label: "For users: rail meets scheduled delivery times (1–5)", type: "likert5" },
      { id: "delay_severity_single", label: "How serious are delays? (1–5)", type: "likert5" },
      { id: "flexibility_vs_truck", label: "Flexibility vs truck (1–5)", type: "likert5" },
      { id: "service_frequency_fit", label: "Adequacy of service frequency (1–5)", type: "likert5" },
      { id: "delay_severity_table", label: "Delay severity by duration", type: "matrix_ordinal", rows: ["1 Day", "2 Days", "3 Days", "4 Days", "5+ Days"], cols: ["Not Serious at All", "Slightly Serious", "Moderately Serious", "Highly Serious", "Very Serious"] },
      { id: "cost_perception", label: "Cost of rail vs road (1=Much More Expensive, 5=Much Cheaper)", type: "likert5" },
      { id: "risk_damage", label: "Risk of damage/theft/loss (1–5)", type: "likert5" },
      { id: "industry_influence", label: "Influence of industry trends/competitors (1–5)", type: "likert5" },
      { id: "sustainability_importance", label: "Importance of sustainability (1–5)", type: "likert5" },
      { id: "low_carbon_priority", label: "Priority for low-carbon modes (1–5)", type: "likert5" },
      { id: "pay_premium_co2", label: "Willingness to pay CO₂ premium (1–5)", type: "likert5" },
      { id: "stick_current_mode", label: "Likelihood to keep current mode (1–5)", type: "likert5" },
      { id: "time_pressure", label: "Decisions under time pressure (1–5)", type: "likert5" },
      { id: "pressure_to_use_rail", label: "Felt pressure to use intermodal rail (1–5)", type: "likert5" },
      { id: "extra_comm_time", label: "Extra communication time vs other modes (1–5)", type: "likert5" },
      { id: "branch_specific_need", label: "Need for branch-specific services (1–5)", type: "likert5" },
      { id: "finance_complexity", label: "Financial process complexity (1–5)", type: "likert5" },
      { id: "transparency", label: "Process transparency (1–5)", type: "likert5" },
      { id: "admin_complexity", label: "Administrative process complexity (1–5)", type: "likert5" },
      { id: "portfolio_fit", label: "Service portfolio fit (1–5)", type: "likert5" },
      { id: "terminal_access", label: "Terminal accessibility (1–5)", type: "likert5" },
      { id: "meets_requirements", label: "Meets logistics requirements (1–5)", type: "likert5" },
      { id: "booking_convenience", label: "Booking convenience in CH (1–5)", type: "likert5" },
      { id: "digital_tracking_importance", label: "Importance of digital tracking (1–5)", type: "likert5" },
      { id: "comfort_new_solutions", label: "Comfort with trying intermodal rail (1–5)", type: "likert5" },
      { id: "concerns", label: "Top concerns about intermodal rail", type: "multiselect", options: ["Unreliable schedules", "Poor customer service", "Lack of flexibility", "Terminal access issues", "Higher cost", "Other"] },
      { id: "psych_open", label: "Other psychological/behavioral factors", type: "textarea" },
    ],
  },
  {
    title: "Section 5: Environmental Impact",
    items: [
      { id: "co2_importance", label: "Importance of CO₂ reduction (1–5)", type: "likert5" },
      { id: "sustainable_energy", label: "Importance of sustainable/alternative energy (1–5)", type: "likert5" },
    ],
  },
  {
    title: "Section 6: Policy & Regulatory",
    items: [
      { id: "policy_encouragement", label: "Policies that would encourage Intermodal (select all)", type: "multiselect", options: ["Subsidies for rail transport", "Carbon tax on road transport", "Investment in rail infrastructure", "Digital freight platforms for easier booking", "More flexible rail schedules", "Priority access for time-sensitive freight", "Branch-specific intermodal services", "Other"] },
      { id: "aware_regulations", label: "Aware of Swiss transport regulations impacting mode choice?", type: "radio", options: ["Yes", "No"] },
      { id: "pilot_test", label: "Open to pilot-testing new intermodal solutions?", type: "radio", options: ["Yes", "No"] },
      { id: "govt_influence", label: "Influence of government campaigns (1–5)", type: "likert5" },
      { id: "policy_suggestions", label: "Policy suggestions (open text)", type: "textarea" },
    ],
  },
  {
    title: "Section 7: Stated-Preference Choice Tasks",
    items: [], // Rendered via SP_TASKS
  },
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

function genId() {
  return `r-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function toFirestoreFields(answers, respondentId) {
  const fields = {
    respondent_id: { stringValue: respondentId },
    submitted_at_utc: { stringValue: new Date().toISOString() },
  };

  for (const [key, value] of Object.entries(answers)) {
    if (value === null || value === undefined) {
      fields[key] = { nullValue: null };
    } else if (Array.isArray(value)) {
      fields[key] = {
        arrayValue: {
          values: value.length
            ? value.map((v) => ({ stringValue: String(v) }))
            : [{ nullValue: null }],
        },
      };
    } else if (typeof value === "object") {
      // Flatten matrix answers: factor_importance__Cost → integerValue
      for (const [subKey, subVal] of Object.entries(value)) {
        const flatKey = `${key}__${subKey.replace(/[^a-zA-Z0-9]/g, "_")}`;
        fields[flatKey] =
          typeof subVal === "number"
            ? { integerValue: String(subVal) }
            : { stringValue: String(subVal) };
      }
    } else if (typeof value === "number") {
      fields[key] = { integerValue: String(value) };
    } else {
      fields[key] = { stringValue: String(value) };
    }
  }
  return fields;
}

// ─── Question Components ──────────────────────────────────────────────────────

const cardCls =
  "bg-white rounded-xl p-4 mb-4 shadow-sm ring-1 ring-slate-200";
const labelCls = "font-extrabold text-slate-900 text-sm mb-2";

function BtnPill({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-lg border text-sm font-medium transition-all ${
        active
          ? "bg-indigo-600 text-white border-indigo-600"
          : "bg-white border-slate-200 text-slate-600 hover:border-indigo-400"
      }`}
    >
      {children}
    </button>
  );
}

function RadioQuestion({ id, label, options, value, onChange }) {
  return (
    <div className={cardCls}>
      <div className={labelCls}>{label}</div>
      <div className="flex flex-wrap gap-2">
        {options.map((opt) => (
          <BtnPill key={opt} active={value === opt} onClick={() => onChange(id, opt)}>
            {opt}
          </BtnPill>
        ))}
      </div>
    </div>
  );
}

function MultiSelectQuestion({ id, label, options, value = [], onChange }) {
  const toggle = (opt) => {
    const next = (value || []).includes(opt)
      ? value.filter((v) => v !== opt)
      : [...(value || []), opt];
    onChange(id, next);
  };
  return (
    <div className={cardCls}>
      <div className={labelCls}>{label}</div>
      <div className="flex flex-wrap gap-2">
        {options.map((opt) => (
          <BtnPill key={opt} active={(value || []).includes(opt)} onClick={() => toggle(opt)}>
            {opt}
          </BtnPill>
        ))}
      </div>
    </div>
  );
}

function LikertQuestion({ id, label, value, onChange }) {
  return (
    <div className={cardCls}>
      <div className={labelCls}>{label}</div>
      <div className="flex items-center gap-2 mt-1">
        <span className="text-xs text-slate-400 w-16 text-right">1 (low)</span>
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            onClick={() => onChange(id, n)}
            className={`w-10 h-10 rounded-lg border font-bold text-sm transition-all ${
              value === n
                ? "bg-indigo-600 text-white border-indigo-600"
                : "bg-white border-slate-200 text-slate-600 hover:border-indigo-400"
            }`}
          >
            {n}
          </button>
        ))}
        <span className="text-xs text-slate-400 w-16">5 (high)</span>
      </div>
    </div>
  );
}

function MatrixLikert5Question({ id, label, rows, value = {}, onChange }) {
  const updateRow = (row, n) => onChange(id, { ...value, [row]: n });
  return (
    <div className={cardCls}>
      <div className={labelCls}>{label}</div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs mt-2">
          <thead>
            <tr>
              <th className="text-left p-1 text-slate-500 font-medium w-44">Factor</th>
              {[1, 2, 3, 4, 5].map((n) => (
                <th key={n} className="text-center p-1 text-slate-500 font-medium w-10">
                  {n}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row} className="border-t border-slate-100">
                <td className="p-1 text-slate-700 font-medium">{row}</td>
                {[1, 2, 3, 4, 5].map((n) => (
                  <td key={n} className="text-center p-1">
                    <button
                      onClick={() => updateRow(row, n)}
                      className={`w-8 h-8 rounded-lg border text-xs font-bold transition-all mx-auto block ${
                        value[row] === n
                          ? "bg-indigo-600 text-white border-indigo-600"
                          : "bg-white border-slate-200 text-slate-500 hover:border-indigo-400"
                      }`}
                    >
                      {n}
                    </button>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function MatrixOrdinalQuestion({ id, label, rows, cols, value = {}, onChange }) {
  const updateRow = (row, col) => onChange(id, { ...value, [row]: col });
  return (
    <div className={cardCls}>
      <div className={labelCls}>{label}</div>
      <p className="text-xs text-slate-500 italic mb-2">Please select one option per row</p>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left p-1 text-slate-500 font-medium w-16"></th>
              {cols.map((col) => (
                <th key={col} className="text-center p-1 text-slate-500 font-medium leading-tight">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row} className="border-t border-slate-100">
                <td className="p-1 text-slate-700 font-medium">{row}</td>
                {cols.map((col) => (
                  <td key={col} className="text-center p-1">
                    <button
                      onClick={() => updateRow(row, col)}
                      className={`w-5 h-5 rounded-full border-2 mx-auto transition-all ${
                        value[row] === col
                          ? "bg-indigo-600 border-indigo-600"
                          : "border-slate-300 hover:border-indigo-400 bg-white"
                      }`}
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function TextAreaQuestion({ id, label, value = "", onChange }) {
  return (
    <div className={cardCls}>
      <div className={labelCls}>{label}</div>
      <textarea
        value={value}
        onChange={(e) => onChange(id, e.target.value)}
        rows={3}
        placeholder="Your answer..."
        className="w-full mt-1 p-2 border border-slate-200 rounded-lg text-sm text-slate-700 resize-none focus:outline-none focus:ring-2 focus:ring-indigo-300"
      />
    </div>
  );
}

const SP_ATTRS = [
  { key: "cost", label: "Cost (CHF/TEU)" },
  { key: "time", label: "Transit time (h)" },
  { key: "reliability", label: "On-time reliability (% within ±2h)" },
  { key: "departures", label: "Departures per day" },
];

function SPTask({ task, value, onChange }) {
  const key = `sp_choice_${task.tNum}`;
  let title = `Task ${task.tNum}: Please choose your preferred option`;
  if (task.isCheck) {
    title = `Check (${task.checkType === "dominance" ? "A – Dominance" : "B – Consistency"}): Please choose your preferred option`;
  }
  return (
    <div className="mb-6 pb-6 border-b border-slate-200 last:border-0">
      <div className="font-extrabold text-slate-900 text-sm bg-white/90 inline-block px-2 py-1 rounded-lg mb-3">
        {title}
      </div>
      <table className="w-full text-sm bg-white rounded-lg overflow-hidden border border-slate-200 mb-3">
        <thead>
          <tr className="bg-slate-50">
            <th className="p-2 border border-slate-200 text-left text-slate-600">Attribute</th>
            <th className="p-2 border border-slate-200 text-center text-slate-600">Road</th>
            <th className="p-2 border border-slate-200 text-center text-slate-600">Intermodal</th>
          </tr>
        </thead>
        <tbody>
          {SP_ATTRS.map((attr) => (
            <tr key={attr.key}>
              <td className="p-2 border border-slate-200 font-medium text-slate-700">{attr.label}</td>
              <td className="p-2 border border-slate-200 text-center">{task.road[attr.key]}</td>
              <td className="p-2 border border-slate-200 text-center">{task.intermodal[attr.key]}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="flex gap-2">
        {["Road", "Intermodal", "Other"].map((alt) => (
          <button
            key={alt}
            onClick={() => onChange(key, alt)}
            className={`flex-1 p-2 rounded-lg border-2 font-bold text-sm transition-all ${
              value === alt
                ? "bg-indigo-600 text-white border-indigo-600"
                : "bg-white border-slate-200 text-slate-600 hover:border-indigo-300"
            }`}
          >
            {alt}
          </button>
        ))}
      </div>
    </div>
  );
}

// ─── Main Survey Component ───────────────────────────────────────────────────

export default function RailFreightSurvey() {
  const [pageIdx, setPageIdx] = useState(0);
  const [answers, setAnswers] = useState({});
  const [respondentId] = useState(() => genId());
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [submitError, setSubmitError] = useState(null);

  const totalPages = SURVEY_SCHEMA.length;
  const currentSection = SURVEY_SCHEMA[pageIdx];
  const isLastPage = pageIdx === totalPages - 1;
  const progress = Math.round(((pageIdx + 1) / totalPages) * 100);

  const updateAnswer = (id, value) => setAnswers((prev) => ({ ...prev, [id]: value }));

  const handleSubmit = async () => {
    setIsSubmitting(true);
    setSubmitError(null);
    try {
      const fields = toFirestoreFields(answers, respondentId);
      const url = `https://firestore.googleapis.com/v1/projects/${FIREBASE_PROJECT_ID}/databases/(default)/documents/${COLLECTION}?key=${FIREBASE_API_KEY}`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ fields }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      // Optional: trigger backend recompute
      fetch(API_BASE_URL, { method: "POST" }).catch(() => {});
      setSubmitted(true);
    } catch (err) {
      console.error(err);
      setSubmitError("Submission failed. Please try again.");
    } finally {
      setIsSubmitting(false);
    }
  };

  const renderQuestion = (item) => {
    const val = answers[item.id];
    const props = { key: item.id, ...item, value: val, onChange: updateAnswer };
    switch (item.type) {
      case "radio": return <RadioQuestion {...props} />;
      case "multiselect": return <MultiSelectQuestion {...props} value={val || []} />;
      case "likert5": return <LikertQuestion {...props} />;
      case "matrix_likert5": return <MatrixLikert5Question {...props} value={val || {}} />;
      case "matrix_ordinal": return <MatrixOrdinalQuestion {...props} value={val || {}} />;
      case "textarea": return <TextAreaQuestion {...props} value={val || ""} />;
      default: return null;
    }
  };

  // ── Success screen ──
  if (submitted) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[url('https://images.unsplash.com/photo-1464802686167-b939a6910659?w=1200&q=80')] bg-cover bg-center p-4">
        <div className="bg-white/90 backdrop-blur-md p-10 rounded-xl text-center max-w-lg shadow-xl">
          <CheckCircle className="mx-auto text-emerald-500 mb-4" size={52} />
          <h2 className="text-2xl font-bold text-emerald-600">Thank You!</h2>
          <p className="text-slate-700 mt-2">Your response has been recorded and the model is updating.</p>
        </div>
      </div>
    );
  }

  // ── Survey screen ──
  return (
    <div
      className="min-h-screen py-10 px-4 bg-cover bg-fixed bg-center"
      style={{ backgroundImage: `url(${mobilityBg})` }}
    >
      <div className="max-w-2xl mx-auto">
        {/* ── Header ── */}
        <div className="text-center mb-6">
          <img
            src="https://logowik.com/content/uploads/images/eth-zurich1144.jpg"
            width="90"
            className="mx-auto mb-3 rounded"
            alt="ETH Zurich"
          />
          <h1 className="text-3xl font-bold text-white drop-shadow">
            Survey: Swiss Intermodal Freight
          </h1>
        </div>

        {/* ── About card (page 0 only) ── */}
        {pageIdx === 0 && (
          <div className="bg-white/90 backdrop-blur-md rounded-xl p-4 shadow-sm ring-1 ring-slate-200 mb-4">
            <div className="flex items-start gap-2">
              <Info size={18} className="text-indigo-500 flex-shrink-0 mt-0.5" />
              <p className="text-sm text-slate-700">
                This pilot study explores short-distance intermodal rail adoption in Switzerland. Your responses will inform a discrete choice model and an optimisation study. It takes ~8–10 minutes. Thank you!
              </p>
            </div>
          </div>
        )}

        {/* ── Progress ── */}
        <div className="flex items-center gap-3 mb-4">
          <div className="flex-1 bg-white/30 rounded-full h-2">
            <div
              className="bg-indigo-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <span className="text-white text-xs font-medium whitespace-nowrap">
            {pageIdx + 1} / {totalPages}
          </span>
        </div>

        {/* ── Section card ── */}
        <div className="bg-white/90 backdrop-blur-md rounded-xl p-5 shadow-sm ring-1 ring-slate-200 mb-4">
          <h2 className="text-lg font-bold text-slate-900 border-b border-slate-200 pb-2 mb-4">
            {currentSection.title}
          </h2>

          {/* Section 7 – SP tasks */}
          {isLastPage ? (
            <>
              <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-3 mb-5 text-sm text-slate-700">
                <strong>Reference shipment (for the following tasks):</strong>
                <br />
                Domestic move, <strong>OD 120–180 km</strong>, <strong>1× 20' container</strong> (general cargo),{" "}
                <strong>medium urgency</strong> (delivery within 24 h acceptable). Compare options below.{" "}
                <em>On-time reliability = % delivered within ±2 hours of promised time.</em>
              </div>
              {SP_TASKS.map((task) => (
                <SPTask
                  key={task.tNum}
                  task={task}
                  value={answers[`sp_choice_${task.tNum}`]}
                  onChange={updateAnswer}
                />
              ))}
            </>
          ) : (
            /* Regular sections */
            <>{currentSection.items.map((item) => renderQuestion(item))}</>
          )}

          {submitError && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-2 text-sm text-red-600 mb-4">
              {submitError}
            </div>
          )}

          {/* ── Navigation ── */}
          <div className="flex justify-between items-center mt-6 pt-4 border-t border-slate-200">
            <button
              disabled={pageIdx === 0}
              onClick={() => setPageIdx((p) => p - 1)}
              className={`flex items-center gap-1 font-bold transition-all ${
                pageIdx === 0
                  ? "text-slate-300 cursor-not-allowed"
                  : "text-slate-600 hover:text-indigo-700"
              }`}
            >
              <ChevronLeft size={20} />
              Previous
            </button>

            {isLastPage ? (
              <button
                onClick={handleSubmit}
                disabled={isSubmitting}
                className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white px-6 py-2 rounded-lg font-bold transition-all"
              >
                {isSubmitting ? "Submitting…" : "Submit"}
                <Send size={16} />
              </button>
            ) : (
              <button
                onClick={() => setPageIdx((p) => p + 1)}
                className="flex items-center gap-2 bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-lg font-bold transition-all"
              >
                Next
                <ChevronRight size={20} />
              </button>
            )}
          </div>
        </div>

        {/* ── Footer ── */}
        <div className="bg-white/80 backdrop-blur p-3 rounded-xl text-center text-xs font-medium text-slate-600">
          © 2025 ETH Zurich. All rights reserved.
        </div>
      </div>
    </div>
  );
}