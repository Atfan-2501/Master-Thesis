import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight, Send, CheckCircle, Info } from 'lucide-react';
import { db } from './firebase_config'; // You will need to create this file
import { collection, addDoc } from 'firebase/firestore';

// --- Sub-Components for Different Field Types ---

const ChoiceTable = ({ task, onSelect, selectedAlt }) => (
  <div className="bg-white rounded-xl border border-slate-200 overflow-hidden mb-4">
    <table className="min-w-full text-sm text-left">
      <thead className="bg-slate-50 border-b border-slate-200">
        <tr>
          <th className="px-4 py-2 font-bold text-slate-700">Attribute</th>
          <th className="px-4 py-2 font-bold text-indigo-600">Road</th>
          <th className="px-4 py-2 font-bold text-emerald-600">Intermodal</th>
        </tr>
      </thead>
      <tbody className="divide-y divide-slate-100">
        {[
          { label: "Cost (CHF/TEU)", key: "cost_chf_teu" },
          { label: "Transit time (h)", key: "time_h" },
          { label: "Reliability (%)", key: "ontime_p" },
          { label: "Freq (deps/day)", key: "dep_per_day" }
        ].map((attr) => (
          <tr key={attr.key}>
            <td className="px-4 py-2 font-medium text-slate-500">{attr.label}</td>
            <td className="px-4 py-2 font-mono text-slate-900">{task.road[attr.key]}</td>
            <td className="px-4 py-2 font-mono text-slate-900">{task.intermodal[attr.key]}</td>
          </tr>
        ))}
      </tbody>
    </table>
    <div className="p-4 bg-slate-50 flex gap-2">
      {["Road", "Intermodal", "Other"].map((alt) => (
        <button
          key={alt}
          onClick={() => onSelect(alt)}
          className={`flex-1 py-2 rounded-lg font-bold text-xs transition-all border-2 ${
            selectedAlt === alt ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-slate-600 border-slate-200 hover:border-indigo-300"
          }`}
        >
          {alt}
        </button>
      ))}
    </div>
  </div>
);

const LikertRow = ({ label, value, onChange }) => (
  <div className="mb-6">
    <div className="text-sm font-semibold text-slate-700 mb-2">{label}</div>
    <div className="flex items-center gap-4">
      <span className="text-[10px] text-slate-400 font-bold">LOW</span>
      <div className="flex-1 flex justify-between gap-1">
        {[1, 2, 3, 4, 5].map((num) => (
          <button
            key={num}
            onClick={() => onChange(num)}
            className={`w-10 h-10 rounded-full border-2 font-bold transition-all ${
              value === num ? "bg-indigo-600 border-indigo-600 text-white" : "bg-white border-slate-200 text-slate-400 hover:border-indigo-300"
            }`}
          >
            {num}
          </button>
        ))}
      </div>
      <span className="text-[10px] text-slate-400 font-bold">HIGH</span>
    </div>
  </div>
);

// --- Main Survey Logic ---

export default function RailFreightSurvey() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);
  const [answers, setAnswers] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);

  // Replicating Sections 1-7 from Streamlit Schema
  const sections = [
    { title: "Introduction", type: "intro" },
    { title: "Company Profile", fields: [
        { id: "annual_teu", label: "Annual Volume (TEU)", type: "radio", options: ["< 100", "100-500", "500-1000", "> 1000"] },
        { id: "distance_ch", label: "Typical Distance (km)", type: "radio", options: ["< 100 km", "100-200 km", "200-350 km", "> 350 km"] }
    ]},
    { title: "Behavioral Factors", fields: [
        { id: "trust_overall", label: "Trust in rail vs truck", type: "likert" },
        { id: "sustainability_importance", label: "Importance of sustainability", type: "likert" }
    ]},
    { title: "Stated Preference Tasks", type: "sp", tasks: [
        { id: 1, road: { cost_chf_teu: 450, time_h: 12, ontime_p: 90, dep_per_day: 6 }, intermodal: { cost_chf_teu: 600, time_h: 18, ontime_p: 95, dep_per_day: 4 } },
        { id: 2, road: { cost_chf_teu: 300, time_h: 8, ontime_p: 85, dep_per_day: 4 }, intermodal: { cost_chf_teu: 450, time_h: 12, ontime_p: 95, dep_per_day: 6 } }
    ]}
  ];

  const handleSubmit = async () => {
    setIsSubmitting(true);
    try {
      // 1. Save to Firebase
      await addDoc(collection(db, "survey_responses"), {
        ...answers,
        submitted_at: new Date().toISOString()
      });
      
      // 2. Trigger Backend Recompute
      await fetch("http://localhost:8000/recompute-mnl", { method: "POST" });
      
      setIsSuccess(true);
    } catch (err) {
      alert("Submission failed. Check backend/firebase connection.");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (isSuccess) {
    return (
      <div className="max-w-md mx-auto text-center py-20">
        <div className="inline-flex p-4 bg-emerald-100 rounded-full mb-6 text-emerald-600"><CheckCircle size={48} /></div>
        <h2 className="text-3xl font-bold mb-4">Submission Success!</h2>
        <p className="text-slate-500 mb-8">The behavioral parameters have been updated.</p>
        <button onClick={() => navigate('/dashboard')} className="w-full bg-indigo-600 text-white py-4 rounded-xl font-bold hover:shadow-lg transition-all">Go to Dashboard</button>
      </div>
    );
  }

  const currentSection = sections[step];

  return (
    <div className="min-h-screen bg-slate-50 py-12 px-4">
      <div className="max-w-2xl mx-auto">
        
        {/* Progress Header */}
        <div className="flex justify-between items-end mb-8">
          <div>
            <h1 className="text-2xl font-black text-slate-900 uppercase tracking-tight">Swiss Intermodal Survey</h1>
            <p className="text-sm text-slate-500 font-medium">{currentSection.title}</p>
          </div>
          <span className="text-xs font-mono font-bold text-indigo-600 bg-indigo-50 px-2 py-1 rounded">PAGE {step + 1} / {sections.length}</span>
        </div>

        <div className="bg-white rounded-3xl shadow-xl shadow-slate-200/50 p-8 border border-white">
          {currentSection.type === "intro" && (
            <div className="space-y-4">
              <div className="p-4 bg-blue-50 rounded-2xl flex gap-3 text-blue-700">
                <Info className="shrink-0" />
                <p className="text-sm leading-relaxed">This study informs the <b>Discrete Choice Model</b> driving the optimization dashboard you just viewed.</p>
              </div>
              <p className="text-slate-600 leading-relaxed">Your participation will help refine how the model allocates freight between Road and Rail.</p>
              <button onClick={() => setStep(1)} className="w-full bg-indigo-600 text-white py-4 rounded-2xl font-bold mt-6 flex items-center justify-center gap-2">Get Started <ChevronRight size={20}/></button>
            </div>
          )}

          {currentSection.type === "sp" ? (
            <div>
              <div className="mb-6 p-4 bg-slate-50 rounded-2xl border-l-4 border-indigo-500 italic text-xs text-slate-600">
                Reference: Domestic move, 120–180km, 1× 20’ container. Compare options.
              </div>
              {currentSection.tasks.map((task) => (
                <ChoiceTable 
                  key={task.id} 
                  task={task} 
                  onSelect={(val) => setAnswers({...answers, [`task_${task.id}`]: val})}
                  selectedAlt={answers[`task_${task.id}`]}
                />
              ))}
            </div>
          ) : currentSection.fields?.map((field) => (
            field.type === "radio" ? (
              <div key={field.id} className="mb-8">
                <label className="block text-sm font-bold text-slate-900 mb-4">{field.label}</label>
                <div className="grid grid-cols-1 gap-2">
                  {field.options.map(opt => (
                    <button
                      key={opt}
                      onClick={() => setAnswers({...answers, [field.id]: opt})}
                      className={`text-left p-4 rounded-xl border-2 transition-all font-medium ${
                        answers[field.id] === opt ? "border-indigo-600 bg-indigo-50 text-indigo-700" : "border-slate-100 hover:border-slate-200 text-slate-600"
                      }`}
                    >
                      {opt}
                    </button>
                  ))}
                </div>
              </div>
            ) : field.type === "likert" && (
              <LikertRow 
                key={field.id} 
                label={field.label} 
                value={answers[field.id]} 
                onChange={(val) => setAnswers({...answers, [field.id]: val})} 
              />
            )
          ))}

          {/* Navigation */}
          {step > 0 && (
            <div className="flex gap-4 mt-12 pt-8 border-t border-slate-100">
              <button onClick={() => setStep(step - 1)} className="flex items-center gap-2 text-slate-400 font-bold hover:text-slate-600"><ChevronLeft size={20}/> BACK</button>
              <div className="flex-1" />
              {step === sections.length - 1 ? (
                <button onClick={handleSubmit} disabled={isSubmitting} className="bg-emerald-600 text-white px-10 py-4 rounded-2xl font-black flex items-center gap-2 hover:shadow-lg transition-all">
                  {isSubmitting ? "SYNCING..." : "FINISH & SYNC"} <Send size={18}/>
                </button>
              ) : (
                <button onClick={() => setStep(step + 1)} className="bg-indigo-600 text-white px-10 py-4 rounded-2xl font-black flex items-center gap-2 hover:shadow-lg transition-all">NEXT <ChevronRight size={20}/></button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}