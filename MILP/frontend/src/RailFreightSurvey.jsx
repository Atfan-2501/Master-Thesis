import React, { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronLeft, ChevronRight, Send, Info, CheckCircle } from 'lucide-react';
import { db } from './firebase_config';
import { collection, addDoc } from 'firebase/firestore';

// Import the JSON designs (Ensure these paths exist)
import coreDesign from '../../../Discrete Choice Model/model_inputs/sp_core_design_blocks.json';
import checksDesign from '../../../Discrete Choice Model/model_inputs/sp_checks_design.json';

export default function RailFreightSurvey() {
  const navigate = useNavigate();
  
  // Replicating Streamlit Session State
  const [pageIdx, setPageIdx] = useState(0);
  const [answers, setAnswers] = useState({});
  const [respondentId] = useState(crypto.randomUUID());
  const [assignedBlock] = useState(Math.floor(Math.random() * 3) + 1);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Styles replicated from Rail_Freight_Survey.py inject_css
  const cardStyle = "bg-white/90 backdrop-blur-md rounded-[12px] p-4 shadow-sm ring-1 ring-slate-200 mb-[10px]";
  const labelStyle = "bg-white/90 p-2 rounded-[8px] font-extrabold text-slate-900 mb-2 inline-block";

  // Replicating Task Builder Logic
  const tasks = useMemo(() => {
    const blockTasks = coreDesign.filter(d => d.block_id === assignedBlock);
    const sortedTasks = [...new Set(blockTasks.map(t => t.task_in_block))].sort((a, b) => a - b);
    
    return sortedTasks.map(tNum => {
      const road = blockTasks.find(d => d.task_in_block === tNum && d.alt_id === 'Road');
      const intermodal = blockTasks.find(d => d.task_in_block === tNum && d.alt_id === 'Intermodal');
      return { tNum, road, intermodal, isCheck: false };
    });
  }, [assignedBlock]);

  const handleSubmit = async () => {
    setIsSubmitting(true);
    const timestamp = new Date().toISOString();
    
    try {
      // 1. Save Flat Row (Section 1-6)
      await addDoc(collection(db, "ch_intermodal_survey_rows"), {
        ...answers,
        respondent_id: respondentId,
        assigned_block: assignedBlock,
        submitted_at_utc: timestamp
      });

      // 2. Trigger Backend Recompute
      await fetch("http://localhost:8000/recompute-mnl", { method: "POST" });
      
      setPageIdx(-1); // Success page
    } catch (error) {
      console.error("Submission error:", error);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Success View
  if (pageIdx === -1) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[url('/project-mobility.jpg')] bg-cover p-4">
        <div className="bg-white/90 p-10 rounded-xl text-center max-w-lg">
          <CheckCircle className="mx-auto text-emerald-500 mb-4" size={48} />
          <h2 className="text-2xl font-bold text-emerald-600">Thank You!</h2>
          <p className="text-slate-700 mt-2">Your response has been recorded and the model is updating.</p>
          <button onClick={() => navigate('/dashboard')} className="mt-6 bg-indigo-600 text-white px-6 py-2 rounded-lg font-bold">View Dashboard</button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-[url('/project-mobility.jpg')] bg-cover bg-fixed py-10 px-4">
      <div className="max-w-2xl mx-auto">
        
        {/* Header */}
        <div className="text-center mb-8">
          <img src="https://logowik.com/content/uploads/images/eth-zurich1144.jpg" width="100" className="mx-auto mb-2" alt="ETH Logo" />
          <h1 className="text-3xl font-bold text-white shadow-sm">Survey: Swiss Intermodal Freight</h1>
        </div>

        {/* Section rendering logic (Simplified for space, apply to all items in survey_schema) */}
        <div className={cardStyle}>
          <h3 className="text-lg font-bold mb-4 border-b pb-2">Section {pageIdx + 1}</h3>
          
          {pageIdx === 6 ? (
            // Stated Preference Task UI
            tasks.map(task => (
              <div key={task.tNum} className="mb-8 border-b pb-6 last:border-0">
                <div className={labelStyle}>Task {task.tNum}: Select preferred option</div>
                <table className="w-full text-sm bg-white rounded-lg overflow-hidden border border-slate-200 mb-4">
                  <thead>
                    <tr className="bg-slate-50">
                      <th className="p-2 border">Attribute</th>
                      <th className="p-2 border">Road</th>
                      <th className="p-2 border">Intermodal</th>
                    </tr>
                  </thead>
                  <tbody>
                    {["cost_chf_teu", "time_h", "ontime_p", "dep_per_day"].map(attr => (
                      <tr key={attr}>
                        <td className="p-2 border font-medium capitalize">{attr.replace(/_/g, ' ')}</td>
                        <td className="p-2 border text-center">{task.road[attr]}</td>
                        <td className="p-2 border text-center">{task.intermodal[attr]}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="flex gap-2">
                  {["Road", "Intermodal", "Other"].map(alt => (
                    <button
                      key={alt}
                      onClick={() => setAnswers({...answers, [`sp_choice_${task.tNum}`]: alt})}
                      className={`flex-1 p-2 rounded-lg border-2 font-bold transition-all ${
                        answers[`sp_choice_${task.tNum}`] === alt 
                        ? "bg-indigo-600 text-white border-indigo-600" 
                        : "bg-white border-slate-200 text-slate-600 hover:border-indigo-300"
                      }`}
                    >
                      {alt}
                    </button>
                  ))}
                </div>
              </div>
            ))
          ) : (
            <div className="text-center italic text-slate-500">
              (Static profile and behavioral questions go here)
            </div>
          )}

          {/* Navigation Buttons */}
          <div className="flex justify-between mt-8 pt-4 border-t border-slate-200">
            <button 
              disabled={pageIdx === 0}
              onClick={() => setPageIdx(pageIdx - 1)}
              className="flex items-center gap-1 text-slate-600 font-bold disabled:opacity-30"
            >
              <ChevronLeft size={20} /> Previous
            </button>
            {pageIdx === 6 ? (
              <button 
                onClick={handleSubmit}
                className="bg-emerald-600 text-white px-8 py-2 rounded-lg font-bold flex items-center gap-2"
              >
                {isSubmitting ? "Submitting..." : "Submit"} <Send size={16} />
              </button>
            ) : (
              <button 
                onClick={() => setPageIdx(pageIdx + 1)}
                className="bg-indigo-600 text-white px-8 py-2 rounded-lg font-bold flex items-center gap-2"
              >
                Next <ChevronRight size={20} />
              </button>
            )}
          </div>
        </div>
        
        {/* Footer */}
        <div className="bg-white p-4 rounded-xl text-center text-xs font-medium">
          © 2025 ETH Zurich. All rights reserved.
        </div>
      </div>
    </div>
  );
}