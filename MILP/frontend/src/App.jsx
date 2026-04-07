import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import IntermodalMILPDashboard from "./IntermodalMILPDashboard";
import RailFreightSurvey from "./RailFreightSurvey"; 

function App() {
  return (
    <Router>
      <Routes>
        {/* Main Dashboard Route */}
        <Route path="/dashboard" element={<IntermodalMILPDashboard />} />
        
        {/* Updated Survey Route: Replaced placeholder with the actual component */}
        <Route path="/survey" element={<RailFreightSurvey />} />

        {/* Redirect root to survey first (standard for data collection) 
            or keep as /dashboard if you want to show results first */}
        <Route path="/" element={<Navigate to="/survey" replace />} />
        
        {/* Fallback for 404s */}
        <Route path="*" element={
          <div className="min-h-screen flex items-center justify-center bg-slate-50">
            <div className="text-center">
              <h2 className="text-2xl font-bold text-slate-800">404</h2>
              <p className="text-slate-500">Page Not Found</p>
              <button 
                onClick={() => window.location.href = '/dashboard'}
                className="mt-4 text-indigo-600 hover:underline font-medium"
              >
                Return to Dashboard
              </button>
            </div>
          </div>
        } />
      </Routes>
    </Router>
  );
}

export default App;