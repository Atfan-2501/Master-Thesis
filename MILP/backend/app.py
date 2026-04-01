from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import pandas as pd
import subprocess
from openpyxl import load_workbook

app = FastAPI(title="Intermodal MILP API")

# Allow your Vite dev server to call FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Paths (adjust if your backend folder differs) ----
ROOT = Path(__file__).resolve().parents[1]  # .../MILP
DEFAULT_MASTER = ROOT  / "model_output" / "master_solution_decisions.xlsx"
DEFAULT_TS = ROOT  / "model_output" / "TS_Subproblem_Solution.xlsx"

# ---- Terminal name -> code mapping (matches your frontend TERMINALS keys) ----
NAME_TO_CODE = {
    "Aarau": "AARA",
    "Basel": "BASE",
    "Baselwolf": "BASE",   # your file uses Baselwolf
    "Bern": "BERN",
    "Chiasso": "CHIA",
    "Stabio": "STAB",
    "Visp": "VISP",
    "Zurich": "ZURI",
    "Lausanne": "LAUS",
    "Geneva": "GENE",
    "Luzern": "LUZE",
    "Olten": "OLTE",
    "Schaffhausen": "SCHA",
    "Winterthur": "WINT",
    "Fribourg": "FRIB",
}

class UtilityParams(BaseModel):
    cost_mean: float
    cost_std: float
    time_mean: float
    time_std: float
    freq_mean: float
    freq_std: float
    asc_road: float
    asc_other: float

def to_code(name: str) -> str:
    """Convert terminal name to code; fallback to original string if unknown."""
    if name is None:
        return ""
    return NAME_TO_CODE.get(str(name), str(name))

def compute_wtp(params: UtilityParams) -> dict:
    # Same convention as your UI: WTP = -beta_attr / beta_cost, scaled by 100 due to "per 100 CHF"
    if params.cost_mean == 0:
        return {"time": 0.0, "frequency": 0.0}
    return {
        "time": - (params.time_mean / params.cost_mean) * 100.0,
        "frequency": - (params.freq_mean / params.cost_mean) * 100.0,
    }

def parse_master(master_path: Path) -> tuple[list, dict]:
    df = pd.read_excel(master_path, sheet_name=0)

    # Keep “active” rows only: any positive flow or positive frequency
    df = df[(df["TEU_y"] > 0) | (df["frequency_f"] > 0)].copy()

    # Compute revenue from price * flow (if price is per TEU)
    df["revenue_chf"] = df["price_p"].fillna(0) * df["TEU_y"].fillna(0)

    routes = []
    for _, r in df.iterrows():
        origin_name = str(r["origin"])
        dest_name = str(r["destination"])
        routes.append({
            "origin": to_code(origin_name),
            "destination": to_code(dest_name),
            "origin_name": origin_name,
            "destination_name": dest_name,
            "mode": str(r["mode"]),
            "is_open": int(r.get("x_open", 0)),
            "flow": float(r.get("TEU_y", 0.0)),
            "frequency": int(r.get("frequency_f", 0)),
            "price": float(r.get("price_p", 0.0)),
            "revenue": float(r.get("revenue_chf", 0.0)),
            "cap_per_dep": float(r.get("cap_per_dep_TEU", 0.0)),
        })

    total_flow = sum(x["flow"] for x in routes)
    total_rev = sum(x["revenue"] for x in routes)

    # Average utilization for intermodal links (if meaningful): flow / (freq * cap_per_dep)
    utilizations = []
    for x in routes:
        if x["mode"].lower().startswith("inter") and x["frequency"] > 0 and x["cap_per_dep"] > 0:
            util = 100.0 * x["flow"] / (x["frequency"] * x["cap_per_dep"])
            utilizations.append(util)

    avg_util = sum(utilizations) / len(utilizations) if utilizations else 0.0

    summary = {
        "totalFlow": total_flow,
        "totalRevenue": total_rev,
        "totalCost": 0.0,   # Phase 1: fill later from your cost breakdown sheets
        "profit": total_rev,  # Phase 1 placeholder: profit = revenue - cost
        "numRoutes": len(routes),
        "avgUtilization": avg_util,
    }
    return routes, summary

def parse_ts(ts_path: Path) -> dict:
    xls = pd.ExcelFile(ts_path)
    ts = {}

    if "OD_Summary" in xls.sheet_names:
        od = pd.read_excel(ts_path, sheet_name="OD_Summary")
        ts["od_summary"] = [
            {
                "origin": to_code(r["origin_id"]),
                "destination": to_code(r["destination_id"]),
                "origin_name": str(r["origin_id"]),
                "destination_name": str(r["destination_id"]),
                "utilization": float(r.get("utilization_%", 0.0)),
                "flow_on_trains": float(r.get("flow_on_trains", 0.0)),
                "cap_group": float(r.get("cap_group", 0.0)),
                "group_id": str(r.get("group_id", "")),
            }
            for _, r in od.iterrows()
        ]

    if "Arc_Flows" in xls.sheet_names:
        arc = pd.read_excel(ts_path, sheet_name="Arc_Flows")
        # Keep it light: only train arcs for visualization/debug
        train_arcs = arc[arc["arc_type"].astype(str).str.lower().eq("train")].copy()
        ts["train_arcs"] = [
            {
                "arc_id": str(r.get("arc_id", "")),
                "from_node": str(r.get("from_node", "")),
                "to_node": str(r.get("to_node", "")),
                "flow_TEU": float(r.get("flow_TEU", 0.0)),
                "arc_cost_per_TEU": float(r.get("arc_cost_per_TEU", 0.0)),
                "group_id": str(r.get("group_id", "")),
            }
            for _, r in train_arcs.iterrows()
        ]

    return ts

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/run")
def run(params: UtilityParams):
    # 1. Update Input Excel with new parameters
    # Path to your master input file based on your project structure
    input_path = ROOT / "Input Data" / "master_problem_inputs_with_taste_draws.xlsx"
    
    if input_path.exists():
        try:
            wb = load_workbook(input_path)
            # Update 'Global_Params' or 'OD_Mode_Params' depending on your model structure
            if "Global_Params" in wb.sheetnames:
                ws = wb["Global_Params"]
                # Example: updating specific cells based on your Excel layout
                ws['B2'] = params.beta_cost
                ws['B3'] = params.beta_time
                ws['B4'] = params.beta_freq
                wb.save(input_path)
        except Exception as e:
            print(f"Warning: Could not update input Excel: {e}")

    # 2. Trigger the Benders Loop logic
    # This runs the actual optimization logic you provided in benders_loop.py
    try:
        # We call the script as a subprocess to ensure a clean environment for each run
        # This matches your 'benders_loop.py' logic for solving the MILP
        subprocess.run(["python", str(ROOT / "backend" / "benders_loop.py")], check=True)
    except subprocess.CalledProcessError as e:
        return {"error": f"Optimization loop failed: {str(e)}"}

    # 3. Read the solved outputs after the loop completes
    master_path = DEFAULT_MASTER
    ts_path = DEFAULT_TS

    if not master_path.exists():
        return {"error": f"Missing file: {master_path}"}
    if not ts_path.exists():
        return {"error": f"Missing file: {ts_path}"}

    # Use your existing parsing logic
    routes, summary = parse_master(master_path)
    ts = parse_ts(ts_path)
    
    # Calculate WTP using the fresh parameters
    wtp = compute_wtp(params)

    return {
        "routes": routes,
        "summary": summary,
        "wtp": wtp,
        "ts": ts,
    }
