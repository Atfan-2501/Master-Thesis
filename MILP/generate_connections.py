import pandas as pd
import math

# ============================================================
# CONFIG
# ============================================================

# Path to your master input file (with OD_Mode_Params)
MASTER_XLSX = "Input Data/master_problem_inputs_with_taste_draws.xlsx"  # or master_problem_inputs_template.xlsx

# Output connections file (Excel)
OUTPUT_CONNECTIONS_XLSX = "model_output/connections_generated.xlsx"

# Time discretization: hours per time step in the TS network
TIME_STEP_HOURS = 1.0  # e.g. 1 hour per step; change if you want coarser time steps


# ============================================================
# 1. LOAD MASTER DATA
# ============================================================

xls = pd.ExcelFile(MASTER_XLSX)
od_mode = pd.read_excel(MASTER_XLSX, sheet_name="OD_Mode_Params")

# Use only Intermodal rows for rail connections
im = od_mode[od_mode["mode_id"] == "Intermodal"].copy()

if im.empty:
    raise RuntimeError(
        "No Intermodal rows found in OD_Mode_Params. "
        "Check the mode_id values and that Intermodal is defined."
    )

# ============================================================
# 2. BUILD TERMINAL → NODE-ID MAPPING
# ============================================================

# Get all terminals that appear in Intermodal ODs
terminals = sorted(set(im["origin_id"].tolist()) | set(im["destination_id"].tolist()))

# 1-based integer IDs like Maxime's Node indices
term_to_node = {term: idx + 1 for idx, term in enumerate(terminals)}

print("Terminal → node mapping:")
for t, nid in term_to_node.items():
    print(f"  {t} -> {nid}")

# ============================================================
# 3. CONSTRUCT CONNECTIONS TABLE (STANDARD ONLY)
# ============================================================

rows = []

for _, row in im.iterrows():
    origin = row["origin_id"]
    dest   = row["destination_id"]

    node_i = term_to_node[origin]
    node_j = term_to_node[dest]

    # --- Transport time in time steps (ceil) ---
    # t_hours (continuous) → transport time in TS steps (integer)
    t_hours = float(row["t_hours"])
    t_steps = max(1, math.ceil(t_hours / TIME_STEP_HOURS))

    # --- Costs and capacity from OD_Mode_Params ---
    c_var = float(row["C_var_chf_per_TEU"])     # variable cost per TEU
    c_fix = float(row["C_fix_chf_per_dep"])     # train cost per departure
    cap   = float(row["CAP_TEU_per_dep"])       # TEU per departure

    rows.append({
        "Node (i)": node_i,
        "Node (j)": node_j,
        "origin_id": origin,
        "destination_id": dest,
        "transport time (Standard)": t_steps,
        "container cost (Standard)": c_var,
        "Train length (Standard)": cap,
        "Train cost(Standard)": c_fix,
    })

connections_df = pd.DataFrame(rows)

# Optional: sort for readability
connections_df = connections_df.sort_values(["Node (i)", "Node (j)"]).reset_index(drop=True)

print("\nGenerated connections table preview:")
print(connections_df.head())


# ============================================================
# 4. WRITE TO EXCEL
# ============================================================

connections_df.to_excel(OUTPUT_CONNECTIONS_XLSX, index=False)
print(f"\nWrote connections table to {OUTPUT_CONNECTIONS_XLSX}")