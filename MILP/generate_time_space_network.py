import pandas as pd
import math

# ============================================================
# CONFIG
# ============================================================

CONNECTIONS_XLSX = "model_output/connections_generated.xlsx"
TIMETABLE_XLSX = "Input Data/HUPAC Timetable.xlsx"
OUTPUT_TS_XLSX = "model_output/TS_Subproblem_Inputs_extended.xlsx"

NUM_TIME_STEPS = 168
TIME_STEP_HOURS = 1.0

STORAGE_COST_DAY_CHF = 32.0
DWELL_COST_PER_TEU = STORAGE_COST_DAY_CHF * (TIME_STEP_HOURS / 24.0)


TERMINAL_STORAGE_AREA = 600.0
AREA_M2_PER_TEU = 25.0
DWELL_CAP_TEU = int(TERMINAL_STORAGE_AREA / AREA_M2_PER_TEU)

BIG_M_FLOW = 1e6

# ============================================================
# 1. LOAD DATA
# ============================================================

connections = pd.read_excel(CONNECTIONS_XLSX)
timetable = pd.read_excel(TIMETABLE_XLSX, sheet_name="Worksheet")

col_node_i = "Node (i)"
col_node_j = "Node (j)"
col_origin = "origin_id"
col_dest   = "destination_id"
col_ttime  = "transport time (Standard)"
col_cvar   = "container cost (Standard)"
col_cap    = "Train length (Standard)"
col_cfix   = "Train cost(Standard)"

# ============================================================
# 2. BUILD MAPPINGS
# ============================================================

node_to_terminal = {}
for _, row in connections.iterrows():
    ni = int(row[col_node_i])
    nj = int(row[col_node_j])
    oi = row[col_origin]
    dj = row[col_dest]
    node_to_terminal.setdefault(ni, oi)
    node_to_terminal.setdefault(nj, dj)

node_ids = sorted(node_to_terminal.keys())
terminal_to_node = {name: idx for idx, name in node_to_terminal.items()}

conn_ods = connections[[col_origin, col_dest]].drop_duplicates()
conn_ods = conn_ods.rename(columns={col_origin: "origin_id", col_dest: "destination_id"})
conn_ods["od_id"] = conn_ods["origin_id"] + "-" + conn_ods["destination_id"]

# ============================================================
# 3. TS_NODES
# ============================================================

ts_nodes_rows = []

for n in node_ids:
    term_name = node_to_terminal[n]
    for t in range(NUM_TIME_STEPS):
        ts_nodes_rows.append({
            "node_id": f"Node{int(n)}_t{t}",
            "terminal_id": term_name,
            "node_index": int(n),
            "time_step": t,
            "node_type": "terminal_time",
        })

for _, row in conn_ods.iterrows():
    origin = row["origin_id"]
    dest   = row["destination_id"]
    source_id = f"OD_{origin}_{dest}_source"
    sink_id   = f"OD_{origin}_{dest}_sink"
    ts_nodes_rows.append({
        "node_id": source_id,
        "terminal_id": origin,
        "node_index": terminal_to_node[origin],
        "time_step": -1,
        "node_type": "od_source",
    })
    ts_nodes_rows.append({
        "node_id": sink_id,
        "terminal_id": dest,
        "node_index": terminal_to_node[dest],
        "time_step": -1,
        "node_type": "od_sink",
    })

ts_nodes = pd.DataFrame(ts_nodes_rows)

# ============================================================
# 4. TS_ARCS - TRAIN ARCS (FIXED)
# ============================================================

ts_arcs_rows = []
day_cols = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
day_index = {d: i for i, d in enumerate(day_cols)}

for _, conn in connections.iterrows():
    origin = conn[col_origin]
    dest   = conn[col_dest]
    node_i = int(conn[col_node_i])
    node_j = int(conn[col_node_j])

    travel_steps = int(conn[col_ttime])
    c_var = float(conn[col_cvar])
    cap   = float(conn[col_cap])
    c_fix = float(conn[col_cfix])

    # ✅ FIX: Group ID based on ACTUAL node direction
    group_id = f"{origin}-{dest}_IM"

    # Match timetable: departure FROM origin TO dest
    mask = ((timetable["Departure term desc."] == origin) &
            (timetable["Arrival terminal desc."] == dest))
    tt_rows = timetable[mask]

    if tt_rows.empty:
        # Fallback: one train per day at noon
        for d_idx in range(7):
            h = 12
            m = 0
            time_hours = d_idx * 24 + h + m / 60.0
            t_dep = int(round(time_hours / TIME_STEP_HOURS))
            t_arr = t_dep + travel_steps
            if t_arr >= NUM_TIME_STEPS:
                continue

            from_node_id = f"Node{node_i}_t{t_dep}"
            to_node_id   = f"Node{node_j}_t{t_arr}"
            arc_id = f"train_{node_i}_{node_j}_dFallback{d_idx}_t{t_dep}"

            ts_arcs_rows.append({
                "arc_id": arc_id,
                "from_node": from_node_id,
                "to_node": to_node_id,
                "arc_type": "train",
                "base_cap_TEU": cap,
                "cost_chf_per_TEU": c_var,
                "group_id": group_id,
                "fixed_cost_chf_per_arc": c_fix,
            })
    else:
        # Real timetable
        for _, tt in tt_rows.iterrows():
            closing_str = tt["Closing time"]
            if pd.isna(closing_str):
                continue

            h_str, m_str = str(closing_str).split(":")
            h = int(h_str)
            m = int(m_str)

            for day_col in day_cols:
                if tt[day_col] == "X":
                    d_idx = day_index[day_col]
                    time_hours = d_idx * 24 + h + m / 60.0
                    t_dep = int(round(time_hours / TIME_STEP_HOURS))
                    t_arr = t_dep + travel_steps
                    if t_arr >= NUM_TIME_STEPS:
                        continue

                    from_node_id = f"Node{node_i}_t{t_dep}"
                    to_node_id   = f"Node{node_j}_t{t_arr}"
                    arc_id = f"train_{node_i}_{node_j}_d{day_col}_t{t_dep}"

                    ts_arcs_rows.append({
                        "arc_id": arc_id,
                        "from_node": from_node_id,
                        "to_node": to_node_id,
                        "arc_type": "train",
                        "base_cap_TEU": cap,
                        "cost_chf_per_TEU": c_var,
                        "group_id": group_id,
                        "fixed_cost_chf_per_arc": c_fix,
                    })

# ============================================================
# 5. DWELL ARCS
# ============================================================

for n in node_ids:
    for t in range(NUM_TIME_STEPS - 1):
        from_node_id = f"Node{int(n)}_t{t}"
        to_node_id   = f"Node{int(n)}_t{t+1}"
        arc_id = f"dwell_{int(n)}_t{t}"

        ts_arcs_rows.append({
            "arc_id": arc_id,
            "from_node": from_node_id,
            "to_node": to_node_id,
            "arc_type": "dwell",
            "base_cap_TEU": DWELL_CAP_TEU,
            "cost_chf_per_TEU": DWELL_COST_PER_TEU,
            "group_id": "",
            "fixed_cost_chf_per_arc": 0.0,
        })

ts_arcs = pd.DataFrame(ts_arcs_rows)

# ============================================================
# 6. TIME WINDOWS PER OD
# ============================================================

timeinfo = {}

for _, r in ts_arcs[ts_arcs["arc_type"] == "train"].iterrows():
    g = r["group_id"]
    if not g:
        continue
    from_t = int(r["from_node"].split("_t")[1])
    to_t   = int(r["to_node"].split("_t")[1])

    if g not in timeinfo:
        timeinfo[g] = {
            "earliest_dep": from_t,
            "earliest_arr": to_t,
            "latest_arr": to_t,
        }
    else:
        timeinfo[g]["earliest_dep"] = min(timeinfo[g]["earliest_dep"], from_t)
        timeinfo[g]["earliest_arr"] = min(timeinfo[g]["earliest_arr"], to_t)
        timeinfo[g]["latest_arr"]   = max(timeinfo[g]["latest_arr"], to_t)

# ============================================================
# 7. ENTRY/EXIT ARCS
# ============================================================

od_entry_rows = []
od_exit_rows = []
od_timewin_rows = []

for _, row in conn_ods.iterrows():
    origin = row["origin_id"]
    dest   = row["destination_id"]
    od_id  = row["od_id"]
    group_id = f"{origin}-{dest}_IM"

    info = timeinfo.get(group_id, None)
    if info is None:
        earliest_dep = 0
        earliest_arr = 0
        latest_arr   = NUM_TIME_STEPS - 1
    else:
        earliest_dep = info["earliest_dep"]
        earliest_arr = info["earliest_arr"]
        latest_arr   = info["latest_arr"]

    source_node_id = f"OD_{origin}_{dest}_source"
    sink_node_id   = f"OD_{origin}_{dest}_sink"

    origin_idx = terminal_to_node[origin]
    dest_idx   = terminal_to_node[dest]

    # Entry arcs
    for t in range(0, earliest_dep + 1):
        entry_arc_id = f"entry_{origin}_{dest}_t{t}"
        ts_arcs_rows.append({
            "arc_id": entry_arc_id,
            "from_node": source_node_id,
            "to_node": f"Node{origin_idx}_t{t}",
            "arc_type": "entry",
            "base_cap_TEU": BIG_M_FLOW,
            "cost_chf_per_TEU": 0.0,
            "group_id": group_id,
            "fixed_cost_chf_per_arc": 0.0,
        })
        od_entry_rows.append({
            "origin_id": origin,
            "destination_id": dest,
            "od_id": od_id,
            "entry_arc_id": entry_arc_id,
        })

    # Exit arcs
    for t in range(earliest_arr, NUM_TIME_STEPS):
        exit_arc_id = f"exit_{origin}_{dest}_t{t}"
        ts_arcs_rows.append({
            "arc_id": exit_arc_id,
            "from_node": f"Node{dest_idx}_t{t}",
            "to_node": sink_node_id,
            "arc_type": "exit",
            "base_cap_TEU": BIG_M_FLOW,
            "cost_chf_per_TEU": 0.0,
            "group_id": group_id,
            "fixed_cost_chf_per_arc": 0.0,
        })
        od_exit_rows.append({
            "origin_id": origin,
            "destination_id": dest,
            "od_id": od_id,
            "exit_arc_id": exit_arc_id,
        })

    od_timewin_rows.append({
        "origin_id": origin,
        "destination_id": dest,
        "od_id": od_id,
        "earliest_dep_t": earliest_dep,
        "earliest_arr_t": earliest_arr,
        "latest_arr_t": latest_arr,
    })

ts_arcs = pd.DataFrame(ts_arcs_rows)
od_entry_arcs = pd.DataFrame(od_entry_rows)
od_exit_arcs  = pd.DataFrame(od_exit_rows)
ts_od_timewindows = pd.DataFrame(od_timewin_rows)

# ============================================================
# 8. TRAIN GROUPS
# ============================================================

cap_per_od = (connections
              .groupby([col_origin, col_dest])[col_cap]
              .max()
              .reset_index())

cap_per_od["od_id"] = cap_per_od[col_origin] + "-" + cap_per_od[col_dest]
cap_per_od["group_id"] = cap_per_od["od_id"] + "_IM"
cap_per_od.rename(columns={col_cap: "cap_per_dep_TEU"}, inplace=True)

train_groups = cap_per_od[["origin_id", "destination_id",
                           "od_id", "group_id", "cap_per_dep_TEU"]]

ts_globals = pd.DataFrame({
    "parameter": [
        "big_M_flow",
        "time_step_minutes",
        "num_time_steps",
        "dwell_cap_TEU",
        "dwell_cost_per_step",
    ],
    "value": [
        BIG_M_FLOW,
        TIME_STEP_HOURS * 60.0,
        NUM_TIME_STEPS,
        DWELL_CAP_TEU,
        DWELL_COST_PER_TEU,
    ],
})

# ============================================================
# 9. DIAGNOSTIC OUTPUT
# ============================================================

print("\n=== TRAIN ARC GROUP ASSIGNMENT VERIFICATION ===")
train_arc_groups = ts_arcs[ts_arcs["arc_type"] == "train"].groupby("group_id").agg({
    "arc_id": "count",
    "from_node": lambda x: set(n.split("_")[0] for n in x),
    "to_node": lambda x: set(n.split("_")[0] for n in x)
}).reset_index()

train_arc_groups.columns = ["group_id", "num_arcs", "from_nodes", "to_nodes"]
print(train_arc_groups.to_string())

# ============================================================
# 10. WRITE OUTPUT
# ============================================================

with pd.ExcelWriter(OUTPUT_TS_XLSX, engine="xlsxwriter") as writer:
    ts_nodes.to_excel(writer, sheet_name="TS_Nodes", index=False)
    ts_arcs.to_excel(writer, sheet_name="TS_Arcs", index=False)
    od_entry_arcs.to_excel(writer, sheet_name="OD_Entry_Arcs", index=False)
    od_exit_arcs.to_excel(writer, sheet_name="OD_Exit_Arcs", index=False)
    train_groups.to_excel(writer, sheet_name="Train_Groups", index=False)
    ts_od_timewindows.to_excel(writer, sheet_name="TS_OD_TimeWindows", index=False)
    ts_globals.to_excel(writer, sheet_name="TS_Global_Params", index=False)

print(f"\nTime-space network written to {OUTPUT_TS_XLSX}")
print(f"TS_Nodes: {len(ts_nodes)} rows")
print(f"TS_Arcs:  {len(ts_arcs)} rows")