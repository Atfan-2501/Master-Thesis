import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals,
    Constraint, Objective, minimize, value, SolverFactory
)
from pyomo.opt import TerminationCondition

TS_XLSX = "model_output/TS_Subproblem_Inputs_extended.xlsx"
MASTER_SOL_XLSX = "model_output/master_solution_decisions.xlsx"
SOLVER = "gurobi"

# --- Load TS data ---
ts_nodes = pd.read_excel(TS_XLSX, sheet_name="TS_Nodes")
ts_arcs = pd.read_excel(TS_XLSX, sheet_name="TS_Arcs")
train_groups = pd.read_excel(TS_XLSX, sheet_name="Train_Groups")

node_ids = ts_nodes["node_id"].tolist()
arc_ids = ts_arcs["arc_id"].tolist()

node_type = ts_nodes.set_index("node_id")["node_type"].to_dict()
terminal_of_node = ts_nodes.set_index("node_id")["terminal_id"].to_dict()

from_node = ts_arcs.set_index("arc_id")["from_node"].to_dict()
to_node   = ts_arcs.set_index("arc_id")["to_node"].to_dict()
arc_type  = ts_arcs.set_index("arc_id")["arc_type"].to_dict()
cost_of_arc = ts_arcs.set_index("arc_id")["cost_chf_per_TEU"].to_dict()
base_cap_of_arc = ts_arcs.set_index("arc_id")["base_cap_TEU"].to_dict()
group_of_arc = ts_arcs.set_index("arc_id")["group_id"].fillna("").to_dict()

# --- Load master solution ---
master_sol = pd.read_excel(MASTER_SOL_XLSX)
master_im = master_sol[master_sol["mode"] == "Intermodal"].copy()
master_im["od_id"] = master_im["origin"] + "-" + master_im["destination"]

train_groups["od_id"] = train_groups["origin_id"] + "-" + train_groups["destination_id"]
master_im = master_im.merge(
    train_groups[["origin_id", "destination_id", "od_id", "group_id", "cap_per_dep_TEU"]],
    left_on=["origin", "destination", "od_id"],
    right_on=["origin_id", "destination_id", "od_id"],
    how="inner",
)

od_set = sorted(list(set(zip(master_im["origin"], master_im["destination"]))))

y_req = {}
f_lev = {}
cap_per_dep = {}
group_of_od = {}
source_node_of_od = {}
sink_node_of_od = {}

for _, row in master_im.iterrows():
    o = row["origin"]
    d = row["destination"]
    od = (o, d)
    od_id = row["od_id"]
    group_id = row["group_id"]

    y_req[od] = float(row["TEU_y"])
    f_lev[od] = float(row["frequency_f"])
    cap_per_dep[od] = float(row["cap_per_dep_TEU"])
    group_of_od[od] = group_id

    source_node_of_od[od] = f"OD_{o}_{d}_source"
    sink_node_of_od[od]   = f"OD_{o}_{d}_sink"

# Diagnostic
print("\n[Diagnostic] Checking OD capacities and arc existence...")
for od in od_set:
    o, d = od
    g = group_of_od.get(od, None)
    y = y_req.get(od, 0.0)
    f = f_lev.get(od, 0.0)
    cap_dep = cap_per_dep.get(od, 0.0)
    cap_group = cap_dep * f
    arcs_for_group = [a for a in arc_ids if group_of_arc.get(a, "") == g]
    print(f"  OD {o}->{d}: y={y}, cap_dep={cap_dep}, f={f}, "
          f"cap_group={cap_group}, #train_arcs={len(arcs_for_group)}")
    if not arcs_for_group and y > 1e-6:
        print("    WARNING: No train arcs but y_req > 0 (no feasible route).")
    if cap_group + 1e-6 < y:
        print("    WARNING: y_req exceeds cap_dep * f (infeasibility likely).")

# --- Build model ---
m = ConcreteModel()
m.N = Set(initialize=node_ids)
m.A = Set(initialize=arc_ids)
m.OD = Set(initialize=od_set, dimen=2)

m.cost_of_arc = Param(m.A, initialize=lambda mod, a: float(cost_of_arc[a]))
m.base_cap_of_arc = Param(m.A, initialize=lambda mod, a: float(base_cap_of_arc[a]))
m.y_req = Param(m.OD, initialize=lambda mod, o, d: float(y_req.get((o, d), 0.0)))
m.f_lev = Param(m.OD, initialize=lambda mod, o, d: float(f_lev.get((o, d), 0.0)))
m.cap_per_dep = Param(m.OD, initialize=lambda mod, o, d: float(cap_per_dep.get((o, d), 0.0)))

m.group_of_od = group_of_od
m.group_of_arc = group_of_arc
m.from_node = from_node
m.to_node = to_node
m.node_type = node_type
m.source_node_of_od = source_node_of_od
m.sink_node_of_od = sink_node_of_od

m.flow = Var(m.OD, m.A, domain=NonNegativeReals)

def node_balance_rule(mod, o, d, n):
    od = (o, d)
    inflow = sum(mod.flow[o, d, a] for a in mod.A if mod.to_node[a] == n)
    outflow = sum(mod.flow[o, d, a] for a in mod.A if mod.from_node[a] == n)
    source_n = mod.source_node_of_od.get(od, None)
    sink_n   = mod.sink_node_of_od.get(od, None)
    if n == source_n:
        return outflow - inflow == mod.y_req[o, d]
    elif n == sink_n:
        return inflow - outflow == mod.y_req[o, d]
    else:
        return inflow - outflow == 0.0

m.NodeBalance = Constraint(m.OD, m.N, rule=node_balance_rule)

def arc_capacity_rule(mod, a):
    return sum(mod.flow[o, d, a] for (o, d) in mod.OD) <= mod.base_cap_of_arc[a]

m.ArcCapacity = Constraint(m.A, rule=arc_capacity_rule)

def group_capacity_rule(mod, o, d):
    od = (o, d)
    g_od = mod.group_of_od.get(od, None)
    if g_od is None:
        return sum(mod.flow[o, d, a] for a in mod.A) <= 0.0
    return sum(
        mod.flow[o, d, a]
        for a in mod.A
        if mod.group_of_arc.get(a, "") == g_od
    ) <= mod.cap_per_dep[o, d] * mod.f_lev[o, d]

m.GroupCapacity = Constraint(m.OD, rule=group_capacity_rule)

def ts_cost_expr(mod):
    return sum(
        mod.cost_of_arc[a] * mod.flow[o, d, a]
        for (o, d) in mod.OD
        for a in mod.A
    )

m.TS_Obj = Objective(rule=ts_cost_expr, sense=minimize)

if __name__ == "__main__":
    solver = SolverFactory(SOLVER)
    results = solver.solve(m, tee=True)
    term_cond = results.solver.termination_condition
    print(f"\nTS solver termination condition: {term_cond}")

    if term_cond != TerminationCondition.optimal:
        print("\nTS subproblem is not optimal (likely infeasible).")
        raise SystemExit(0)

    print("\nTS subproblem optimal value (CHF):", value(m.TS_Obj))

    print("\nOD-level flows and capacity usage:")
    for (o, d) in m.OD:
        g_od = m.group_of_od.get((o, d), None)
        total_flow = 0.0
        if g_od is not None:
            for a in m.A:
                if m.group_of_arc.get(a, "") == g_od:
                    fv = m.flow[o, d, a].value
                    if fv is not None:
                        total_flow += float(fv)
        cap_val = float(value(m.cap_per_dep[o, d] * m.f_lev[o, d]))
        print(f"{o} -> {d}: y_req={float(m.y_req[o,d]):.2f}, "
              f"flow_on_TS={total_flow:.2f}, cap_group={cap_val:.2f}")

    # ---- Save solution ----
    OUTPUT_SUBPROBLEM_XLSX = "model_output/TS_Subproblem_Solution.xlsx"

    rows = []
    for (o, d) in m.OD:
        for a in m.A:
            fv = m.flow[o, d, a].value
            if fv is None or fv <= 1e-6:
                continue
            rows.append({
                "origin_id": o,
                "destination_id": d,
                "arc_id": a,
                "from_node": m.from_node[a],
                "to_node": m.to_node[a],
                "arc_type": arc_type[a],
                "flow_TEU": float(fv),
                "arc_cost_per_TEU": float(m.cost_of_arc[a]),
                "group_id": m.group_of_arc[a],
            })
    df_flows = pd.DataFrame(rows)

    summary_rows = []
    for (o, d) in m.OD:
        g_od = m.group_of_od.get((o, d), None)
        total_flow = 0.0
        if g_od is not None:
            for a in m.A:
                if m.group_of_arc.get(a, "") == g_od:
                    fv = m.flow[o, d, a].value
                    if fv is not None:
                        total_flow += float(fv)
        cap_val = float(value(m.cap_per_dep[o, d] * m.f_lev[o, d]))
        summary_rows.append({
            "origin_id": o,
            "destination_id": d,
            "y_req": float(m.y_req[o, d]),
            "flow_on_TS": total_flow,
            "cap_group": cap_val,
            "group_id": g_od,
        })
    df_summary = pd.DataFrame(summary_rows)

    with pd.ExcelWriter(OUTPUT_SUBPROBLEM_XLSX, engine="xlsxwriter") as writer:
        df_flows.to_excel(writer, sheet_name="Arc_Flows", index=False)
        df_summary.to_excel(writer, sheet_name="OD_Summary", index=False)

    print(f"\nSaved TS subproblem solution to {OUTPUT_SUBPROBLEM_XLSX}")
