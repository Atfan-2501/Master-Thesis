import pandas as pd
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, NonNegativeReals, Binary, Reals,
    Constraint, ConstraintList, Objective, maximize, value, SolverFactory
)

# ============================================================
# CONFIG
# ============================================================

EXCEL_PATH = "Input Data/master_problem_inputs_with_taste_draws.xlsx"  # path to your Excel
SOLVER = "gurobi"  # or "cbc", "glpk", etc.


# ============================================================
# 1. LOAD DATA FROM EXCEL
# ============================================================

xls = pd.ExcelFile(EXCEL_PATH)
sheets = {name: pd.read_excel(EXCEL_PATH, sheet_name=name) for name in xls.sheet_names}

od_pairs     = sheets["OD_pairs"]
modes_df     = sheets["Modes"]
exposure_df  = sheets["Exposure_Dij"]
od_mode_df   = sheets["OD_Mode_Params"]
taste_draws  = sheets["Taste_Draws"]
globals_df   = sheets["Global_Params"]


# Represent OD as tuple (origin, dest)
od_pairs["od_id"] = list(zip(od_pairs["origin_id"], od_pairs["destination_id"]))

# --- modes: drop 'Other' ---
mode_ids = modes_df["mode_id"].tolist()
mode_ids = [m for m in mode_ids if m != "Other"]  # Only Road & Intermodal

# --- demand per OD (TEU per week) ---
exposure_df["od_id"] = list(zip(exposure_df["origin_id"], exposure_df["destination_id"]))
demand_map = exposure_df.set_index("od_id")["exposure_TEU_per_week"].to_dict()

# --- OD-mode attributes (t, p-bounds, f-bounds, capacity, costs) ---
od_mode_df["od_id"] = list(zip(od_mode_df["origin_id"], od_mode_df["destination_id"]))
# Filter only active ODs & relevant modes
od_mode_df = od_mode_df[od_mode_df["od_id"].isin(od_pairs["od_id"])]
od_mode_df = od_mode_df[od_mode_df["mode_id"].isin(mode_ids)]

# Build dictionaries keyed by (od,mode)
def odm_dict(col):
    return {(row["od_id"], row["mode_id"]): row[col]
            for _, row in od_mode_df.iterrows()}

t_hours     = odm_dict("t_hours")
p_min       = odm_dict("p_min_chf_per_TEU")
p_max       = odm_dict("p_max_chf_per_TEU")
f_min       = odm_dict("f_min_dep_per_week")
f_max       = odm_dict("f_max_dep_per_week")
cap_per_dep = odm_dict("CAP_TEU_per_dep")
c_fix       = odm_dict("C_fix_chf_per_dep")
c_var       = odm_dict("C_var_chf_per_TEU")

# --- Taste draws: we assume Taste_Draws is already populated ---
if taste_draws.empty:
    raise RuntimeError("Taste_Draws sheet is empty. Generate taste draws first.")

draw_ids = sorted(taste_draws["draw_id"].unique().tolist())
# ASC per (r, mode)
asc_map = {(int(r["draw_id"]), r["mode_id"]): float(r["ASC_r"])
           for _, r in taste_draws.iterrows()}

# betas: in your current draw generation, these are same across modes for each draw
beta_cost_map = {int(r["draw_id"]): float(r["beta_cost_r_per_100CHF"])
                 for _, r in taste_draws.drop_duplicates(subset=["draw_id"]).iterrows()}
beta_time_map = {int(r["draw_id"]): float(r["beta_time_r_per_hour"])
                 for _, r in taste_draws.drop_duplicates(subset=["draw_id"]).iterrows()}
beta_rel_map  = {int(r["draw_id"]): float(r["beta_rel_r_per_fraction"])
                 for _, r in taste_draws.drop_duplicates(subset=["draw_id"]).iterrows()}
beta_freq_map = {int(r["draw_id"]): float(r["beta_freq_r_per_departure"])
                 for _, r in taste_draws.drop_duplicates(subset=["draw_id"]).iterrows()}

# We are DROPPING reliability in the master: beta_rel is NOT used.

# --- Global params ---
globals_map = dict(zip(globals_df["parameter"], globals_df["value"]))
big_M_utility = float(globals_map.get("big_M_utility", 1000.0))


# ============================================================
# 2. BUILD PYOMO MODEL
# ============================================================

m = ConcreteModel()

# --- Sets ---
m.OD = Set(initialize=list(od_pairs["od_id"]), dimen=2)          # (origin, dest)
m.MODES = Set(initialize=mode_ids)                              # Road, Intermodal
m.R = Set(initialize=draw_ids)                                  # taste draw ids

# --- Parameters ---
m.Demand = Param(m.OD, initialize=lambda mod, o, d: demand_map.get((o, d), 0.0))

m.t_hours = Param(m.OD, m.MODES,
                  initialize=lambda mod, o, d, mo: t_hours.get(((o, d), mo), 0.0))

m.p_min = Param(m.OD, m.MODES,
                initialize=lambda mod, o, d, mo: p_min.get(((o, d), mo), 0.0)
                )

m.p_max = Param(m.OD, m.MODES,
                initialize=lambda mod, o, d, mo: p_max.get(((o, d), mo), 0.0)
                )

m.f_min = Param(m.OD, m.MODES,
                initialize=lambda mod, o, d, mo: f_min.get(((o, d), mo), 0.0)
                )

m.f_max = Param(m.OD, m.MODES,
                initialize=lambda mod, o, d, mo: f_max.get(((o, d), mo), 0.0)
                )

m.cap_per_dep = Param(m.OD, m.MODES,
                      initialize=lambda mod, o, d, mo: cap_per_dep.get(((o, d), mo), 0.0)
                      )

m.C_fix = Param(m.OD, m.MODES,
                initialize=lambda mod, o, d, mo: c_fix.get(((o, d), mo), 0.0)
                )

m.C_var = Param(m.OD, m.MODES,
                initialize=lambda mod, o, d, mo: c_var.get(((o, d), mo), 0.0)
                )

m.ASC = Param(m.R, m.MODES,
              initialize=lambda mod, r, mo: asc_map.get((int(r), mo), 0.0))

m.beta_cost = Param(m.R, initialize=lambda mod, r: beta_cost_map[int(r)])
m.beta_time = Param(m.R, initialize=lambda mod, r: beta_time_map[int(r)])
# beta_rel is not used
m.beta_freq = Param(m.R, initialize=lambda mod, r: beta_freq_map[int(r)])

m.bigM = Param(initialize=big_M_utility)

# --- Decision Variables ---

# service activation
m.x = Var(m.OD, m.MODES, domain=Binary)              # 1 if mode is "offered" for OD

# frequency and price
m.f = Var(m.OD, m.MODES, domain=NonNegativeReals)    # departures per week
m.p = Var(m.OD, m.MODES, domain=NonNegativeReals)    # price CHF/TEU

# flow per OD-mode (TEU/week)
m.y = Var(m.OD, m.MODES, domain=NonNegativeReals)

# SAA argmax: which mode is chosen by draw r for OD (fraction of demand)
m.w = Var(m.OD, m.R, m.MODES, domain=Binary)

# subproblem cost surrogate (for later Benders decomposition)
m.theta = Var(domain=NonNegativeReals)


# ============================================================
# 3. CONSTRAINTS
# ============================================================

# --- Price bounds gated by activation x ---
def price_lb_rule(mod, o, d, mo):
    return mod.p[o, d, mo] >= mod.p_min[o, d, mo] * mod.x[o, d, mo]
m.PriceLB = Constraint(m.OD, m.MODES, rule=price_lb_rule)

def price_ub_rule(mod, o, d, mo):
    return mod.p[o, d, mo] <= mod.p_max[o, d, mo] * mod.x[o, d, mo]
m.PriceUB = Constraint(m.OD, m.MODES, rule=price_ub_rule)

# --- Frequency bounds gated by activation x ---
def freq_lb_rule(mod, o, d, mo):
    return mod.f[o, d, mo] >= mod.f_min[o, d, mo] * mod.x[o, d, mo]
m.FreqLB = Constraint(m.OD, m.MODES, rule=freq_lb_rule)

def freq_ub_rule(mod, o, d, mo):
    return mod.f[o, d, mo] <= mod.f_max[o, d, mo] * mod.x[o, d, mo]
m.FreqUB = Constraint(m.OD, m.MODES, rule=freq_ub_rule)

# --- Capacity constraint: total TEU <= cap_per_dep * freq ---
def capacity_rule(mod, o, d, mo):
    return mod.y[o, d, mo] <= mod.cap_per_dep[o, d, mo] * mod.f[o, d, mo]
m.Capacity = Constraint(m.OD, m.MODES, rule=capacity_rule)

# --- SAA demand allocation: y = Demand * share ---
# share = (1/R) * sum_r w[od,r,m]
def demand_alloc_rule(mod, o, d, mo):
    R = len(mod.R)
    if R == 0:
        return mod.y[o, d, mo] == 0
    return mod.y[o, d, mo] == (mod.Demand[o, d] / R) * sum(mod.w[o, d, r, mo] for r in mod.R)
m.DemandAlloc = Constraint(m.OD, m.MODES, rule=demand_alloc_rule)

# --- Choice normalization: each draw chooses exactly one mode ---
def choice_norm_rule(mod, o, d, r):
    return sum(mod.w[o, d, r, mo] for mo in mod.MODES) == 1
m.ChoiceNormalization = Constraint(m.OD, m.R, rule=choice_norm_rule)

# --- Argmax linearization for utilities ---
def argmax_rule(mod, o, d, r, mo1, mo2):
    if mo1 == mo2:
        return Constraint.Skip

    # Utility for mode mo1 and mo2 for draw r and OD (o,d)
    U1 = (mod.ASC[r, mo1]
          + mod.beta_cost[r] * (mod.p[o, d, mo1] / 100.0)
          + mod.beta_time[r] * mod.t_hours[o, d, mo1]
          # reliability dropped
          + mod.beta_freq[r] * mod.f[o, d, mo1])

    U2 = (mod.ASC[r, mo2]
          + mod.beta_cost[r] * (mod.p[o, d, mo2] / 100.0)
          + mod.beta_time[r] * mod.t_hours[o, d, mo2]
          # reliability dropped
          + mod.beta_freq[r] * mod.f[o, d, mo2])

    # U1 >= U2 - M*(1 - w[o,d,r,mo1])
    return U1 >= U2 - mod.bigM * (1 - mod.w[o, d, r, mo1])

m.Argmax = Constraint(m.OD, m.R, m.MODES, m.MODES, rule=argmax_rule)


# ============================================================
# 4. BENDERS CUTS PLACEHOLDER FOR θ
# ============================================================

# For now, theta is just in the objective with no cuts.
# Later, when the TS subproblem is formulated, we will:
#
#  - solve the master with some (x,f,p,y),
#  - fix those decisions in the subproblem,
#  - solve the TS network problem to get a cost C_TS and duals,
#  - add a cut of the form:
#        theta >= alpha_k + sum_{od,mode} beta_k[od,mode] * y[od,mode]
#    or a simpler "no-good" or aggregated cut.
#
# Here we just prepare a ConstraintList that can be populated iteratively.
m.BendersCuts = ConstraintList()
# example (commented):
# m.BendersCuts.add(m.theta >= 0.0)  # trivial initial cut


# ============================================================
# 5. OBJECTIVE: MAXIMIZE PROFIT - θ
# ============================================================

def profit_expr(mod):
    revenue = sum(mod.p[o, d, mo] * mod.y[o, d, mo] for (o, d) in mod.OD for mo in mod.MODES)
    op_cost = sum(mod.C_fix[o, d, mo] * mod.f[o, d, mo]
                  + mod.C_var[o, d, mo] * mod.y[o, d, mo]
                  for (o, d) in mod.OD for mo in mod.MODES)
    # theta will later represent TS network cost from the subproblem
    return revenue - op_cost - mod.theta

m.OBJ = Objective(rule=profit_expr, sense=maximize)


# ============================================================
# 6. SOLVE
# ============================================================

if __name__ == "__main__":
    solver = SolverFactory(SOLVER)
    results = solver.solve(m, tee=True)

    print("\nObjective value:", value(m.OBJ))

    # Example: print chosen intermodal flows and prices
    print("\nIntermodal services:")
    for (o, d) in m.OD:
        for mo in m.MODES:
            if mo == "Intermodal" and value(m.x[o, d, mo]) > 0.5:
                print(f"{o}->{d}: f={value(m.f[o,d,mo]):.2f}, "
                      f"p={value(m.p[o,d,mo]):.2f}, "
                      f"y={value(m.y[o,d,mo]):.2f}")


# ============================================================
# 7. EXPORT DECISION VARIABLES TO EXCEL
# ============================================================

import pandas as pd

output_data = []

for (o, d) in m.OD:
    for mo in m.MODES:
        output_data.append({
            "origin": o,
            "destination": d,
            "mode": mo,
            "x_open": value(m.x[o, d, mo]),
            "frequency_f": value(m.f[o, d, mo]),
            "price_p": value(m.p[o, d, mo]),
            "TEU_y": value(m.y[o, d, mo])
        })

df_decisions = pd.DataFrame(output_data)

# Write the file
df_decisions.to_excel("model_outputservice_plan.csv", index=False)
print("\nMaster decision variables written to master_solution_decisions.xlsx")