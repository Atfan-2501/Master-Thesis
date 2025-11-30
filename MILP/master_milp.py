import pandas as pd
from pyomo.environ import *

def build_master_problem():
    """Build master problem for Benders decomposition"""
    
    # Load data (same as before)
    EXCEL_PATH = "Input Data/master_problem_inputs_with_taste_draws.xlsx"
    xls = pd.ExcelFile(EXCEL_PATH)
    sheets = {name: pd.read_excel(EXCEL_PATH, sheet_name=name) for name in xls.sheet_names}

    od_pairs     = sheets["OD_pairs"]
    modes_df     = sheets["Modes"]
    exposure_df  = sheets["Exposure_Dij"]
    od_mode_df   = sheets["OD_Mode_Params"]
    taste_draws  = sheets["Taste_Draws"]
    globals_df   = sheets["Global_Params"]

    # Data processing (same as before)
    od_pairs["od_id"] = list(zip(od_pairs["origin_id"], od_pairs["destination_id"]))

    mode_ids = modes_df["mode_id"].tolist()
    mode_ids = [m for m in mode_ids if m != "Other"]

    exposure_df["od_id"] = list(zip(exposure_df["origin_id"], exposure_df["destination_id"]))
    demand_map = exposure_df.set_index("od_id")["exposure_TEU_per_week"].to_dict()

    od_mode_df["od_id"] = list(zip(od_mode_df["origin_id"], od_mode_df["destination_id"]))
    od_mode_df = od_mode_df[od_mode_df["od_id"].isin(od_pairs["od_id"])]
    od_mode_df = od_mode_df[od_mode_df["mode_id"].isin(mode_ids)]

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

    if taste_draws.empty:
        raise RuntimeError("Taste_Draws sheet is empty.")

    draw_ids = sorted(taste_draws["draw_id"].unique().tolist())
    asc_map = {(int(r["draw_id"]), r["mode_id"]): float(r["ASC_r"])
            for _, r in taste_draws.iterrows()}

    beta_cost_map = {int(r["draw_id"]): float(r["beta_cost_r_per_100CHF"])
                    for _, r in taste_draws.drop_duplicates(subset=["draw_id"]).iterrows()}
    beta_time_map = {int(r["draw_id"]): float(r["beta_time_r_per_hour"])
                    for _, r in taste_draws.drop_duplicates(subset=["draw_id"]).iterrows()}
    beta_freq_map = {int(r["draw_id"]): float(r["beta_freq_r_per_departure"])
                    for _, r in taste_draws.drop_duplicates(subset=["draw_id"]).iterrows()}

    globals_map = dict(zip(globals_df["parameter"], globals_df["value"]))
    big_M_utility = float(globals_map.get("big_M_utility", 1000.0))

    # ============================================================
    # 2. BUILD MODEL
    # ============================================================

    m = ConcreteModel()

    # Sets
    m.OD = Set(initialize=list(od_pairs["od_id"]), dimen=2)
    m.MODES = Set(initialize=mode_ids)
    m.R = Set(initialize=draw_ids)

    # Parameters
    m.Demand = Param(m.OD, initialize=lambda mod, o, d: demand_map.get((o, d), 0.0))
    m.t_hours = Param(m.OD, m.MODES,
                    initialize=lambda mod, o, d, mo: t_hours.get(((o, d), mo), 0.0))
    m.p_min = Param(m.OD, m.MODES,
                    initialize=lambda mod, o, d, mo: p_min.get(((o, d), mo), 0.0))
    m.p_max = Param(m.OD, m.MODES,
                    initialize=lambda mod, o, d, mo: p_max.get(((o, d), mo), 0.0))
    m.f_min = Param(m.OD, m.MODES,
                    initialize=lambda mod, o, d, mo: f_min.get(((o, d), mo), 0.0))
    m.f_max = Param(m.OD, m.MODES,
                    initialize=lambda mod, o, d, mo: f_max.get(((o, d), mo), 0.0))
    m.cap_per_dep = Param(m.OD, m.MODES,
                        initialize=lambda mod, o, d, mo: cap_per_dep.get(((o, d), mo), 0.0))
    m.C_fix = Param(m.OD, m.MODES,
                    initialize=lambda mod, o, d, mo: c_fix.get(((o, d), mo), 0.0))
    m.C_var = Param(m.OD, m.MODES,
                    initialize=lambda mod, o, d, mo: c_var.get(((o, d), mo), 0.0))
    m.ASC = Param(m.R, m.MODES,
                initialize=lambda mod, r, mo: asc_map.get((int(r), mo), 0.0))
    m.beta_cost = Param(m.R, initialize=lambda mod, r: beta_cost_map[int(r)])
    m.beta_time = Param(m.R, initialize=lambda mod, r: beta_time_map[int(r)])
    m.beta_freq = Param(m.R, initialize=lambda mod, r: beta_freq_map[int(r)])
    m.bigM = Param(initialize=big_M_utility)


    # MIN_TEU_IM = 5.0

    # def min_teu_im_init(mod, o, d):
    #     total_demand = mod.Demand[o, d]  # if you have this param
    #     return min(MIN_TEU_IM, total_demand)  # don't force more than exists

    # m.MinTEU_IM = Param(m.OD,initialize=min_teu_im_init,within=NonNegativeReals)

    # Variables
    m.x = Var(m.OD, m.MODES, domain=Binary)
    m.f = Var(m.OD, m.MODES, domain=NonNegativeReals)
    m.p = Var(m.OD, m.MODES, domain=NonNegativeReals)
    m.y = Var(m.OD, m.MODES, domain=NonNegativeReals)
    m.w = Var(m.OD, m.R, m.MODES, domain=Binary)
    
    # Benders variable for subproblem cost approximation
    m.theta = Var(domain=NonNegativeReals)

    # Constraints (same as before)
    def price_lb_rule(mod, o, d, mo):
        return mod.p[o, d, mo] >= mod.p_min[o, d, mo] * mod.x[o, d, mo]
    m.PriceLB = Constraint(m.OD, m.MODES, rule=price_lb_rule)

    def price_ub_rule(mod, o, d, mo):
        return mod.p[o, d, mo] <= mod.p_max[o, d, mo] * mod.x[o, d, mo]
    m.PriceUB = Constraint(m.OD, m.MODES, rule=price_ub_rule)

    def freq_lb_rule(mod, o, d, mo):
        return mod.f[o, d, mo] >= mod.f_min[o, d, mo] * mod.x[o, d, mo]
    m.FreqLB = Constraint(m.OD, m.MODES, rule=freq_lb_rule)

    def freq_ub_rule(mod, o, d, mo):
        return mod.f[o, d, mo] <= mod.f_max[o, d, mo] * mod.x[o, d, mo]
    m.FreqUB = Constraint(m.OD, m.MODES, rule=freq_ub_rule)

    def capacity_rule(mod, o, d, mo):
        return mod.y[o, d, mo] <= mod.cap_per_dep[o, d, mo] * mod.f[o, d, mo]
    m.Capacity = Constraint(m.OD, m.MODES, rule=capacity_rule)

    def flow_activation_rule(mod, o, d, mo):
        return mod.y[o, d, mo] <= mod.Demand[o, d] * mod.x[o, d, mo]
    m.FlowActivation = Constraint(m.OD, m.MODES, rule=flow_activation_rule)

    # ✅ FIX 3: Ensure positive frequency when flow exists
    # If y > 0, then f >= f_min (via x=1 and FreqLB)
    # This is implicitly handled by FreqLB + FlowActivation, but we can add explicit link:
    def freq_when_flow_rule(mod, o, d, mo):
        # If y > epsilon, then x must be 1, which forces f >= f_min
        # This is already handled by FlowActivation, so this is redundant but safe
        return Constraint.Skip  # Already covered by FlowActivation + FreqLB
    m.FreqWhenFlow = Constraint(m.OD, m.MODES, rule=freq_when_flow_rule)

    # --- SAA demand allocation ---
    def demand_alloc_rule(mod, o, d, mo):
        R = len(mod.R)
        if R == 0:
            return mod.y[o, d, mo] == 0
        return mod.y[o, d, mo] == (mod.Demand[o, d] / R) * sum(mod.w[o, d, r, mo] for r in mod.R)
    m.DemandAlloc = Constraint(m.OD, m.MODES, rule=demand_alloc_rule)

    def choice_norm_rule(mod, o, d, r):
        return sum(mod.w[o, d, r, mo] for mo in mod.MODES) == 1
    m.ChoiceNormalization = Constraint(m.OD, m.R, rule=choice_norm_rule)

    # Minimum flow if a mode is open (to avoid x=1, y=0)
    epsilon_f = 0.001  # or 0.1 TEU if you're in whole-container units

    def min_use_if_open_rule(mod, o, d, mo):
        return mod.f[o, d, mo] >= epsilon_f * mod.x[o, d, mo]

    m.MinUseIfOpen = Constraint(m.OD, m.MODES, rule=min_use_if_open_rule)

    # def min_im_flow_if_open_rule(mod, o, d):
    #     if ("Intermodal" not in mod.MODES):
    #         return Constraint.Skip
    #     return mod.y[o, d, "Intermodal"] >= mod.MinTEU_IM[o, d] * mod.x[o, d, "Intermodal"]

    # m.MinIMFlowIfOpen = Constraint(m.OD, rule=min_im_flow_if_open_rule)

    # --- Argmax linearization ---
    def argmax_rule(mod, o, d, r, mo1, mo2):
        if mo1 == mo2:
            return Constraint.Skip

        U1 = (mod.ASC[r, mo1]
            + mod.beta_cost[r] * (mod.p[o, d, mo1] / 100.0)
            + mod.beta_time[r] * mod.t_hours[o, d, mo1]
            + mod.beta_freq[r] * mod.f[o, d, mo1])

        U2 = (mod.ASC[r, mo2]
            + mod.beta_cost[r] * (mod.p[o, d, mo2] / 100.0)
            + mod.beta_time[r] * mod.t_hours[o, d, mo2]
            + mod.beta_freq[r] * mod.f[o, d, mo2])

        return U1 >= U2 - mod.bigM * (1 - mod.w[o, d, r, mo1])

    m.Argmax = Constraint(m.OD, m.R, m.MODES, m.MODES, rule=argmax_rule)

    # --- Benders cuts placeholder ---
    m.BendersCuts = ConstraintList()

    # Objective
    def profit_expr(mod):
        revenue = sum(mod.p[o, d, mo] * mod.y[o, d, mo] 
                    for (o, d) in mod.OD for mo in mod.MODES)
        op_cost = sum(mod.C_fix[o, d, mo] * mod.f[o, d, mo]
                    + mod.C_var[o, d, mo] * mod.y[o, d, mo]
                    for (o, d) in mod.OD for mo in mod.MODES)
        return revenue - op_cost - mod.theta

    m.OBJ = Objective(rule=profit_expr, sense=maximize)

    return m

def solve_master(model):
    """Solve the master problem"""
    solver = SolverFactory("gurobi")
    results = solver.solve(model, tee=False)  # Set tee=True for detailed output
    
    if results.solver.termination_condition == TerminationCondition.optimal:
        return results
    else:
        print(f"Master problem solve status: {results.solver.termination_condition}")
        return None
    


if __name__ == "__main__":
    # Standalone execution
    m = build_master_problem()
    results = solve_master(m)
    
    if results:
        print(f"Master objective: {value(m.OBJ):.2f}")
        
        # Save solution
        output_data = []
        for (o, d) in m.OD:
            for mo in m.MODES:
                output_data.append({
                    "origin": o, "destination": d, "mode": mo,
                    "x_open": value(m.x[o, d, mo]), "frequency_f": value(m.f[o, d, mo]),
                    "price_p": value(m.p[o, d, mo]), "TEU_y": value(m.y[o, d, mo]),
                    "cap_per_dep_TEU": value(m.cap_per_dep[o, d, mo])
                })
        
        df_decisions = pd.DataFrame(output_data)
        df_decisions.to_excel("model_output/master_solution_decisions.xlsx", index=False)
        print("Solution saved to model_output/master_solution_decisions.xlsx")