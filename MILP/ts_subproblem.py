import pandas as pd
from pyomo.environ import *

def solve_ts_subproblem(master_solution):
    """Solve TS subproblem with fixed master decisions"""
    
    # Load TS data
    TS_XLSX = "model_output/TS_Subproblem_Inputs_extended.xlsx"
    ts_nodes = pd.read_excel(TS_XLSX, sheet_name="TS_Nodes")
    ts_arcs = pd.read_excel(TS_XLSX, sheet_name="TS_Arcs")
    train_groups = pd.read_excel(TS_XLSX, sheet_name="Train_Groups")

    # Filter out ODs with zero flow
    intermodal_flows = master_solution.get('intermodal_flows', {})
    if not intermodal_flows:
        print("No intermodal flows to route")
        return 0.0, {}  # Zero cost, empty duals

    # Build model
    m = ConcreteModel()

    # Sets
    node_ids = ts_nodes["node_id"].tolist()
    arc_ids = ts_arcs["arc_id"].tolist()
    
    m.N = Set(initialize=node_ids)
    m.A = Set(initialize=arc_ids)
    m.OD = Set(initialize=list(intermodal_flows.keys()), dimen=2)

    # Parameters from TS network
    from_node = ts_arcs.set_index("arc_id")["from_node"].to_dict()
    to_node = ts_arcs.set_index("arc_id")["to_node"].to_dict()
    arc_type = ts_arcs.set_index("arc_id")["arc_type"].to_dict()
    cost_of_arc = ts_arcs.set_index("arc_id")["cost_chf_per_TEU"].to_dict()
    base_cap_of_arc = ts_arcs.set_index("arc_id")["base_cap_TEU"].to_dict()
    group_of_arc = ts_arcs.set_index("arc_id")["group_id"].fillna("").to_dict()

    # OD-specific parameters from master solution
    y_req = {}
    f_lev = {}
    cap_per_dep = {}
    group_of_od = {}
    source_node_of_od = {}
    sink_node_of_od = {}

    for od, data in intermodal_flows.items():
        o, d = od
        y_req[od] = data['flow']
        f_lev[od] = data['frequency']
        cap_per_dep[od] = data['cap_per_dep']
        
        # Find group ID from train groups
        group_row = train_groups[
            (train_groups['origin_id'] == o) & 
            (train_groups['destination_id'] == d)
        ]
        if not group_row.empty:
            group_of_od[od] = group_row.iloc[0]['group_id']
        else:
            group_of_od[od] = f"{o}-{d}_IM"  # Fallback
        
        source_node_of_od[od] = f"OD_{o}_{d}_source"
        sink_node_of_od[od] = f"OD_{o}_{d}_sink"

    m.cost_of_arc = Param(m.A, initialize=cost_of_arc)
    m.base_cap_of_arc = Param(m.A, initialize=base_cap_of_arc)
    m.y_req = Param(m.OD, initialize=y_req)
    m.f_lev = Param(m.OD, initialize=f_lev)
    m.cap_per_dep = Param(m.OD, initialize=cap_per_dep)

    m.group_of_od = group_of_od
    m.group_of_arc = group_of_arc
    m.arc_type = arc_type
    m.from_node = from_node
    m.to_node = to_node
    m.source_node_of_od = source_node_of_od
    m.sink_node_of_od = sink_node_of_od

    # Variables
    m.flow = Var(m.OD, m.A, domain=NonNegativeReals)

    # Constraints
    def node_balance_rule(mod, o, d, n):
        inflow = sum(mod.flow[o, d, a] for a in mod.A if mod.to_node[a] == n)
        outflow = sum(mod.flow[o, d, a] for a in mod.A if mod.from_node[a] == n)
        source_n = mod.source_node_of_od.get((o, d), None)
        sink_n = mod.sink_node_of_od.get((o, d), None)
        
        if n == source_n:
            return outflow - inflow == mod.y_req[o, d]
        elif n == sink_n:
            return inflow - outflow == mod.y_req[o, d]
        else:
            return inflow - outflow == 0

    m.NodeBalance = Constraint(m.OD, m.N, rule=node_balance_rule)

    def arc_capacity_rule(mod, a):
        return sum(mod.flow[o, d, a] for (o, d) in mod.OD) <= mod.base_cap_of_arc[a]

    m.ArcCapacity = Constraint(m.A, rule=arc_capacity_rule)

    def group_capacity_rule(mod, o, d):
        od = (o, d)
        g_od = mod.group_of_od.get(od, None)
        if g_od is None:
            return Constraint.Skip
        
        capacity_limit = mod.cap_per_dep[o, d] * mod.f_lev[o, d]
        
        # Only count train arcs
        train_flow = sum(
            mod.flow[o, d, a]
            for a in mod.A
            if (mod.group_of_arc.get(a, "") == g_od and mod.arc_type.get(a) == "train")
        )
        
        return train_flow <= capacity_limit * 1.01  # 1% tolerance

    m.GroupCapacity = Constraint(m.OD, rule=group_capacity_rule)

    # Objective
    def ts_cost_expr(mod):
        return sum(
            mod.cost_of_arc[a] * mod.flow[o, d, a]
            for (o, d) in mod.OD for a in mod.A
        )

    m.TS_Obj = Objective(rule=ts_cost_expr, sense=minimize)

    # Solve
    solver = SolverFactory("gurobi")
    results = solver.solve(m, tee=False)

    if results.solver.termination_condition == TerminationCondition.optimal:
        objective_value = value(m.TS_Obj)
        
        # Extract dual values (simplified - in practice you'd get proper duals)
        duals = extract_duals_simplified(m)
        
        # NEW: Save the subproblem results in the desired format
        save_subproblem_results(m, master_solution, ts_arcs, train_groups)
        
        print(f"✅ Subproblem solved successfully - Cost: {objective_value:.2f}")
        return objective_value, duals
    else:
        print(f"❌ Subproblem failed: {results.solver.termination_condition}")
        return None

def save_subproblem_results(model, master_solution, ts_arcs_df, train_groups_df):
    """Save subproblem results in the desired Excel format"""
    
    # Create Arc_Flows sheet
    arc_flows_data = []
    for a in model.A:
        total_flow = sum(value(model.flow[o, d, a]) for (o, d) in model.OD)
        if total_flow > 1e-6:  # Only include arcs with non-zero flow
            arc_flows_data.append({
                'origin_id': 'All',  # Or you could track which OD uses which arc
                'destination_id': 'All',
                'arc_id': a,
                'from_node': model.from_node[a],
                'to_node': model.to_node[a],
                'arc_type': model.arc_type[a],
                'flow_TEU': total_flow,
                'arc_cost_per_TEU': model.cost_of_arc[a],
                'group_id': model.group_of_arc[a]
            })
    
    arc_flows_df = pd.DataFrame(arc_flows_data)
    
    # Create OD_Summary sheet
    od_summary_data = []
    for (o, d) in model.OD:
        y_req = model.y_req[o, d]
        
        # Calculate flow on trains for this OD
        g_od = model.group_of_od.get((o, d), "")
        train_flow = sum(
            value(model.flow[o, d, a])
            for a in model.A
            if (model.group_of_arc.get(a, "") == g_od and model.arc_type.get(a) == "train")
        )
        
        capacity_limit = model.cap_per_dep[o, d] * model.f_lev[o, d]
        utilization = (train_flow / capacity_limit * 100) if capacity_limit > 0 else 0
        
        od_summary_data.append({
            'origin_id': o,
            'destination_id': d,
            'y_req': y_req,
            'flow_on_trains': train_flow,
            'cap_group': capacity_limit,
            'utilization_%': utilization,
            'group_id': g_od
        })
    
    od_summary_df = pd.DataFrame(od_summary_data)
    
    # Save to Excel file
    output_file = "model_output/TS_Subproblem_Solution.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        arc_flows_df.to_excel(writer, sheet_name='Arc_Flows', index=False)
        od_summary_df.to_excel(writer, sheet_name='OD_Summary', index=False)
    
    print(f"✓ Subproblem results saved to {output_file}")

def extract_duals_simplified(model):
    """Extract simplified dual values for Benders cuts"""
    duals = {}
    
    # In a real implementation, you would extract duals from constraints
    # This is a simplified version
    for od in model.OD:
        duals[od] = {
            'demand_dual': 1.0,  # Placeholder
            'capacity_dual': 1.0  # Placeholder
        }
    
    return duals

if __name__ == "__main__":
    # For testing
    test_solution = {
        'intermodal_flows': {
            ('Aarau', 'Chiasso'): {'flow': 100, 'frequency': 3, 'cap_per_dep': 80},
            ('Baselwolf', 'Chiasso'): {'flow': 500, 'frequency': 5, 'cap_per_dep': 80}
        }
    }
    
    result = solve_ts_subproblem(test_solution)
    if result:
        cost, duals = result
        print(f"Test subproblem cost: {cost:.2f}")