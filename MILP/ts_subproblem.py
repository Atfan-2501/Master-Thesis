import pandas as pd
import numpy as np
from pyomo.environ import *

def solve_ts_subproblem(master_solution):
    """
    Solve TS subproblem with fixed master decisions.
    Returns: (objective_value, dual_info, model) if feasible, None if infeasible
    """
    
    # Load TS data
    TS_XLSX = "model_output/TS_Subproblem_Inputs_extended.xlsx"
    ts_nodes = pd.read_excel(TS_XLSX, sheet_name="TS_Nodes")
    ts_arcs = pd.read_excel(TS_XLSX, sheet_name="TS_Arcs")
    train_groups = pd.read_excel(TS_XLSX, sheet_name="Train_Groups")

    # Filter out ODs with zero flow
    intermodal_flows = master_solution.get('intermodal_flows', {})
    if not intermodal_flows:
        print("No intermodal flows to route")
        return 0.0, {}, None

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
            group_of_od[od] = f"{o}-{d}_IM"
        
        source_node_of_od[od] = f"OD_{o}_{d}_source"
        sink_node_of_od[od] = f"OD_{o}_{d}_sink"

    m.cost_of_arc = Param(m.A, initialize=cost_of_arc)
    m.base_cap_of_arc = Param(m.A, initialize=base_cap_of_arc)
    m.y_req = Param(m.OD, initialize=y_req)
    m.f_lev = Param(m.OD, initialize=f_lev)
    m.cap_per_dep = Param(m.OD, initialize=cap_per_dep)

    # Store as attributes for dual extraction
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
        
        return train_flow <= capacity_limit

    m.GroupCapacity = Constraint(m.OD, rule=group_capacity_rule)

    # Objective
    def ts_cost_expr(mod):
        return sum(
            mod.cost_of_arc[a] * mod.flow[o, d, a]
            for (o, d) in mod.OD for a in mod.A
        )

    m.TS_Obj = Objective(rule=ts_cost_expr, sense=minimize)

    # Enable duals from the solver
    if not hasattr(m, "dual"):
        m.dual = Suffix(direction=Suffix.IMPORT)

    # Solve with dual information
    solver = SolverFactory("gurobi")
    solver.options['QCPDual'] = 1  # Enable dual information
    results = solver.solve(m, tee=False, load_solutions=True)

    if results.solver.termination_condition == TerminationCondition.optimal:
        objective_value = value(m.TS_Obj)
        
        # Extract proper dual values
        dual_info = extract_dual_information(m, results)
        
        # Save results
        save_subproblem_results(m, master_solution, ts_arcs, train_groups)
        
        print(f"✅ Subproblem solved successfully - Cost: {objective_value:.2f}")
        return objective_value, dual_info, m
    
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        print(f"❌ Subproblem infeasible")
        
        # Compute IIS (Irreducible Inconsistent Subsystem)
        infeasibility_info = compute_minimal_infeasible_set(m, solver)
        
        return None, infeasibility_info, m
    
    else:
        print(f"❌ Subproblem failed: {results.solver.termination_condition}")
        return None, {}, m


def extract_dual_information(model, results):
    """
    Extract dual information for optimality cuts using Pyomo's dual suffix.

    Returns dual values for:
    1. NodeBalance source constraints  -> flow_coefficients π_od (w.r.t. y_req)
    2. GroupCapacity constraints      -> frequency_coefficients μ_od (w.r.t. f_lev)
    """
    dual_info = {
        "constant_term": 0.0,
        "flow_coefficients": {},
        "frequency_coefficients": {},
        "constraint_details": {},
        "extraction_method": "pyomo_dual_suffix",
        "extraction_successful": False,
    }

    try:
        # We assume model.dual was created BEFORE solve_ts_subproblem called solver.solve
        if not hasattr(model, "dual"):
            print("   ⚠️  No dual suffix on model - cannot extract duals")
            return dual_info

        # Use the existing helper which reads model.dual[constraint]
        success = extract_duals_via_suffix(model, dual_info)

        if not success:
            print("   ⚠️  Dual extraction via Pyomo suffix failed or returned no duals")
            return dual_info

        # Compute constant term α₀ so that:
        #   θ ≥ α₀ + Σ π_od y_req_od + Σ μ_od f_lev_od
        # reproduces current TS objective at the current (y,f)
        current_cost = value(model.TS_Obj)

        flow_term = sum(
            dual_info["flow_coefficients"].get((o, d), 0.0) * value(model.y_req[o, d])
            for (o, d) in model.OD
        )

        freq_term = sum(
            dual_info["frequency_coefficients"].get((o, d), 0.0) * value(model.f_lev[o, d])
            for (o, d) in model.OD
        )

        dual_info["constant_term"] = current_cost - flow_term - freq_term
        dual_info["objective_value"] = current_cost
        dual_info["num_ods"] = len(model.OD)
        dual_info["extraction_successful"] = True

        print(
            f"   Dual summary: α₀={dual_info['constant_term']:.2f}, "
            f"{len(dual_info['flow_coefficients'])} flow coeffs, "
            f"{len(dual_info['frequency_coefficients'])} freq coeffs"
        )

    except Exception as e:
        print(f"⚠️  Exception during dual extraction: {e}")
        import traceback
        traceback.print_exc()
        dual_info["extraction_successful"] = False

    return dual_info


def extract_duals_via_suffix(model, dual_info):
    """Extract duals using Pyomo dual suffix"""
    try:
        # Create dual suffix if it doesn't exist
        if not hasattr(model, 'dual'):
            model.dual = Suffix(direction=Suffix.IMPORT)
        
        # Extract from GroupCapacity constraints
        for (o, d) in model.OD:
            try:
                constraint = model.GroupCapacity[o, d]
                dual_value = model.dual[constraint]
                
                if abs(dual_value) > 1e-10:
                    # Dual w.r.t. frequency: -dual * cap_per_dep
                    dual_info['frequency_coefficients'][(o, d)] = -dual_value * value(model.cap_per_dep[o, d])
                    
                    dual_info['constraint_details'][f'GroupCap_{o}_{d}'] = {
                        'dual': dual_value,
                        'current_flow': value(model.y_req[o, d]),
                        'current_freq': value(model.f_lev[o, d]),
                    }
            except:
                pass
        
        # Extract from NodeBalance constraints (source nodes)
        for (o, d) in model.OD:
            try:
                source_node = model.source_node_of_od[o, d]
                constraint = model.NodeBalance[o, d, source_node]
                dual_value = model.dual[constraint]
                
                if abs(dual_value) > 1e-10:
                    dual_info['flow_coefficients'][(o, d)] = dual_value
                    
                    dual_info['constraint_details'][f'NodeBalance_{o}_{d}'] = {
                        'dual': dual_value,
                        'flow_requirement': value(model.y_req[o, d])
                    }
            except:
                pass
        
        # Check if we got any duals
        has_duals = len(dual_info['flow_coefficients']) > 0 or len(dual_info['frequency_coefficients']) > 0
        return has_duals
        
    except Exception as e:
        print(f"   Suffix method failed: {e}")
        return False


def extract_duals_via_gurobi(model, dual_info):
    """Extract duals directly from Gurobi model"""
    try:
        # This requires the solver object, which we don't have here
        # Would need to pass it from solve_ts_subproblem
        return False
    except:
        return False


def extract_duals_via_results(model, results, dual_info):
    """Extract duals from results object"""
    try:
        if not results or not hasattr(results, 'solution'):
            return False
        
        solution = results.solution(0)
        if not hasattr(solution, 'constraint'):
            return False
        
        # Try to extract from constraint dictionary
        for (o, d) in model.OD:
            # Group capacity
            try:
                constraint = model.GroupCapacity[o, d]
                if constraint in solution.constraint:
                    dual_value = solution.constraint[constraint].get('dual', 0.0)
                    if abs(dual_value) > 1e-10:
                        dual_info['frequency_coefficients'][(o, d)] = -dual_value * value(model.cap_per_dep[o, d])
            except:
                pass
            
            # Node balance
            try:
                source_node = model.source_node_of_od[o, d]
                constraint = model.NodeBalance[o, d, source_node]
                if constraint in solution.constraint:
                    dual_value = solution.constraint[constraint].get('dual', 0.0)
                    if abs(dual_value) > 1e-10:
                        dual_info['flow_coefficients'][(o, d)] = dual_value
            except:
                pass
        
        has_duals = len(dual_info['flow_coefficients']) > 0 or len(dual_info['frequency_coefficients']) > 0
        return has_duals
        
    except Exception as e:
        print(f"   Results method failed: {e}")
        return False


def compute_minimal_infeasible_set(model, solver):
    """
    Compute Irreducible Inconsistent Subsystem (IIS) for infeasible subproblem.
    This identifies the minimal set of constraints causing infeasibility.
    """
    infeas_info = {
        'type': 'infeasibility',
        'violated_constraints': [],
        'affected_ods': set(),
        'bottleneck_arcs': [],
        'recommendations': []
    }
    
    try:
        # Use Gurobi's IIS computation
        if solver.name == 'gurobi':
            print("   Computing IIS (Irreducible Inconsistent Subsystem)...")
            
            # Get Gurobi model object
            gurobi_model = solver._solver_model
            
            # Compute IIS
            gurobi_model.computeIIS()
            
            # Extract IIS constraints
            for constr in gurobi_model.getConstrs():
                if constr.IISConstr:
                    constr_name = constr.ConstrName
                    infeas_info['violated_constraints'].append({
                        'name': constr_name,
                        'type': 'constraint',
                        'slack': constr.Slack if hasattr(constr, 'Slack') else None
                    })
                    
                    # Parse constraint name to identify affected ODs
                    if 'GroupCapacity' in constr_name:
                        # Extract OD from constraint name
                        parts = constr_name.split('_')
                        if len(parts) >= 3:
                            od = (parts[1], parts[2])
                            infeas_info['affected_ods'].add(od)
                            infeas_info['recommendations'].append(
                                f"Increase frequency for {od[0]} → {od[1]}"
                            )
                    
                    elif 'ArcCapacity' in constr_name:
                        infeas_info['bottleneck_arcs'].append(constr_name)
                        infeas_info['recommendations'].append(
                            f"Arc capacity exceeded: {constr_name}"
                        )
            
            # Write IIS to file for debugging
            iis_file = "model_output/subproblem_iis.ilp"
            gurobi_model.write(iis_file)
            print(f"   IIS written to {iis_file}")
            
        else:
            # Fallback: heuristic analysis
            print("   Using heuristic infeasibility analysis...")
            infeas_info = heuristic_infeasibility_analysis(model)
    
    except Exception as e:
        print(f"⚠️  Warning: Could not compute IIS: {e}")
        # Fallback to heuristic
        infeas_info = heuristic_infeasibility_analysis(model)
    
    infeas_info['affected_ods'] = list(infeas_info['affected_ods'])
    
    print(f"   Found {len(infeas_info['violated_constraints'])} violated constraints")
    print(f"   Affected ODs: {infeas_info['affected_ods']}")
    
    return infeas_info


def heuristic_infeasibility_analysis(model):
    """
    Heuristic method to identify likely causes of infeasibility.
    Used when IIS computation is not available.
    """
    infeas_info = {
        'type': 'infeasibility_heuristic',
        'violated_constraints': [],
        'affected_ods': set(),
        'bottleneck_arcs': [],
        'recommendations': []
    }
    
    # Check group capacity violations
    for (o, d) in model.OD:
        required_capacity = model.y_req[o, d]
        available_capacity = model.cap_per_dep[o, d] * model.f_lev[o, d]
        
        utilization = required_capacity / available_capacity if available_capacity > 0 else float('inf')
        
        if utilization > 1.0:
            infeas_info['affected_ods'].add((o, d))
            infeas_info['violated_constraints'].append({
                'name': f'GroupCapacity_{o}_{d}',
                'type': 'capacity',
                'utilization': utilization,
                'required': required_capacity,
                'available': available_capacity
            })
            infeas_info['recommendations'].append(
                f"Reduce flow or increase frequency for {o} → {d} "
                f"(utilization: {utilization*100:.1f}%)"
            )
    
    return infeas_info


def save_subproblem_results(model, master_solution, ts_arcs_df, train_groups_df):
    """Save subproblem results in Excel format"""
    
    # Create Arc_Flows sheet
    arc_flows_data = []
    for a in model.A:
        total_flow = sum(value(model.flow[o, d, a]) for (o, d) in model.OD)
        if total_flow > 1e-6:
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
    
    # Save to Excel
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