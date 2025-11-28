import pandas as pd
import numpy as np
import json
from pyomo.environ import *
import subprocess
import sys
import time
from datetime import datetime


class BendersDecomposition:
    def __init__(self, clean_start=True):
        self.master_model = None
        self.iteration = 0
        self.max_iterations = 50
        self.tolerance = 1e-4
        self.upper_bound = float("inf")
        self.lower_bound = float("-inf")
        self.convergence_data = []
        
        # Cut tracking
        self.optimality_cuts_added = 0
        self.feasibility_cuts_added = 0
        
        # Diagnostics
        self.subproblem_feasible_count = 0
        self.subproblem_infeasible_count = 0

        # Files
        self.master_sol_file = "model_output/master_solution_decisions.xlsx"
        self.ts_input_file = "model_output/TS_Subproblem_Inputs_extended.xlsx"
        self.benders_cuts_file = "model_output/benders_cuts.xlsx"
        
        # Clean previous run artifacts if requested
        if clean_start:
            self.clean_previous_run()

    # ============================================================
    # 0) Clean start
    # ============================================================
    def clean_previous_run(self):
        """Remove artifacts from previous runs to ensure clean state"""
        import os
        
        files_to_clean = [
            self.benders_cuts_file,
            "model_output/benders_convergence.xlsx",
            "model_output/subproblem_iis.ilp",
        ]
        
        for filepath in files_to_clean:
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"🧹 Cleaned: {filepath}")
                except Exception as e:
                    print(f"⚠️  Could not remove {filepath}: {e}")

    # ============================================================
    # 0) Network generation (unchanged)
    # ============================================================
    def initialize_network(self):
        """Regenerate the time-space network"""
        print("=== INITIALIZING TIME-SPACE NETWORK ===")

        result = subprocess.run(
            [sys.executable, "generate_connections.py"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate connections: {result.stderr}")
        print("✅ Connections generated")

        result = subprocess.run(
            [sys.executable, "generate_time_space_network.py"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to generate TS network: {result.stderr}")
        print("✅ Time-space network generated")

    # ============================================================
    # Master problem (unchanged)
    # ============================================================
    def solve_master_problem(self, add_cuts=True):
        """Solve the master problem with optional Benders cuts"""
        print(f"\n=== SOLVING MASTER PROBLEM (Iteration {self.iteration}) ===")

        from master_milp import build_master_problem, solve_master

        self.master_model = build_master_problem()

        if add_cuts and self.iteration > 1:
            self.add_benders_cuts_to_master()

        master_result = solve_master(self.master_model)
        if master_result is None:
            raise RuntimeError("Master problem failed to solve")

        master_solution = self.extract_master_solution()

        obj_value = value(self.master_model.OBJ)

        # Master is a relaxation → gives an UPPER bound for a maximization problem
        self.upper_bound = min(self.upper_bound, obj_value)

        print(f"Master objective (relaxed): {obj_value:.2f}")
        print(f"Current bounds: LB={self.lower_bound:.2f}, UB={self.upper_bound:.2f}")


        df = self.master_solution_to_df()
        df.to_excel(self.master_sol_file, index=False)
        print(f"✓ Master solution written to {self.master_sol_file}")

        return master_solution

    def master_solution_to_df(self):
        """Convert master solution to DataFrame"""
        rows = []
        for (o, d) in self.master_model.OD:
            for mo in self.master_model.MODES:
                rows.append(
                    {
                        "origin": o,
                        "destination": d,
                        "mode": mo,
                        "x_open": float(value(self.master_model.x[o, d, mo])),
                        "frequency_f": float(
                            value(self.master_model.f[o, d, mo])
                        ),
                        "price_p": float(
                            value(self.master_model.p[o, d, mo])
                        ),
                        "TEU_y": float(value(self.master_model.y[o, d, mo])),
                        "cap_per_dep_TEU": float(
                            value(self.master_model.cap_per_dep[o, d, mo])
                            if hasattr(self.master_model, "cap_per_dep")
                            else 0.0
                        ),
                    }
                )
        return pd.DataFrame(rows)

    # ============================================================
    # Subproblem (updated to use enhanced version)
    # ============================================================
    def solve_subproblem(self, master_solution):
        """Solve the time-space subproblem"""
        print(f"\n=== SOLVING SUBPROBLEM (Iteration {self.iteration}) ===")

        from ts_subproblem import solve_ts_subproblem

        subproblem_result = solve_ts_subproblem(master_solution)

        if subproblem_result is None:
            print("❌ Subproblem infeasible")
            return None

        # Unpack result
        if len(subproblem_result) == 3:
            sp_obj, info, sp_model = subproblem_result
        else:
            sp_obj, info = subproblem_result
            sp_model = None

        # Check if this is infeasibility info or dual info
        if info.get('type') == 'infeasibility' or info.get('type') == 'infeasibility_heuristic':
            print("❌ Subproblem infeasible - generating feasibility cut")
            self.generate_proper_feasibility_cut(master_solution, info)
            return None

        # Expect (sp_obj, duals) or (sp_obj, duals, model)
        if len(subproblem_result) == 2:
            sp_obj, duals = subproblem_result
        else:
            sp_obj, duals, _ = subproblem_result

        # Update upper bound: correct master objective using true TS cost
        master_obj = value(self.master_model.OBJ)
        theta_value = (
            value(self.master_model.theta)
            if hasattr(self.master_model, "theta")
            else 0.0
        )

        # True profit = (revenue - op_cost - theta) + (theta - Z_TS)
        true_obj = master_obj + theta_value - sp_obj

        # For a maximization problem, this is a LOWER bound on the optimal profit
        self.lower_bound = max(self.lower_bound, true_obj)

        print(f"Subproblem cost (TS routing): {sp_obj:.2f}")
        print(f"True profit at current solution: {true_obj:.2f}")
        print(f"Current bounds: LB={self.lower_bound:.2f}, UB={self.upper_bound:.2f}")


        self.generate_proper_optimality_cut(master_solution, info, sp_obj)

        return sp_obj, info

    # ============================================================
    # PROPER OPTIMALITY CUT GENERATION
    # ============================================================
    def generate_proper_optimality_cut(self, master_solution, dual_info, sp_cost):
        """
        Generate proper dual-based optimality cut:
        
        θ ≥ α₀ + Σᵢ πᵢ yᵢ + Σⱼ μⱼ fⱼ
        
        where:
        - α₀ is the constant term
        - πᵢ are dual values for flow variables
        - μⱼ are dual values for frequency variables
        """
        print("Generating proper dual-based optimality cut...")

        cut_data = {
            "iteration": self.iteration,
            "cut_type": "optimality",
            "subproblem_cost": float(sp_cost),
            "timestamp": datetime.now().isoformat(),
            
            # Dual coefficients
            "constant_term": dual_info.get('constant_term', sp_cost),
            "flow_coefficients": {},
            "frequency_coefficients": {},
            
            # Metadata
            "num_ods": dual_info.get('num_ods', 0),
            "objective_value": dual_info.get('objective_value', sp_cost)
        }

        # Extract flow coefficients (πᵢ)
        flow_coeffs = dual_info.get('flow_coefficients', {})
        for od, coeff in flow_coeffs.items():
            cut_data["flow_coefficients"][f"{od[0]}_{od[1]}"] = float(coeff)

        # Extract frequency coefficients (μⱼ)
        freq_coeffs = dual_info.get('frequency_coefficients', {})
        for od, coeff in freq_coeffs.items():
            cut_data["frequency_coefficients"][f"{od[0]}_{od[1]}"] = float(coeff)

        # Compute expected cut strength
        # Cut value at current point should equal sp_cost
        alpha_0 = cut_data["constant_term"]
        cut_value_at_current = alpha_0
        
        for od, data in master_solution['intermodal_flows'].items():
            od_key = f"{od[0]}_{od[1]}"
            pi = cut_data["flow_coefficients"].get(od_key, 0.0)
            mu = cut_data["frequency_coefficients"].get(od_key, 0.0)
            
            cut_value_at_current += pi * data['flow'] + mu * data['frequency']

        cut_data["cut_value_at_current"] = cut_value_at_current
        cut_data["expected_gap"] = abs(cut_value_at_current - sp_cost)

        print(f"   Cut: θ ≥ {alpha_0:.2f} + Σπᵢyᵢ + Σμⱼfⱼ")
        print(f"   At current point: {cut_value_at_current:.2f} (expected: {sp_cost:.2f})")
        print(f"   Gap: {cut_data['expected_gap']:.4f}")

        if cut_data["expected_gap"] > 1.0:
            print(f"   ⚠️  Warning: Large cut gap - check dual extraction")

        self.save_benders_cut(cut_data)
        self.optimality_cuts_added += 1

    # ============================================================
    # PROPER FEASIBILITY CUT GENERATION
    # ============================================================
    def generate_proper_feasibility_cut(self, master_solution, infeas_info):
        """
        Generate proper constraint-based feasibility cut from IIS.
        
        For each violated OD in the minimal infeasible set:
        - If capacity violated: require f_od ≥ y_od / cap_per_dep
        - If multiple ODs: generate combinatorial cut
        """
        print("Generating proper constraint-based feasibility cut...")

        cut_data = {
            "iteration": self.iteration,
            "cut_type": "feasibility",
            "timestamp": datetime.now().isoformat(),
            "infeasibility_type": infeas_info.get('type', 'unknown'),
            "violated_constraints": infeas_info.get('violated_constraints', []),
            "affected_ods": infeas_info.get('affected_ods', []),
            "recommendations": infeas_info.get('recommendations', [])
        }

        affected_ods = infeas_info.get('affected_ods', [])
        
        if not affected_ods:
            print("   ⚠️  No specific ODs identified - using heuristic cut")
            # Fallback: reduce total intermodal flow by 10%
            total_im_flow = sum(
                data['flow'] 
                for data in master_solution['intermodal_flows'].values()
            )
            cut_data["cut_form"] = "total_flow_reduction"
            cut_data["rhs_value"] = 0.9 * total_im_flow
            cut_data["affected_ods"] = list(master_solution['intermodal_flows'].keys())
        
        else:
            # Proper constraint-based cut
            # For each violated OD: enforce f ≥ y / cap_per_dep
            
            cut_data["cut_form"] = "frequency_capacity_ratio"
            cut_data["od_constraints"] = []
            
            for od in affected_ods:
                if isinstance(od, str):
                    # Parse string format "Origin_Dest"
                    parts = od.split('_')
                    if len(parts) >= 2:
                        od = (parts[0], parts[1])
                    else:
                        continue
                
                # Get current values
                flow_data = master_solution['intermodal_flows'].get(od)
                if flow_data:
                    current_y = flow_data['flow']
                    current_f = flow_data['frequency']
                    cap_per_dep = flow_data['cap_per_dep']
                    
                    # Required frequency for current flow
                    required_f = current_y / cap_per_dep if cap_per_dep > 0 else 0
                    
                    # Add constraint: f_od ≥ 1.05 * required_f (5% safety margin)
                    min_freq = 1.05 * required_f
                    
                    cut_data["od_constraints"].append({
                        "origin": od[0],
                        "destination": od[1],
                        "current_flow": current_y,
                        "current_freq": current_f,
                        "cap_per_dep": cap_per_dep,
                        "required_freq": required_f,
                        "min_freq": min_freq,
                        "constraint": f"f[{od[0]},{od[1]},Intermodal] >= {min_freq:.2f}"
                    })
            
            print(f"   Generated {len(cut_data['od_constraints'])} frequency constraints")
            for constraint_info in cut_data["od_constraints"]:
                print(f"      {constraint_info['origin']} → {constraint_info['destination']}: "
                      f"f ≥ {constraint_info['min_freq']:.2f} "
                      f"(current: {constraint_info['current_freq']:.2f})")

        self.save_benders_cut(cut_data)
        self.feasibility_cuts_added += 1

    # ============================================================
    # ADD CUTS TO MASTER
    # ============================================================
    def add_benders_cuts_to_master(self):
        """Read all stored cuts and add them to master model"""
        try:
            cuts_df = pd.read_excel(self.benders_cuts_file)
        except FileNotFoundError:
            print("No previous Benders cuts found")
            return

        feasibility_cuts = cuts_df[cuts_df["cut_type"] == "feasibility"]
        optimality_cuts = cuts_df[cuts_df["cut_type"] == "optimality"]

        print(f"Adding {len(feasibility_cuts)} feasibility cuts and "
              f"{len(optimality_cuts)} optimality cuts")

        # Add feasibility cuts
        for idx, cut in feasibility_cuts.iterrows():
            self.add_feasibility_cut_to_master(cut)

        # Add optimality cuts
        for idx, cut in optimality_cuts.iterrows():
            self.add_optimality_cut_to_master(cut)

    def add_optimality_cut_to_master(self, cut):
        """
        Add proper dual-based optimality cut to master:
        θ ≥ α₀ + Σᵢ πᵢ yᵢ + Σⱼ μⱼ fⱼ
        """
        if not hasattr(self.master_model, "BendersCuts"):
            print("⚠️ Master model has no BendersCuts ConstraintList")
            return

        try:
            # Extract coefficients
            alpha_0 = float(cut.get('constant_term', 0.0))
            
            # Parse flow coefficients
            flow_coeffs_str = cut.get('flow_coefficients', '{}')
            if isinstance(flow_coeffs_str, str):
                flow_coeffs = json.loads(flow_coeffs_str)
            else:
                flow_coeffs = {}
            
            # Parse frequency coefficients
            freq_coeffs_str = cut.get('frequency_coefficients', '{}')
            if isinstance(freq_coeffs_str, str):
                freq_coeffs = json.loads(freq_coeffs_str)
            else:
                freq_coeffs = {}

            # Build constraint expression
            # θ ≥ α₀ + Σ πᵢ y[o,d,IM] + Σ μⱼ f[o,d,IM]
            
            cut_expr = alpha_0
            
            for od_key, pi in flow_coeffs.items():
                parts = od_key.split('_')
                if len(parts) >= 2:
                    o, d = parts[0], '_'.join(parts[1:])  # Handle underscores in names
                    if (o, d) in self.master_model.OD and 'Intermodal' in list(self.master_model.MODES):
                        cut_expr += pi * self.master_model.y[o, d, 'Intermodal']
            
            for od_key, mu in freq_coeffs.items():
                parts = od_key.split('_')
                if len(parts) >= 2:
                    o, d = parts[0], '_'.join(parts[1:])
                    if (o, d) in self.master_model.OD and 'Intermodal' in list(self.master_model.MODES):
                        cut_expr += mu * self.master_model.f[o, d, 'Intermodal']

            # Add cut
            self.master_model.BendersCuts.add(
                self.master_model.theta >= cut_expr
            )
            
            print(f"  ➕ Added optimality cut from iteration {cut.get('iteration', '?')}")

        except Exception as e:
            print(f"⚠️  Failed to add optimality cut: {e}")

    def add_feasibility_cut_to_master(self, cut):
        """Add proper constraint-based feasibility cut to master"""
        if not hasattr(self.master_model, "BendersCuts"):
            print("⚠️ Master model has no BendersCuts ConstraintList")
            return

        try:
            cut_form = cut.get('cut_form', 'unknown')
            
            if cut_form == 'frequency_capacity_ratio':
                # Add proper frequency constraints
                od_constraints_str = cut.get('od_constraints', '[]')
                if isinstance(od_constraints_str, str):
                    od_constraints = json.loads(od_constraints_str)
                else:
                    od_constraints = []
                
                for constraint_info in od_constraints:
                    o = constraint_info['origin']
                    d = constraint_info['destination']
                    min_freq = constraint_info['min_freq']
                    
                    if (o, d) in self.master_model.OD and 'Intermodal' in list(self.master_model.MODES):
                        # f[o,d,IM] ≥ min_freq
                        self.master_model.BendersCuts.add(
                            self.master_model.f[o, d, 'Intermodal'] >= min_freq
                        )
                        print(f"  ➕ Added frequency constraint: f[{o},{d}] ≥ {min_freq:.2f}")
            
            elif cut_form == 'total_flow_reduction':
                # Fallback: total flow reduction
                rhs = float(cut.get('rhs_value', 0.0))
                affected_ods_str = cut.get('affected_ods', '[]')
                
                if isinstance(affected_ods_str, str):
                    affected_ods = json.loads(affected_ods_str)
                else:
                    affected_ods = []
                
                flow_sum = sum(
                    self.master_model.y[o, d, 'Intermodal']
                    for od in affected_ods
                    for o, d in [(od if isinstance(od, tuple) else tuple(od))]
                    if (o, d) in self.master_model.OD and 'Intermodal' in list(self.master_model.MODES)
                )
                
                self.master_model.BendersCuts.add(flow_sum <= rhs)
                print(f"  ➕ Added flow reduction cut: Σy_IM ≤ {rhs:.2f}")

        except Exception as e:
            print(f"⚠️  Failed to add feasibility cut: {e}")

    # ============================================================
    # Extract master solution (unchanged)
    # ============================================================
    def extract_master_solution(self):
        """Extract master solution for subproblem"""
        solution = {"intermodal_flows": {}, "iteration": self.iteration}

        for (o, d) in self.master_model.OD:
            if "Intermodal" not in list(self.master_model.MODES):
                continue

            y_val = float(value(self.master_model.y[o, d, "Intermodal"]))
            f_val = float(value(self.master_model.f[o, d, "Intermodal"]))
            cap_val = float(
                value(self.master_model.cap_per_dep[o, d, "Intermodal"])
                if hasattr(self.master_model, "cap_per_dep")
                else 0.0
            )

            if y_val > 1e-6:
                solution["intermodal_flows"][(o, d)] = {
                    "flow": y_val,
                    "frequency": f_val,
                    "cap_per_dep": cap_val,
                }

        return solution

    # ============================================================
    # Cut storage (unchanged)
    # ============================================================
    def save_benders_cut(self, cut_data):
        """Save cut to Excel file"""
        cut = dict(cut_data)

        # Encode nested structures as JSON
        for key in ["flow_coefficients", "frequency_coefficients", "od_constraints", 
                    "affected_ods", "violated_constraints", "recommendations"]:
            if key in cut and cut[key] is not None:
                try:
                    cut[key] = json.dumps(cut[key])
                except TypeError:
                    cut[key] = str(cut[key])

        try:
            existing = pd.read_excel(self.benders_cuts_file)
            new_cuts = pd.concat([existing, pd.DataFrame([cut])], ignore_index=True)
        except FileNotFoundError:
            new_cuts = pd.DataFrame([cut])

        new_cuts.to_excel(self.benders_cuts_file, index=False)
        print(f"✓ Cut stored to {self.benders_cuts_file}")

    # ============================================================
    # Convergence check (unchanged)
    # ============================================================
    def check_convergence(self):
        """Check if Benders loop has converged"""
        if self.upper_bound == float("inf") or self.lower_bound == float("-inf"):
            return False

        gap = self.upper_bound - self.lower_bound
        relative_gap = abs(gap) / (abs(self.upper_bound) + 1e-10)

        self.convergence_data.append({
            "iteration": self.iteration,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "gap": gap,
            "relative_gap": relative_gap,
            "optimality_cuts": self.optimality_cuts_added,
            "feasibility_cuts": self.feasibility_cuts_added,
        })

        print(f"Convergence check: Gap = {gap:.2f}, RelGap = {relative_gap:.4f}")
        return relative_gap < self.tolerance

    # ============================================================
    # Main loop (unchanged)
    # ============================================================
    def run(self):
        """Main Benders decomposition loop"""
        print("🚀 STARTING BENDERS DECOMPOSITION WITH PROPER DUAL-BASED CUTS")
        print("=" * 60)

        start_time = time.time()

        self.initialize_network()

        for self.iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"BENDERS ITERATION {self.iteration}")
            print(f"{'='*60}")

            try:
                master_solution = self.solve_master_problem()
                subproblem_result = self.solve_subproblem(master_solution)

                if subproblem_result is None:
                    if self.check_convergence():
                        print(f"\n🎉 CONVERGENCE ACHIEVED AT ITERATION {self.iteration}!")
                        break
                    continue

                if self.check_convergence():
                    print(f"\n🎉 CONVERGENCE ACHIEVED AT ITERATION {self.iteration}!")
                    break

                if self.lower_bound > self.upper_bound + 1000:
                    print("⚠️  Bound inversion detected - stopping")
                    break

            except Exception as e:
                print(f"❌ Error in iteration {self.iteration}: {str(e)}")
                import traceback
                traceback.print_exc()
                break

        self.print_final_results(start_time)

    def print_final_results(self, start_time):
        """Print final results"""
        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print("BENDERS DECOMPOSITION COMPLETED")
        print(f"{'='*60}")
        print(f"Total iterations: {self.iteration}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Optimality cuts added: {self.optimality_cuts_added}")
        print(f"Feasibility cuts added: {self.feasibility_cuts_added}")
        print(f"Final lower bound: {self.lower_bound:.2f}")
        print(f"Final upper bound: {self.upper_bound:.2f}")

        if self.upper_bound != float("inf") and self.lower_bound != float("-inf"):
            gap = self.upper_bound - self.lower_bound
            relative_gap = abs(gap) / (abs(self.upper_bound) + 1e-10)
            print(f"Final gap: {gap:.2f}")
            print(f"Final relative gap: {relative_gap:.4f}")

        conv_df = pd.DataFrame(self.convergence_data)
        conv_df.to_excel("model_output/benders_convergence.xlsx", index=False)
        print("Convergence history saved to model_output/benders_convergence.xlsx")


if __name__ == "__main__":
    benders = BendersDecomposition()
    benders.run()