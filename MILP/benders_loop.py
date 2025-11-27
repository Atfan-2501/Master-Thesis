import pandas as pd
import numpy as np
import json
from pyomo.environ import value
import subprocess
import sys
import time
from datetime import datetime


class BendersDecomposition:
    def __init__(self):
        self.master_model = None
        self.iteration = 0
        self.max_iterations = 50
        self.tolerance = 1e-4
        self.upper_bound = float("inf")
        self.lower_bound = float("-inf")
        self.convergence_data = []

        # Files
        self.master_sol_file = "model_output/master_solution_decisions.xlsx"
        self.ts_input_file = "model_output/TS_Subproblem_Inputs_extended.xlsx"
        self.benders_cuts_file = "model_output/benders_cuts.xlsx"

    # ============================================================
    # 0) Network generation
    # ============================================================
    def initialize_network(self):
        """Regenerate the time-space network (connections + TS nodes/arcs)."""
        print("=== INITIALIZING TIME-SPACE NETWORK ===")

        # Regenerate connections
        result = subprocess.run(
            [sys.executable, "generate_connections.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to generate connections: {result.stderr}"
            )
        print("✅ Connections generated")

        # Regenerate TS network
        result = subprocess.run(
            [sys.executable, "generate_time_space_network.py"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to generate TS network: {result.stderr}"
            )
        print("✅ Time-space network generated")

    # ============================================================
    # 1) Master problem
    # ============================================================
    def solve_master_problem(self, add_cuts=True):
        """Solve the master problem with optional Benders cuts."""
        print(f"\n=== SOLVING MASTER PROBLEM (Iteration {self.iteration}) ===")

        from master_milp import build_master_problem, solve_master

        # Build master problem
        self.master_model = build_master_problem()

        # Add Benders cuts from previous iterations
        if add_cuts and self.iteration > 1:
            self.add_benders_cuts_to_master()

        # Solve master problem
        master_result = solve_master(self.master_model)
        if master_result is None:
            raise RuntimeError("Master problem failed to solve")

        # Extract solution for TS
        master_solution = self.extract_master_solution()

        # Update lower bound (relaxation value)
        obj_value = value(self.master_model.OBJ)
        self.lower_bound = max(self.lower_bound, obj_value)

        print(f"Master objective: {obj_value:.2f}")
        print(
            f"Current bounds: LB={self.lower_bound:.2f}, "
            f"UB={self.upper_bound:.2f}"
        )

        # Also dump the master solution to Excel if you want to inspect
        df = self.master_solution_to_df()
        df.to_excel(self.master_sol_file, index=False)
        print(f"✓ Master solution written to {self.master_sol_file}")

        return master_solution

    def master_solution_to_df(self):
        """Helper: pack master x,f,p,y into a DataFrame for debugging/export."""
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
    # 2) Subproblem
    # ============================================================
    def solve_subproblem(self, master_solution):
        """Solve the time-space subproblem given master decisions."""
        print(f"\n=== SOLVING SUBPROBLEM (Iteration {self.iteration}) ===")

        from ts_subproblem import solve_ts_subproblem

        # Solve TS with fixed master decisions (frequency, TEU)
        subproblem_result = solve_ts_subproblem(master_solution)

        if subproblem_result is None:
            print("❌ Subproblem infeasible - generating feasibility cut")
            self.generate_feasibility_cut(master_solution)
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

        # True upper bound = master_obj + (actual_ts_cost - theta)
        true_obj = master_obj + (sp_obj - theta_value)
        self.upper_bound = min(self.upper_bound, true_obj)

        print(f"Subproblem cost: {sp_obj:.2f}")
        print(f"True objective: {true_obj:.2f}")
        print(
            f"Current bounds: LB={self.lower_bound:.2f}, "
            f"UB={self.upper_bound:.2f}"
        )

        # Generate optimality cut from this TS solution
        self.generate_optimality_cut(master_solution, duals, sp_obj)

        return sp_obj, duals

    # ============================================================
    # 3) Cut generation
    # ============================================================
    def generate_feasibility_cut(self, master_solution):
        """
        Generate a heuristic feasibility cut when the subproblem is infeasible.

        We mark ODs with very high utilization and later add a cut that
        reduces the *total* intermodal flow on those ODs by 10%.
        """
        print("Generating feasibility cut...")

        cut_data = {
            "iteration": self.iteration,
            "cut_type": "feasibility",
            "timestamp": datetime.now().isoformat(),
        }

        problematic_ods = []
        total_flow = 0.0

        for od, info in master_solution["intermodal_flows"].items():
            o, d = od
            flow = info["flow"]
            freq = info["frequency"]
            cap_per_dep = info["cap_per_dep"]
            if freq > 0 and cap_per_dep > 0:
                utilization = flow / (freq * cap_per_dep)
            else:
                utilization = 0.0

            if utilization > 0.95:
                problematic_ods.append([o, d])
                total_flow += flow

        cut_data["problematic_ods"] = problematic_ods
        # Reduce total intermodal flow on these ODs by 10%
        cut_data["rhs"] = 0.9 * total_flow

        self.save_benders_cut(cut_data)

    def generate_optimality_cut(self, master_solution, duals, sp_cost):
        """
        Generate an (initially simplified) optimality cut from subproblem duals.

        For now:
            theta >= sp_cost * 0.95
        Later you can replace this with a dual-based affine function
        of y and f.
        """
        print("Generating optimality cut...")

        cut_data = {
            "iteration": self.iteration,
            "cut_type": "optimality",
            "subproblem_cost": float(sp_cost),
            "duals": duals,
            "timestamp": datetime.now().isoformat(),
        }

        self.save_benders_cut(cut_data)

    # ============================================================
    # 4) Add cuts back to master
    # ============================================================
    def add_benders_cuts_to_master(self):
        """Read all stored cuts and add them to the current master model."""
        try:
            cuts_df = pd.read_excel(self.benders_cuts_file)
        except FileNotFoundError:
            print("No previous Benders cuts found")
            return

        feasibility_cuts = cuts_df[cuts_df["cut_type"] == "feasibility"]
        optimality_cuts = cuts_df[cuts_df["cut_type"] == "optimality"]

        print(
            f"Adding {len(feasibility_cuts)} feasibility cuts and "
            f"{len(optimality_cuts)} optimality cuts"
        )

        # Feasibility cuts
        for _, cut in feasibility_cuts.iterrows():
            raw_ods = cut.get("problematic_ods", None)
            rhs_val = cut.get("rhs", None)

            if isinstance(raw_ods, str) and raw_ods.strip() and rhs_val is not None:
                try:
                    ods_list = json.loads(raw_ods)
                    problematic_ods = [tuple(od) for od in ods_list]
                    rhs = float(rhs_val)
                    self.add_simplified_feasibility_cut(problematic_ods, rhs)
                except Exception as e:
                    print(f"⚠️ Failed to parse feasibility cut row: {e}")

        # Optimality cuts
        for _, cut in optimality_cuts.iterrows():
            sp_cost = cut.get("subproblem_cost", None)
            if sp_cost is not None and not pd.isna(sp_cost):
                self.add_simplified_optimality_cut(float(sp_cost))

    def add_simplified_feasibility_cut(self, problematic_ods, rhs):
        """
        Add a simple feasibility cut to the master:

            sum_{(o,d) in problematic_ods} y[o,d,Intermodal] <= rhs

        where rhs is typically 0.9 * current total flow on those ODs.
        """
        if not hasattr(self.master_model, "BendersCuts"):
            print("⚠️ Master model has no BendersCuts ConstraintList")
            return

        expr = sum(
            self.master_model.y[o, d, "Intermodal"] for (o, d) in problematic_ods
        ) <= rhs

        self.master_model.BendersCuts.add(expr)
        print(
            f"  ➕ Added feasibility cut: "
            f"sum y_IM(problematic_ods) <= {rhs:.2f}"
        )

    def add_simplified_optimality_cut(self, sp_cost):
        """
        Add a simple optimality cut:

            theta >= 0.95 * sp_cost

        This is a valid (if weak) lower bound on TS cost.
        """
        if not hasattr(self.master_model, "BendersCuts"):
            print("⚠️ Master model has no BendersCuts ConstraintList")
            return

        expr = self.master_model.theta >= 0.95 * sp_cost
        self.master_model.BendersCuts.add(expr)
        print(
            f"  ➕ Added optimality cut: theta >= {0.95 * sp_cost:.2f}"
        )

    # ============================================================
    # 5) Extract master solution
    # ============================================================
    def extract_master_solution(self):
        """
        Build the dictionary passed to ts_subproblem.solve_ts_subproblem.

        Structure:
        master_solution = {
            'iteration': k,
            'intermodal_flows': {
                (o,d): {'flow': y_IM, 'frequency': f_IM, 'cap_per_dep': cap_per_dep_IM}
            }
        }
        """
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
    # 6) Cut storage (Excel + JSON)
    # ============================================================
    def save_benders_cut(self, cut_data):
        """Append a cut to the Excel file, JSON-encoding nested structures."""
        cut = dict(cut_data)  # shallow copy

        # Encode nested structures as JSON strings for safe storage
        for key in ["problematic_ods", "duals"]:
            if key in cut and cut[key] is not None:
                try:
                    cut[key] = json.dumps(cut[key])
                except TypeError:
                    # As a fallback, cast to string
                    cut[key] = str(cut[key])

        try:
            existing = pd.read_excel(self.benders_cuts_file)
            new_cuts = pd.concat(
                [existing, pd.DataFrame([cut])], ignore_index=True
            )
        except FileNotFoundError:
            new_cuts = pd.DataFrame([cut])

        new_cuts.to_excel(self.benders_cuts_file, index=False)
        print(f"✓ Cut stored to {self.benders_cuts_file}")

    # ============================================================
    # 7) Convergence check
    # ============================================================
    def check_convergence(self):
        """Check if Benders loop has converged based on LB/UB gap."""
        if self.upper_bound == float("inf") or self.lower_bound == float("-inf"):
            return False

        gap = self.upper_bound - self.lower_bound
        relative_gap = abs(gap) / (abs(self.upper_bound) + 1e-10)

        self.convergence_data.append(
            {
                "iteration": self.iteration,
                "lower_bound": self.lower_bound,
                "upper_bound": self.upper_bound,
                "gap": gap,
                "relative_gap": relative_gap,
            }
        )

        print(f"Convergence check: Gap = {gap:.2f}, RelGap = {relative_gap:.4f}")
        return relative_gap < self.tolerance

    # ============================================================
    # 8) Main loop
    # ============================================================
    def run(self):
        """Main Benders decomposition loop."""
        print("🚀 STARTING BENDERS DECOMPOSITION")
        print("=" * 60)

        start_time = time.time()

        # 1) Build network once (or each iteration if you want dynamic TS)
        self.initialize_network()

        # 2) Main iteration loop
        for self.iteration in range(1, self.max_iterations + 1):
            print(f"\n{'='*60}")
            print(f"BENDERS ITERATION {self.iteration}")
            print(f"{'='*60}")

            try:
                # Master
                master_solution = self.solve_master_problem()

                # Subproblem
                subproblem_result = self.solve_subproblem(master_solution)

                # If TS infeasible, we generated a feasibility cut and skip UB update
                if subproblem_result is None:
                    # Still check convergence (might not make sense yet but harmless)
                    if self.check_convergence():
                        print(
                            f"\n🎉 CONVERGENCE ACHIEVED AT ITERATION {self.iteration}!"
                        )
                        break
                    continue

                # Convergence
                if self.check_convergence():
                    print(
                        f"\n🎉 CONVERGENCE ACHIEVED AT ITERATION {self.iteration}!"
                    )
                    break

                # Safety check: inverted bounds
                if self.lower_bound > self.upper_bound + 1000:
                    print("⚠️  Bound inversion detected - stopping")
                    break

            except Exception as e:
                print(f"❌ Error in iteration {self.iteration}: {str(e)}")
                import traceback

                traceback.print_exc()
                break

        self.print_final_results(start_time)

    # ============================================================
    # 9) Final report
    # ============================================================
    def print_final_results(self, start_time):
        total_time = time.time() - start_time

        print(f"\n{'='*60}")
        print("BENDERS DECOMPOSITION COMPLETED")
        print(f"{'='*60}")
        print(f"Total iterations: {self.iteration}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Final lower bound: {self.lower_bound:.2f}")
        print(f"Final upper bound: {self.upper_bound:.2f}")

        if (
            self.upper_bound != float("inf")
            and self.lower_bound != float("-inf")
        ):
            gap = self.upper_bound - self.lower_bound
            relative_gap = abs(gap) / (abs(self.upper_bound) + 1e-10)
            print(f"Final gap: {gap:.2f}")
            print(f"Final relative gap: {relative_gap:.4f}")

        # Save convergence history
        conv_df = pd.DataFrame(self.convergence_data)
        conv_df.to_excel("model_output/benders_convergence.xlsx", index=False)
        print(
            "Convergence history saved to "
            "model_output/benders_convergence.xlsx"
        )


if __name__ == "__main__":
    benders = BendersDecomposition()
    benders.run()
