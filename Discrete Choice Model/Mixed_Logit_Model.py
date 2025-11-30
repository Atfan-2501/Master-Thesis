import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import json
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
NUM_HALTON_DRAWS = 200  # Number of Halton draws for simulation
NUM_TASTE_DRAWS = 100   # Number of taste draws for optimization model

np.random.seed(RANDOM_SEED)

# ============================================================
# 1. LOAD AND PREPARE DATA
# ============================================================

print("Loading data...")
df_wide = pd.read_csv("model_inputs/synthetic_responses_intermodal_CH_100.csv")
core = pd.read_csv("model_inputs/sp_core_design_blocks.csv")
checks = pd.read_csv("model_inputs/sp_checks_design.csv")

sp_cols = [c for c in df_wide.columns if c.startswith("SP_Task")]
df_wide = df_wide.copy()
df_wide["respondent_id"] = np.arange(len(df_wide))

df_choice = df_wide[["respondent_id"] + sp_cols].melt(
    id_vars=["respondent_id"],
    value_name="chosen_alt_label",
    var_name="task_name"
)

task_map = {
    **{f"SP_Task{i}": ("core", i) for i in range(1, 9)},
    "SP_Task9_Dominance": ("check", 1),
    "Synchrotron_Task10_Consistency": ("check", 2)
}
df_choice["task_type"] = df_choice["task_name"].map(lambda x: task_map.get(x, ("core", None))[0])
df_choice["task_in_block"] = df_choice["task_name"].map(lambda x: task_map.get(x, ("core", None))[1])

df_choice = df_choice.merge(df_wide[["respondent_id", "block_id"]], on="respondent_id", how="left")

core2 = core.copy()
core2["is_check"] = 0
checks2 = checks.copy()
checks2["is_check"] = 1
design_all = pd.concat([core2, checks2], ignore_index=True)

df_long = df_choice.merge(
    design_all,
    how="left",
    left_on=["block_id", "task_in_block"],
    right_on=["block_id", "task_in_block"]
)

other_rows = (
    df_long[["respondent_id","block_id","task_in_block","task_type","chosen_alt_label"]]
    .drop_duplicates()
    .assign(
        alt_id="Other", 
        cost_chf_teh=np.nan, 
        time_h=np.nan, 
        dep_per_day=np.nan, 
        is_check=0,
        check_type=""
    )
)

df_long = pd.concat([df_long, other_rows], ignore_index=True, sort=False)

df_long["obs_id"] = df_long["respondent_id"].astype(int)*100 + df_long["task_in_block"]
alt_map = {"Road":1, "Intermodal":2, "Other":3}
df_long["alt_id_num"] = df_long["alt_id"].map(alt_map)

# Fix the "chosen" column calculation
df_long["chosen_alt_label"] = df_long["chosen_alt_label"].fillna("").astype(str).str.strip().str.lower()
df_long["alt_id_lower"] = df_long["alt_id"].astype(str).str.strip().str.lower()
df_long["chosen"] = (df_long["chosen_alt_label"] == df_long["alt_id_lower"]).astype(int)

# Filter to core tasks only
est = df_long[df_long["task_type"]=="core"].copy()

# Clean up missing values
for col in ["cost_chf_teh","time_h","dep_per_day"]:
    est[col] = np.where(est["alt_id"].isin(["Road","Intermodal"]), est[col], 0.0)

# Create analysis variables
est["cost_100"] = est["cost_chf_teh"] / 100.0
est["time_h"] = est["time_h"].astype(float)
est["freq"] = est["dep_per_day"].astype(float)
est["ASC_Road"] = (est["alt_id"]=="Road").astype(int)
est["ASC_Other"] = (est["alt_id"]=="Other").astype(int)  # FIXED: was "asot()"

est = est.dropna(subset=["alt_id_num"])

# Validation
print(f"Prepared {len(est)} rows, {est['obs_id'].nunique()} obs, {est['respondent_id'].nunique()} resp")

# Validate that each observation has exactly one chosen alternative
choice_counts = est.groupby("obs_id")["chosen"].sum()
if not all(choice_counts == 1):
    print(f"WARNING: {sum(choice_counts != 1)} observations do not have exactly 1 chosen alt")
    print(choice_counts[choice_counts != 1].head(10))
else:
    print("✓ Data validation passed: all observations have exactly 1 chosen alternative")

# ============================================================
# 2. HALTON SEQUENCE GENERATION
# ============================================================

def generate_halton_sequence(size, dim, skip=0):
    """Generate Halton sequence for simulation"""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    
    def halton_1d(index, base):
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f = f / base
        return result
    
    sequence = np.zeros((size, dim))
    for d in range(dim):
        for s in range(size):
            sequence[s, d] = halton_1d(s + skip, primes[d])
    
    return sequence

# ============================================================
# 3. MIXED LOGIT ESTIMATION
# ============================================================

class MixedLogitModel:
    def __init__(self, data, n_draws=200):
        self.data = data
        self.n_draws = n_draws
        
        # Generate Halton draws once
        # For 3 random parameters: cost, time, freq
        self.halton_draws = generate_halton_sequence(n_draws, 3, skip=100)
        # Convert uniform to standard normal
        self.normal_draws = norm.ppf(self.halton_draws)
        
        # Data preparation
        self.n_obs = data['obs_id'].nunique()
        self.n_resp = data['respondent_id'].nunique()
        
    def compute_utility(self, params, data, draws=None):
        """
        Compute utility for each alternative
        
        params: [μ_cost, σ_cost, μ_time, σ_time, μ_freq, σ_freq, ASC_Road, ASC_Other]
        """
        mu_cost, sigma_cost, mu_time, sigma_time, mu_freq, sigma_freq, asc_road, asc_other = params
        
        # Base utility (no draws)
        util = np.zeros(len(data))
        
        if draws is None:
            # Use mean values only (for initial MNL-like evaluation)
            util += data['cost_100'] * mu_cost
            util += data['time_h'] * mu_time
            util += data['freq'] * mu_freq
        else:
            # Use specific draw
            beta_cost = mu_cost + sigma_cost * draws[0]
            beta_time = mu_time + sigma_time * draws[1]
            beta_freq = mu_freq + sigma_freq * draws[2]
            
            util += data['cost_100'] * beta_cost
            util += data['time_h'] * beta_time
            util += data['freq'] * beta_freq
        
        # Add alternative-specific constants
        util += data['ASC_Road'] * asc_road
        util += data['ASC_Other'] * asc_other
        
        return util
    
    def simulate_choice_probability(self, params, obs_data):
        """Simulate choice probability for one observation using Halton draws"""
        
        # For each draw, compute probability
        choice_probs = np.zeros(len(obs_data))
        
        for draw_idx in range(self.n_draws):
            draw = self.normal_draws[draw_idx, :]
            
            # Compute utilities for all alternatives
            utils = self.compute_utility(params, obs_data, draws=draw)
            
            # Compute choice probabilities
            exp_utils = np.exp(utils - np.max(utils))  # Numerical stability
            probs = exp_utils / np.sum(exp_utils)
            
            choice_probs += probs
        
        # Average over draws
        choice_probs /= self.n_draws
        
        return choice_probs
    
    def log_likelihood(self, params):
        """Compute simulated log-likelihood"""
        
        # Ensure standard deviations are positive
        params[1] = np.abs(params[1])  # sigma_cost
        params[3] = np.abs(params[3])  # sigma_time
        params[5] = np.abs(params[5])  # sigma_freq
        
        ll = 0.0
        
        # Loop over observations
        for obs_id in self.data['obs_id'].unique():
            obs_data = self.data[self.data['obs_id'] == obs_id].copy()
            
            # Simulate choice probabilities
            probs = self.simulate_choice_probability(params, obs_data)
            
            # Get chosen alternative - find which row has chosen == 1
            chosen_rows = obs_data[obs_data['chosen'] == 1]
            
            if len(chosen_rows) == 0:
                print(f"Warning: No chosen alternative for obs_id {obs_id}")
                continue
            
            # Get the position of chosen alternative in the obs_data subset
            # Find the index position (0, 1, or 2) of the chosen alternative
            chosen_alt_idx = chosen_rows.index[0]
            chosen_alt_pos = obs_data.index.get_loc(chosen_alt_idx)
            
            # Ensure probs is a numpy array
            if isinstance(probs, pd.Series):
                probs = probs.values
            
            # Get probability of chosen alternative
            prob_chosen = np.maximum(probs[chosen_alt_pos], 1e-10)  # Avoid log(0)
            ll += np.log(prob_chosen)
        
        return -ll  # Negative for minimization
    
    
    def estimate(self, initial_params=None):
        """Estimate mixed logit model"""
        
        if initial_params is None:
            # Initial values from simple MNL or reasonable guesses
            initial_params = np.array([
                -0.2, 0.1,   # μ_cost, σ_cost
                -0.05, 0.02, # μ_time, σ_time
                -0.15, 0.05, # μ_freq, σ_freq
                -0.3, -4.0   # ASC_Road, ASC_Other
            ])
        
        print("\nEstimating Mixed Logit Model...")
        print("Initial parameters:", initial_params)
        
        # Optimize
        result = minimize(
            self.log_likelihood,
            initial_params,
            method='BFGS',
            options={'disp': True, 'maxiter': 200}
        )
        
        if not result.success:
            print("Warning: Optimization did not converge")
        
        # Ensure positive standard deviations
        result.x[1] = np.abs(result.x[1])
        result.x[3] = np.abs(result.x[3])
        result.x[5] = np.abs(result.x[5])
        
        return result

# ============================================================
# 4. ESTIMATION
# ============================================================

model = MixedLogitModel(est, n_draws=NUM_HALTON_DRAWS)
results = model.estimate()

# Extract parameters
mu_cost, sigma_cost = results.x[0], results.x[1]
mu_time, sigma_time = results.x[2], results.x[3]
mu_freq, sigma_freq = results.x[4], results.x[5]
asc_road, asc_other = results.x[6], results.x[7]

print("\n" + "="*60)
print("MIXED LOGIT ESTIMATION RESULTS")
print("="*60)
print(f"\nCost (per 100 CHF):")
print(f"  Mean (μ):     {mu_cost:.4f}")
print(f"  Std Dev (σ):  {sigma_cost:.4f}")
print(f"\nTime (per hour):")
print(f"  Mean (μ):     {mu_time:.4f}")
print(f"  Std Dev (σ):  {sigma_time:.4f}")
print(f"\nFrequency (per dep/week):")
print(f"  Mean (μ):     {mu_freq:.4f}")
print(f"  Std Dev (σ):  {sigma_freq:.4f}")
print(f"\nAlternative-Specific Constants:")
print(f"  ASC Road:     {asc_road:.4f}")
print(f"  ASC Other:    {asc_other:.4f}")
print(f"\nLog-Likelihood: {-results.fun:.2f}")

# ============================================================
# 5. GENERATE TASTE DRAWS FOR OPTIMIZATION
# ============================================================

print(f"\nGenerating {NUM_TASTE_DRAWS} taste draws for optimization model...")

# Generate new Halton sequence for optimization (different from estimation)
taste_halton = generate_halton_sequence(NUM_TASTE_DRAWS, 3, skip=1000)
taste_normal = norm.ppf(taste_halton)

# Generate taste draws
taste_draws_data = []

for draw_id in range(1, NUM_TASTE_DRAWS + 1):
    # Get draw from Halton sequence
    draw_idx = draw_id - 1
    
    # Generate random coefficients from estimated distributions
    beta_cost_draw = mu_cost + sigma_cost * taste_normal[draw_idx, 0]
    beta_time_draw = mu_time + sigma_time * taste_normal[draw_idx, 1]
    beta_freq_draw = mu_freq + sigma_freq * taste_normal[draw_idx, 2]
    
    # Add draws for both Road and Intermodal (same coefficients, different ASCs)
    for mode in ["Road", "Intermodal"]:
        asc_value = asc_road if mode == "Road" else 0.0  # Intermodal is base
        
        taste_draws_data.append({
            "draw_id": draw_id,
            "mode_id": mode,
            "ASC_r": asc_value,
            "beta_cost_r_per_100CHF": beta_cost_draw,
            "beta_time_r_per_hour": beta_time_draw,
            "beta_freq_r_per_departure": beta_freq_draw,
            "dist_cost": "N({:.4f}, {:.4f})".format(mu_cost, sigma_cost),
            "dist_time": "N({:.4f}, {:.4f})".format(mu_time, sigma_time),
            "dist_freq": "N({:.4f}, {:.4f})".format(mu_freq, sigma_freq)
        })

taste_draws_df = pd.DataFrame(taste_draws_data)

# ============================================================
# 6. SAVE RESULTS
# ============================================================

import os
OUTDIR = "model_outputs"
os.makedirs(OUTDIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1. Save parameter estimates
params_df = pd.DataFrame({
    "parameter": [
        "cost_mean", "cost_std",
        "time_mean", "time_std",
        "freq_mean", "freq_std",
        "ASC_Road", "ASC_Other"
    ],
    "value": [
        mu_cost, sigma_cost,
        mu_time, sigma_time,
        mu_freq, sigma_freq,
        asc_road, asc_other
    ]
})
params_df.to_csv(os.path.join(OUTDIR, f"mixed_logit_parameters_{stamp}.csv"), index=False)

# 2. Save taste draws
taste_draws_df.to_csv(os.path.join(OUTDIR, f"taste_draws_{stamp}.csv"), index=False)

# 3. Compute WTP measures
wtp = {
    "time_h__CHF_per_hour": - (mu_time / mu_cost) * 100.0,
    "freq__CHF_per_departure": - (mu_freq / mu_cost) * 100.0,
}
wtp_path = os.path.join(OUTDIR, f"mixed_logit_wtp_{stamp}.json")
with open(wtp_path, "w", encoding="utf-8") as f:
    json.dump(wtp, f, indent=2)

print(f"\nSaved results to {OUTDIR}:")
print(f"  - Parameters: mixed_logit_parameters_{stamp}.csv")
print(f"  - Taste draws: taste_draws_{stamp}.csv")
print(f"  - WTP: mixed_logit_wtp_{stamp}.json")

# 4. Save for MILP input (replace existing taste draws in master input file)
print("\n" + "="*60)
print("IMPORTANT: Update the Taste_Draws sheet in")
print("'master_problem_inputs_with_taste_draws.xlsx'")
print("with the generated taste_draws CSV file")
print("="*60)


# 3a. Save human–readable summary for the mixed logit estimation
summary_path = os.path.join(OUTDIR, f"mixed_logit_summary_{stamp}.txt")

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("MIXED LOGIT MODEL ESTIMATION SUMMARY\n")
    f.write("=" * 60 + "\n\n")

    # Basic info
    f.write(f"Timestamp:            {stamp}\n")
    f.write(f"Number of respondents: {model.n_resp}\n")
    f.write(f"Number of observations: {model.n_obs}\n")
    f.write(f"Number of Halton draws (estimation): {model.n_draws}\n\n")

    # Log-likelihood and optimisation status
    f.write(f"Log-Likelihood at optimum: {-results.fun:.3f}\n")
    f.write(f"Converged: {results.success}\n")
    f.write(f"Message:   {results.message}\n\n")

    f.write("Parameter estimates (means and standard deviations)\n")
    f.write("-" * 60 + "\n")
    f.write(f"Cost (per 100 CHF):   mu = {mu_cost: .4f},  sigma = {sigma_cost: .4f}\n")
    f.write(f"Time (per hour):      mu = {mu_time: .4f},  sigma = {sigma_time: .4f}\n")
    f.write(f"Frequency (per dep):  mu = {mu_freq: .4f},  sigma = {sigma_freq: .4f}\n")
    f.write(f"ASC Road:             {asc_road: .4f}\n")
    f.write(f"ASC Other:            {asc_other: .4f}\n\n")

    f.write("Implied willingness-to-pay (WTP)\n")
    f.write("-" * 60 + "\n")
    f.write(f"Time:   {wtp['time_h__CHF_per_hour']: .2f} CHF/hour\n")
    f.write(f"Freq.:  {wtp['freq__CHF_per_departure']: .2f} CHF per additional departure\n\n")

    f.write("Notes:\n")
    f.write(" - Random coefficients for cost, time and frequency are assumed normal.\n")
    f.write(" - Halton sequences are used for simulation of random taste draws.\n")
    f.write(" - Taste draws for the MILP are saved separately in the CSV file.\n")

print(f"\nSummary written to: {summary_path}")

