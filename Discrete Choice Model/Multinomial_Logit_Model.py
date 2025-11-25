import pandas as pd
import numpy as np
import pylogit as pl
from collections import OrderedDict

print("Loading data...")
df_wide = pd.read_csv("synthetic_responses_intermodal_CH_100.csv")
core = pd.read_csv("sp_core_design_blocks.csv")
checks = pd.read_csv("sp_checks_design.csv")

sp_cols = [c for c in df_wide.columns if c.startswith("SP_Task")]
df_wide = df_wide.copy()
df_wide["respondent_id"] = np.arange(len(df_wide))

df_choice = df_wide[["respondent_id"] + sp_cols].melt(
    id_vars=["respondent_id"],
    var_name="task_name",
    value_name="chosen_alt_label"
)

task_map = {
    **{f"SP_Task{i}": ("core", i) for i in range(1, 9)},
    "SP_Task9_Dominance": ("check", 1),
    "SP_Task10_Consistency": ("check", 2)
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
    .assign(alt_id="Other", cost_chf_teu=np.nan, time_h=np.nan, ontime_p=np.nan,
            dep_per_day=np.nan, is_check=lambda x: np.where(x["task_type"]=="check",1,0),
            check_type=np.where(lambda: False, "", ""))
)

df_long = pd.concat([df_long, other_rows], ignore_index=True, sort=False)

df_long["obs_id"] = df_long["respondent_id"].astype(int)*100 + df_long["task_in_block"].astype(int)
alt_map = {"Road":1, "Intermodal":2, "Other":3}
df_long["alt_id_num"] = df_long["alt_id"].map(alt_map)
df_long["chosen"] = (df_long["chosen_alt_label"].str.strip().str.lower()
                     == df_long["alt_id"].astype(str).str.strip().str.lower()).astype(int)

est = df_long[df_long["task_type"]=="core"].copy()

for col in ["cost_chf_teu","time_h","ontime_p","dep_per_day"]:
    est[col] = np.where(est["alt_id"].isin(["Road","Intermodal"]), est[col], 0.0)

est["cost_100"] = est["cost_chf_teu"] / 100.0
est["time_h"]   = est["time_h"].astype(float)
est["reliab"]   = est["ontime_p"].astype(float) / 100.0
est["freq"]     = est["dep_per_day"].astype(float)
est["ASC_Road"]  = (est["alt_id"]=="Road").astype(int)
est["ASC_Other"] = (est["alt_id"]=="Other").astype(int)

est = est.dropna(subset=["alt_id_num"])

print(f"Prepared {len(est)} rows, {est['obs_id'].nunique()} obs, {est['respondent_id'].nunique()} resp")

spec_names = OrderedDict()
spec_names["ASC_Road"]  = ["ASC_Road"]
spec_names["ASC_Other"] = ["ASC_Other"]
spec_names["cost_100"]  = ["cost_100"]
spec_names["time_h"]    = ["time_h"]
spec_names["reliab"]    = ["reliab"]
spec_names["freq"]      = ["freq"]

spec = OrderedDict()
spec["ASC_Road"]  = [1]
spec["ASC_Other"] = [3]
spec["cost_100"]  = [[1,2]]
spec["time_h"]    = [[1,2]]
spec["reliab"]    = [[1,2]]
spec["freq"]      = [[1,2]]

if "alt_id" in est.columns and "alt_id_num" in est.columns:
    est = est.rename(columns={"alt_id": "alt_label", "alt_id_num": "alt_id"})

est["alt_id"] = est["alt_id"].astype("int64")
est = est.loc[:, ~est.columns.duplicated(keep="first")]
est = est.sort_values(["obs_id", "alt_id"]).reset_index(drop=True)

sizes = est.groupby("obs_id").size()
if not sizes.eq(3).all():
    print(f"ERROR: {(~sizes.eq(3)).sum()} obs don't have 3 alts")
    print(sizes[~sizes.eq(3)].head())
    raise AssertionError("Each obs must have exactly 3 alternatives")

choice_counts = est.groupby("obs_id")["chosen"].sum()
if not choice_counts.eq(1).all():
    print(f"\nERROR: {(~choice_counts.eq(1)).sum()} obs don't have exactly 1 choice")
    print("\nChoice counts distribution:")
    print(choice_counts.value_counts().sort_index())
    print("\nExample obs with issues:")
    bad_obs = choice_counts[~choice_counts.eq(1)].index[:5]
    print(est[est["obs_id"].isin(bad_obs)][["obs_id","alt_label","chosen_alt_label","chosen"]].sort_values("obs_id"))
    raise AssertionError("Each obs must have exactly 1 chosen alt")

mnl = pl.create_choice_model(
    data=est,
    alt_id_col="alt_id",
    obs_id_col="obs_id",
    choice_col="chosen",
    specification=spec,
    model_type="MNL",
    names=spec_names
)

print("\nEstimating 3-alt MNL...")
mnl.fit_mle(np.zeros(len(spec)))


# --- Save model outputs: params, summary, WTP, and pickle the model ---

import os, json, pickle, pandas as pd
from datetime import datetime

OUTDIR = "model_outputs"
os.makedirs(OUTDIR, exist_ok=True)
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 1) Save pretty text summary
summary = mnl.get_statsmodels_summary()
with open(os.path.join(OUTDIR, f"mnl_summary_{stamp}.txt"), "w", encoding="utf-8") as f:
    f.write(summary.as_text())

# 2) Extract the coefficient table robustly from the Summary
#    (works across different pylogit/statsmodels versions)
coef_df = None
try:
    coef_df = pd.read_html(summary.tables[1].as_html(), header=0, index_col=0)[0]
except Exception:
    # Fallback: try to coerce via string
    coef_df = pd.read_html(str(summary.tables[1]), header=0, index_col=0)[0]

# Normalize column names (lowercase) for easy access
coef_df.columns = [c.strip().lower() for c in coef_df.columns]
coef_df.index.name = "parameter"

# 3) Save raw params table
coef_csv = os.path.join(OUTDIR, f"mnl_parameters_{stamp}.csv")
coef_df.to_csv(coef_csv, float_format="%.6g")
print(f"\nSaved parameters to: {coef_csv}")

# 4) Compute WTP
#    Remember: cost_100 is cost per 100 CHF, so:
#      WTP(attr) in CHF/unit = - (beta_attr / beta_cost_100) * 100
#    For reliability, 'reliab' is 0..1, so CHF per 1%-pt = WTP(0..1)/100.
def get_coef(name):
    # helper that tolerates slight name variations (e.g., spaces)
    idx = [i for i in coef_df.index if i.strip().lower() == name]
    if not idx:
        raise KeyError(f"Parameter '{name}' not found in estimated table. "
                       f"Available: {list(coef_df.index)}")
    col = "coef" if "coef" in coef_df.columns else "coef."
    return float(coef_df.loc[idx[0], col])

beta_cost = get_coef("cost_100")
beta_time = get_coef("time_h")
beta_rel  = get_coef("reliab")
beta_freq = get_coef("freq")

wtp = {
    "time_h__CHF_per_hour":        - (beta_time / beta_cost) * 100.0,
    "reliab__CHF_per_pct_point":   - (beta_rel  / beta_cost),         # per +1%-pt
    "reliab__CHF_per_unit_0to1":   - (beta_rel  / beta_cost) * 100.0, # per +100%-pts
    "freq__CHF_per_departure_day": - (beta_freq / beta_cost) * 100.0,
}

wtp_path = os.path.join(OUTDIR, f"mnl_wtp_{stamp}.json")
with open(wtp_path, "w", encoding="utf-8") as f:
    json.dump(wtp, f, indent=2)
print(f"Saved WTP to: {wtp_path}")

