import pandas as pd

def make_interfaces(service_plan_path, flows_out, freq_out):
    df = pd.read_csv(service_plan_path)
    intermodal = df[df["mode"]=="Intermodal"].copy()
    intermodal[["od_id","TEU"]].to_csv(flows_out, index=False)
    intermodal[["od_id","freq"]].to_csv(freq_out, index=False)

if __name__ == "__main__":
    make_interfaces("service_plan.csv", "intermodal_flows_from_master.csv", "freq_from_master.csv")