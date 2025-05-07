import pandas as pd

conversion_file = "CUPS_ID_CONVERSION.csv"
ct_info_file = "all_cups_ids.xlsx"
profiles_file = "2025_02_26_platon_cups.csv"
dir = 'inputs/api_get/'

conversion_df = pd.read_csv(dir + conversion_file)
df_ct = pd.read_excel(dir + ct_info_file)
df_profiles = pd.read_csv(dir + profiles_file)

id_mapping = dict(zip(conversion_df["CUPS_ID"], conversion_df["QGIS_CUPS_ID"]))

df_profiles["ID"] = df_profiles["ID"].map(id_mapping).fillna(df_profiles["ID"])

merged_df = df_profiles.merge(df_ct, left_on="ID", right_on="id", how="left")

columns_to_keep = ["ct", "ID", "time_reading", "season", "active_import", "active_export", "R1", "R2", "R3", "R4", "granularity"]
merged_df = merged_df[columns_to_keep]

for ct in merged_df["ct"].unique():
    ct_data = merged_df[merged_df["ct"] == ct]
    ct_data.to_csv(dir + f" {ct}_{profiles_file}", index=False)

print("CSV files per CT saved successfully.")
