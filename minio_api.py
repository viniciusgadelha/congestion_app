import pandas as pd
import logging
import io
import csv
from minio import Minio

# logging.basicConfig(level=logging.INFO)


def clean_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.reader(fin, delimiter=';')
        writer = csv.writer(fout)

        # Skip the header if present
        next(reader, None)

        writer.writerow(["IdMeasurePoint", "Timestamp", "Imported", "Exported", "R1", "R2", "R3", "R4"])

        for row in reader:
            # Expect 9 columns: Id, Timestamp, Season, Imported, Exported, R1, R2, R3, R4
            if len(row) != 9:
                continue
            Id, Timestamp, Season, Imp, Exp, r1, r2, r3, r4 = row

            def convert_to_float(value):
                # Remove spaces and commas (thousands separators)
                value = value.replace(' ', '').replace(',', '')
                return float(value) / 1000 if value else 0.0

            Imp = convert_to_float(Imp)
            Exp = convert_to_float(Exp)
            r1 = convert_to_float(r1)
            r2 = convert_to_float(r2)
            r3 = convert_to_float(r3)
            r4 = convert_to_float(r4)

            writer.writerow([Id, Timestamp, Imp, Exp, r1, r2, r3, r4])

    return


def merge_profiles(input_file, output_file):

    # Input and output file paths

    # Load the input Excel file (assumes only one sheet)
    input_df = pd.read_csv(input_file)
    conv_table = pd.read_csv('inputs/api_get/CUPS_ID_CONVERSION.csv')

    for i in input_df.IdMeasurePoint:
        try:
            if int(i) in conv_table.MINIO_ID.values:
                pp_id = conv_table.loc[conv_table.MINIO_ID == int(i), 'PANDAPOWER_ID']
                input_df.loc[input_df.IdMeasurePoint == i, 'IdMeasurePoint'] = str(pp_id.values[0])
        except:
            input_df.drop(input_df.loc[input_df.IdMeasurePoint == i].index, inplace=True)

    # Define the mapping of sheet names to data columns
    sheet_map = {
        'Import': 'Imported',
        'Export': 'Exported',
        'r1': 'R1',
        'r2': 'R2',
        'r3': 'R3',
        'r4': 'R4',
    }

    # Prepare pivoted DataFrames
    pivoted_dfs = {}

    for sheet_name, column in sheet_map.items():
        pivot = input_df.pivot_table(
            index='Timestamp',
            columns='IdMeasurePoint',
            values=column,
            aggfunc='first'  # assuming one value per timestamp/point
        ).reset_index()

        pivot.rename(columns={'Timestamp': 'dataLectura'}, inplace=True)
        pivoted_dfs[sheet_name] = pivot

    # Write all transformed sheets to a new Excel file
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        for sheet_name, df in pivoted_dfs.items():
            df.to_excel(writer, index=False, sheet_name=sheet_name)

    print(f"Successfully created: {output_file}")

    return


def minio_get(path='2025_03_29', output_path='inputs/api_get/'):

    MINIO_API_URL = 'minioapi.platon.ari-energy.eu'
    MINIO_ACCESS_KEY = 'wNfrKNFn5Sgn1Cnh5A3c'
    MINIO_SECRET_ACCESS_KEY = 'vY5EYobaernCPd49R71XEfCblZS6ZTxqh0x82RW0'
    MINIO_BUCKET = 'anellupcbucket'
    MODEL_PATH = 'curvas_cups/' + path + '.csv'

    minio_client = Minio(MINIO_API_URL, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_ACCESS_KEY)

    # 2) Get the CSV file from MinIO
    response = minio_client.get_object(MINIO_BUCKET, MODEL_PATH)
    print(f"Response status: {response.status}")

    # 3) Decode its contents and save
    csv_data = response.read().decode('utf-8')
    raw_file = output_path + path + '_raw.csv'
    with open(raw_file, "w", encoding="utf-8") as f:
        f.write(csv_data)

    # 4) Clean the file using your clean_csv function
    cleaned_file = output_path + path + '.csv'
    clean_csv(raw_file, cleaned_file)

    print(f"Cleaned CSV saved to: {cleaned_file}")

    # Convert the clean CSV to a pandas DataFrame

    merge_profiles(cleaned_file, 'inputs/' + path + '.xlsx')

    return



