import zipfile, pandas as pd, io

paths = [
    ('1617', 'data-pack-2016-17/ProvComm Hip and Knee Replacement 1617.csv.zip'),
    ('1718', 'data-pack-2017-18/ProvComm Hip and Knee Replacements 1718.csv.zip'),
    ('1819', 'data-pack-2018-19/ProvComm Hip and Knee Replacement 1819.csv.zip'),
]

print("=== ProvComm Files ===")
for year, path in paths:
    with zipfile.ZipFile(path) as z:
        csv_names = [n for n in z.namelist() if n.endswith('.csv') and 'MACOSX' not in n]
        df = pd.read_csv(io.BytesIO(z.read(csv_names[0])))
        print(f'\nProvComm {year} columns ({len(df.columns)}):')
        print(list(df.columns))
        knee_prov = df[
            (df['Procedure'].str.contains('Knee', na=False)) &
            (df['Organisation Type'] == 'Provider')
        ]
        measures = knee_prov['Measure'].unique().tolist()
        print(f'  Measures: {measures}')
        print(f'  Total knee+provider rows: {len(knee_prov)}')
        print(knee_prov.head(3).to_string())
        print()

print("\n\n=== Time Series Files ===")
ts_paths = [
    ('1617', 'data-pack-2016-17/Time Series Hip and Knee Replacement 1617.csv.zip'),
    ('1718', 'data-pack-2017-18/Time Series 1718.csv.zip'),
    ('1819', 'data-pack-2018-19/Time Series Hip and Knee Replacement 1819.csv.zip'),
]
for year, path in ts_paths:
    with zipfile.ZipFile(path) as z:
        csv_names = [n for n in z.namelist() if n.endswith('.csv') and 'MACOSX' not in n]
        df = pd.read_csv(io.BytesIO(z.read(csv_names[0])))
        print(f'\nTimeSeries {year} columns ({len(df.columns)}):')
        print(list(df.columns))
        knee = df[df['Procedure'].str.contains('Knee', na=False)]
        print(f'  Measures: {knee["Measure"].unique().tolist() if "Measure" in knee.columns else "n/a"}')
        print(knee.head(3).to_string())

print("\n\n=== PartLink Files ===")
pl_paths = [
    ('1617', 'data-pack-2016-17/PartLink Hip and Knee Replacement 1617.csv.zip'),
    ('1718', 'data-pack-2017-18/PartLink Hip and Knee Replacements 1718.csv.zip'),
    ('1819', 'data-pack-2018-19/PartLink Hip and Knee Replacement 1819.csv.zip'),
]
for year, path in pl_paths:
    with zipfile.ZipFile(path) as z:
        csv_names = [n for n in z.namelist() if n.endswith('.csv') and 'MACOSX' not in n]
        df = pd.read_csv(io.BytesIO(z.read(csv_names[0])))
        print(f'\nPartLink {year} columns ({len(df.columns)}):')
        print(list(df.columns))
        print(df.head(3).to_string())
