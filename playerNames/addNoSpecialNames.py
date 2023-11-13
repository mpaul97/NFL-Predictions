import pandas as pd

df = pd.read_csv("%s.csv" % "playerNames")

for index, row in df.iterrows():
    name = row['name']
    if '-' in name:
        print(name)