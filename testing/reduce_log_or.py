import pandas as pd
import numpy as np
from functools import reduce

cols = ['pid_RUSHER', 'pid_RECEIVER', 'pid_RETURNER', 'pid_PENALIZER', 'pid_FUMBLER']

df = pd.read_csv("%s.csv" % "../playByPlay_v2/data/allTables_pid_entities")
df = df.loc[df['primary_key'].str.contains('202401140dal')]
df.fillna('', inplace=True)

pid = 'DoubRo00'

mask = reduce(np.logical_or, [df[col].str.contains(pid) for col in cols])

df = df.loc[mask, 'primary_key'].values

print(df)