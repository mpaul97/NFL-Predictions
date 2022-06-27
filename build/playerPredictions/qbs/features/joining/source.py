from tokenize import Ignore
import pandas as pd
import numpy as np
import os

DATA_PATH = "../../targets/"

source = pd.read_csv("%s.csv" % "source")
df = pd.read_csv("%s.csv" % (DATA_PATH + "target"))

source.drop(columns=['p_id'], inplace=True, errors='ignore')

pids = []

for index, row in source.iterrows():
    key = row['key']
    pid = df.loc[df['key']==key, 'p_id'].values[0]
    pids.append(pid)

source['p_id'] = pids

source.to_csv("%s.csv" % "source", index=False)