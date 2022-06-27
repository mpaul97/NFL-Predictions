import pandas as pd
import numpy as np
import os

TARGET_PATH = "../../targets/"

df = pd.read_csv("%s.csv" % (TARGET_PATH + "target"))

df.drop(columns=['won', 'points'], inplace=True)

df.to_csv("%s.csv" % "source", index=False)