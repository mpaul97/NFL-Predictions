from tkinter import Label
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder

def simplify():

    df = pd.read_csv("%s.csv" % "target")

    simple = []

    for index, row in df.iterrows():
        p = row['points']
        sp = int(p/7)
        simple.append(sp)

    df['simplePoints'] = simple

    df.to_csv("%s.csv" % "target", index=False)

############################

simplify()