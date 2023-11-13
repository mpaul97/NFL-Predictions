import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelEncoder

def buildEncodedPosition(source: pd.DataFrame, _dir):
    if 'encodedPosition.csv' in os.listdir(_dir):
        print('encodedPosition already created.')
        return
    print('Creating encodedPosition...')
    encoder = LabelEncoder()
    pos_encodings = encoder.fit_transform(source['position'].values)
    pickle.dump(encoder, open((_dir + 'encoder.sav'), 'wb'))
    source['encodedPosition'] = pos_encodings
    source.to_csv("%s.csv" % (_dir + "encodedPosition"), index=False)
    return

def buildNewEncodedPosition(source: pd.DataFrame, _dir):
    print('Creating new encodedPosition...')
    encoder: LabelEncoder = pickle.load(open((_dir + 'encoder.sav'), 'rb'))
    pos_encodings = encoder.transform(source['position'].values)
    source['encodedPosition'] = pos_encodings
    return source

###################

# source = pd.read_csv("%s.csv" % "../source/new_source")

# buildNewEncodedPosition(source, './')