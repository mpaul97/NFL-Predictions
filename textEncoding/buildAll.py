import sys
sys.path.append("../")
import warnings

import pandas as pd
from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import os
from random import randrange
import time
import regex as re
from ordered_set import OrderedSet

from nltk import tokenize
from sklearn.feature_extraction.text import CountVectorizer

from paths import DATA_PATH

from Converter import Converter

def getUsedKeys(c: Converter):
    df_list = []
    for fn in os.listdir(c.base_dir + c.sub_dir):
        if fn != c.fn:
            new_dir = c.base_dir + c.sub_dir + fn + '/'
            df_list.append(pd.read_csv("%s.csv" % (new_dir + "info_" + fn)))
    df = pd.concat(df_list)
    return list(set(df['key'].values))

def getAllKeys(c: Converter):
    df = pd.read_csv("%s.csv" % (c.base_dir + 'data/rawTrain'))
    return list(OrderedSet(df['key'].values))

def getTargets(c: Converter):
    df = pd.read_csv("%s.csv" % (c.base_dir + 'data/targetNames'))
    return list(df['name'].values)

def transformAll(type):
    
    c = Converter(fn='all', type=type)

    c.setTrain()

    if __name__ == '__main__':
        c.setInfoParallel()
        c.vectorize()
        
    return

def transformTrain(fn, type):
    
    c = Converter(fn, type)

    # fill if new train data
    # c.test_keys = ['202301080clt']

    # c.setTrain()
    c.setInfo()
    c.vectorize()
    
    return

def newTrain(fn, type):
    
    c = Converter(fn, type)

    # fill if new train data
    c.test_keys = ['202301080clt']
    
    used_keys = getUsedKeys(c)
    all_keys = getAllKeys(c)
    
    new_keys = []
    while len(new_keys) < 5:
        rand_index = randrange(0, len(all_keys))
        key = all_keys[rand_index]
        if key not in used_keys:
            new_keys.append(key)
            
    c.test_keys += new_keys
    
    c.setTrain()
    c.setInfoOverwrite()
    
    targetNames = getTargets(c)
    
    c.addTargets(targetNames)
    
    return

def mergeTrain(type):
    _dir = 'playByPlay/train/'
    if type == 's':
        _dir = 'scoringSummaries/train/'
    vector_list, target_list = [], []
    for fn in os.listdir(_dir):
        new_dir = _dir + fn + '/'
        vector_list.append(pd.read_csv("%s.csv" % (new_dir + 'vector_' + fn)))
        target_list.append(pd.read_csv("%s.csv" % (new_dir + 'targets_' + fn)))
    vdf = pd.concat(vector_list)
    tdf = pd.concat(target_list)
    if 'merged' not in os.listdir(_dir):
        os.mkdir(_dir + 'merged/')
    _dir += 'merged/'
    vdf.to_csv("%s.csv" % (_dir + 'vector_merged'), index=False)
    tdf.to_csv("%s.csv" % (_dir + 'targets_merged'), index=False)
    return

def createModels(fn, type):
    
    c = Converter(fn, type)
    
    c.setVector()
    c.setTargets()
    
    c.buildModels()
    
    return

def createPreds(type):
    
    c = Converter('all', type)
    
    c.setVector()
    
    c.buildPreds()
    
    return

def testPreds(type):
    
    c = Converter('all', type)
    
    c.setInfo()
    c.setPreds()
    
    test = c.info.merge(c.preds, on=['key', 'num'], how='left')
    
    # display targets names
    targetNames = [col for col in c.preds.columns if col != 'key' and col != 'num']
    
    entries = []
    
    for i, name in enumerate(targetNames):
        print(str(i) + '. ' + name)
        entries.append((i, name))
        
    option = input('Enter targetName number: ')
    
    try:
        target = [x[1] for x in entries if x[0] == int(option)][0]
    except IndexError:
        print('Option invalid.')
        return
    
    test = test.loc[test[target]==1]
    
    test.to_csv("%s.csv" % (c.base_dir + "testing/test_" + target), index=False)
    
    return

#########################

fn = 'merged'
type = 's'

# newTrain(fn, type)

# transformTrain(fn, type)

# mergeTrain(type)

# createModels(fn, type)

# createPreds(type)

# testPreds(type)