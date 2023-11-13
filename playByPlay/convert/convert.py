import pandas as pd
import numpy as np
import os
import math
from random import randrange
import regex as re

import sys
sys.path.append('../../')
# from playByPlay.namesRegex import getNames, nameToInfo, teamNameToAbbr, PENALTY_TOKENS
from myRegex.namesRegex import getNames, nameToInfo, nameToInfo2, teamNameToAbbr, getTeamNames, getKickoffNames, kickoffNameToAbbr
from paths import DATA_PATH

NAMES_PATH = "../../playerNames/"

# rawTrain all lowercase column names

def saveTestData(key):
    df = pd.read_csv("%s.csv" % "../data/rawTrain")
    df = df.loc[df['key']==key]
    df.to_csv("%s.csv" % "../data/test", index=False)
    return

def saveTestDataByKeyword(keyword):
    df = pd.read_csv("%s.csv" % "../data/rawTrain")
    df.fillna('', inplace=True)
    df = df.loc[df['detail'].str.contains(keyword)]
    df.reset_index(inplace=True, drop=True)
    rand_indexes = [randrange(0, len(df.index)) for i in range(20)]
    df = df.loc[df.index.isin(rand_indexes)]
    df.to_csv("%s.csv" % "../data/test", index=False)
    return

def saveTestDataByMissingPenaltyKeywords():
    df = pd.read_csv("%s.csv" % "../data/rawTrain")
    df.fillna('', inplace=True)
    df = df.loc[df['detail'].str.contains('Penalty')]
    df.reset_index(inplace=True, drop=True)
    lines = list(df['detail'].values)
    all_names = []
    for index, line in enumerate(lines):
        print(index, len(df.index))
        if 'Penalty' in line:
            line = re.sub('Penalty', 'penalty', line)
        [all_names.append(n) for n in getNames(line, False)]
    all_names_set = list(set(all_names))
    counts = [(val, all_names.count(val)) for val in all_names_set]
    counts.sort(key=lambda x: x[1], reverse=True)
    with open('penaltyInfo.txt', 'w') as f:
        for val, count in counts:
            f.write(str(count) + ': ' + val + '\n')
        f.close()
    return

def getPenaltyKeywords():
    df = pd.read_csv("%s.csv" % "../data/rawTrain")
    df.fillna('', inplace=True)
    df = df.loc[df['detail'].str.contains('Penalty')]
    df.reset_index(inplace=True, drop=True)
    # rand_indexes = [randrange(0, len(df.index)) for _ in range(20)]
    # df = df.loc[df.index.isin(rand_indexes)]
    lines = list(df['detail'].values)
    keywords = []
    for index, line in enumerate(lines):
        print(index, len(lines))
        vals = re.findall(r":\s.+", line)
        for i in range(len(vals)):
            if ',' in vals[i]:
                vals[i] = vals[i][:vals[i].index(',')]
            vals[i] = vals[i].replace(': ', '')
            vals[i] = re.sub(r"\(.*?\)", '', vals[i])
        [keywords.append(val) for val in vals]
    keywords_set = list(set(keywords))
    counts = [(val, keywords.count(val)) for val in keywords_set]
    counts.sort(key=lambda x: x[1], reverse=True)
    with open('penaltyKeywords.txt', 'w') as f:
        for val, count in counts:
            f.write(str(count) + ': ' + val + '\n')
        f.close()
    return

def verifyTestInfo():
    
    df = pd.read_csv("%s.csv" % "../data/test_info")
    
    info = ' '.join(df['info'].values)
    
    caps = re.findall(r"[A-Z][a-z]+\s", info)
    
    print(caps)
    
    return

def rawToText():
    df = pd.read_csv("%s.csv" % "../data/rawTrain")
    file = open("rawTrain.txt", 'w')
    for line in df['detail'].values:
        if type(line) is str and len(line) != 0:
            file.write(line + "\n")
    return

def testPossession():
    df = pd.read_csv("%s.csv" % "../data/test")
    df = df.head(10)
    for index, row in df.iterrows():
        num = row['num']
        line = row['detail']
        key = row['key']
        if num != 0:
            if 'Timeout' not in line:
                names = getNames(line, False)
            else:
                names = getTeamNames(line)
            for name in names:
                if 'Timeout' not in line:
                    pid, position = nameToInfo(name, key)
                    info_val = '|' + pid + ',' + position + '|'
                    line = line.replace(name, info_val)
                else:
                    abbr = teamNameToAbbr(name)
                    abbr_val = '|' + abbr + '|'
                    line = line.replace(name, abbr_val)
        else:
            names = getKickoffNames(line)
            for name in names:
                abbr = kickoffNameToAbbr(name)
                abbr_val = '|' + abbr + '|'
                line = line.replace(name, abbr_val)
        # new_lines.append(line)
    return

def convert(useTest):
    
    if useTest:
        df = pd.read_csv("%s.csv" % "../data/test")
        df = df.head(20)
    else:
        df = pd.read_csv("%s.csv" % "../data/rawTrain")
        df = df.head(200)
    
    new_lines = []
    
    cd = pd.read_csv("%s.csv" % "../../data/gameData")
    
    for index, row in df.iterrows():
        num = row['num']
        print(num, max(df['num'].values))
        line = row['detail']
        key = row['key']
        p_abbr = row['possession']
        all_abbrs = list(cd.loc[cd['key']==key, ['home_abbr', 'away_abbr']].values[0])
        wy = row['wy']
        year = wy.split(" | ")[1]
        if num != 0:
            if 'Timeout' not in line:
                names = getNames(line, False)
                isOff_names = [] # append names before parantheses
                if '(' in line:
                    par_idx = line.index('(')
                    for name in names:
                        n_idx = line.index(name)
                        if n_idx < par_idx:
                            isOff_names.append(name)
                else: # no parantheses => all offense
                    isOff_names = names.copy()
            else:
                names = getTeamNames(line)
            for name in names:
                if 'Timeout' not in line:
                    isOff = True if name in isOff_names else False
                    pid, position = nameToInfo2(name, isOff, p_abbr, all_abbrs, year)
                    info_val = '|' + pid + ',' + position + '|'
                    line = line.replace(name, info_val)
                else:
                    abbr = teamNameToAbbr(name)
                    abbr_val = '|' + abbr + '|'
                    line = line.replace(name, abbr_val)
        else:
            names = getKickoffNames(line)
            for name in names:
                abbr = kickoffNameToAbbr(name)
                abbr_val = '|' + abbr + '|'
                line = line.replace(name, abbr_val)
        new_lines.append(line)
    
    df.insert(9, 'info', new_lines)
    
    if useTest:
        df.to_csv("%s.csv" % "../data/test_info", index=False)
    else:
        df.to_csv("%s.csv" % "../data/info", index=False)
    
    return

###########################

# saveTestData('202111070jax')

convert(
    useTest=True
)

# ----------

# testPossession()

# saveTestDataByKeyword(
#     keyword='Penalty'
# )

# saveTestDataByMissingPenaltyKeywords()

# getPenaltyKeywords()

# testPenaltyKeywords()

# verifyTestInfo()

# rawToText()