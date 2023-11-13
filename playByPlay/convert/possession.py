import pandas as pd
import numpy as np
import os
import regex as re
import urllib.request
from urllib.error import HTTPError
import time

from ordered_set import OrderedSet
import random

import sys
sys.path.append('../../')
from myRegex.namesRegex import getNames, nameToInfo, teamNameToAbbr, getTeamNames, convertAltAbbr, getKickoffNames, kickoffNameToAbbr

def find_all(mystr, sub):
    start = 0
    while True:
        start = mystr.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub) # use start += 1 to find overlapping matches

def collectDividers(url, key):
    try:
        fp = urllib.request.urlopen(url)
        mybytes = fp.read()

        mystr = mybytes.decode("utf8", errors='ignore')
        fp.close()

        start0 = mystr.index('id="div_pbp')

        mystr = mystr[start0:]

        s_divs = list(find_all(mystr, '<tr class="divider" '))
        s1_divs = list(find_all(mystr, '<tr class="divider score" '))
        s_divs += s1_divs
        s_divs.sort()
        
        new_df = pd.DataFrame(columns=['info'])
        
        for s in s_divs:
            temp1 = mystr[s:]
            end1 = temp1.index('data-stat="exp_pts_before"')
            temp2 = temp1[:end1]
            temp2 = re.sub(r"<[^>]*>", "", temp2)
            names = getNames(temp2, False)
            for name in names:
                temp2 = temp2.replace(name, '|')
            temp2 = temp2.replace(' (','|')
            temp2 = temp2.replace(' -','|')
            if '|' not in temp2:
                b_index = temp2.index('<')
                temp2 = temp2[:b_index] + '|' + temp2[b_index:]
            info = temp2.split('|')[0]
            new_df.loc[len(new_df.index)] = [info]

        new_df.to_csv("%s.csv.gz" % ("../data/dividers/" + key), index=False, compression='gzip')

    except ValueError:
        print("PBP not found at:", url)

    return

def addPoss(key):
    
    df = pd.read_csv("%s.csv" % ("../data/rawTrain"))
    div = pd.read_csv("%s.csv.gz" % ("../data/dividers/" + key), compression='gzip')
    cd = pd.read_csv("%s.csv" % "../../data/gameData")
    
    df: pd.DataFrame = df.loc[df['key']==key]
    
    abbrs = cd.loc[cd['key']==key, ['home_abbr', 'away_abbr']].values[0]
    
    info_cols = [
        'quarter', 'time', 'down', 'togo',
        'location', 'away_points', 'home_points'
    ]
    df.fillna('', inplace=True)
    
    receiving_abbr = ''
    poss_changes = []
    
    for index, row in df.iterrows():
        line = row['detail']
        num = row['num']
        quarter = row['quarter']
        time = row['time']
        down = row['down']
        if 'challenged' not in line: # challenges
            if quarter != '': # not kickoff
                vals = []
                for col in info_cols:
                    val = row[col]
                    if col not in ['quarter', 'down', 'togo', 'away_points', 'home_points']:
                        vals.append(val)
                    else:
                        try:
                            vals.append(str(int(float(val))))
                        except ValueError:
                            vals.append(val)
                info = ''.join(vals)
                stats = div.loc[div['info'].str.contains(info)]
                if not stats.empty: # possession change
                    other_abbr = list(set(abbrs).difference(set([curr_abbr])))
                    poss_changes.append(other_abbr[0])
                    curr_abbr = other_abbr[0]
                else:
                    if quarter == '3.0' and time == '15:00' and down == '': # start of second half
                        other_abbr = list(set(abbrs).difference(set([receiving_abbr])))
                        poss_changes.append(other_abbr[0])
                        curr_abbr = other_abbr[0]
                    else:
                        poss_changes.append(curr_abbr)
            else:
                names = getKickoffNames(line)
                kickoff_abbrs = []
                for name in names:
                    abbr = kickoffNameToAbbr(name)
                    kickoff_abbrs.append(abbr)
                    abbr_val = '|' + abbr + '|'
                    line = line.replace(name, abbr_val)
                if len(kickoff_abbrs) > 1:
                    abbrs = [abbr.replace('|','') for abbr in re.findall(r"\|.*?\|", line)]
                    poss_changes.append(abbrs[-1])
                    curr_abbr = abbrs[-1]
                    receiving_abbr = curr_abbr
                else:
                    poss_changes.append(kickoff_abbrs[0])
                    curr_abbr = kickoff_abbrs[0]
                    receiving_abbr = curr_abbr
        else:
            poss_changes.append(curr_abbr)
                
    df.insert(2, 'possession', poss_changes)
    
    df = df[['key', 'num', 'possession']]
    
    return df

def buildPossession():
    
    df = pd.read_csv("%s.csv" % "../data/rawTrain")
    
    keys = list(OrderedSet(df['key'].values))
    
    # keys = [keys[random.randrange(0, len(keys))] for _ in range(10)]
    
    df_list = []
    
    # -------------------------------------------------------
    # # test
    
    # rand_index = random.randrange(0, len(keys))
    # key = keys[rand_index]
    # print(key)
    # url = 'https://www.pro-football-reference.com/boxscores/' + key + '.htm'
    # collectDividers(url, key)
    # df_list.append(addPoss(key))
    
    # df = pd.concat(df_list)
    
    # df.to_csv("%s.csv" % "../data/possessions", index=False)
    
    # -------------------------------------------------------
    
    if 'possessions.csv' in os.listdir("../data"):
        new_df = pd.read_csv("%s.csv" % "../data/possessions")
        found_keys = OrderedSet(new_df['key'].values)
        keys = list(OrderedSet(keys).difference(found_keys))
        df_list.append(new_df)
    
    for index, key in enumerate(keys):
        if index == 1000:
            break
        try:
            print(key, index, len(keys))
            url = 'https://www.pro-football-reference.com/boxscores/' + key + '.htm'
            collectDividers(url, key)
            df_list.append(addPoss(key))
            time.sleep(5)
        except (HTTPError, IndexError, ValueError, UnboundLocalError) as err:
            print(err)
            break
        
    df = pd.concat(df_list)
    
    df.to_csv("%s.csv" % "../data/possessions", index=False)
    
    return

def join():
    pf = pd.read_csv("%s.csv" % "../data/possessions")
    df = pd.read_csv("%s.csv" % "../data/rawTrain")
    df = df.merge(pf, on=['key', 'num'], how='left')
    new_cols = [
        'key', 'num', 'quarter',
        'time', 'down', 'togo', 
        'location', 'away_points', 'home_points', 
        'possession', 'detail', 'epb', 
        'epa' 
    ]
    df = df[new_cols]
    df.to_csv("%s.csv" % "../data/rawTrain", index=False)
    return

######################

key = '202301290phi'
url = 'https://www.pro-football-reference.com/boxscores/' + key + '.htm'
collectDividers(url, key)
# addPoss(key)

# buildPossession()

# join()

# 200009100cin - KEY ERROR - curr_abbr unbound