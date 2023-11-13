import pandas as pd
import numpy as np
import os
import urllib.request
from urllib.error import HTTPError
import regex as re
from googlesearch import search
import time

import sys
sys.path.append('../')

from myRegex.namesRegex import getNames, TEST_SENTENCES

def getContent(name, url):
    
    pattern = r"/players/[A-Z]/.+.htm"
    
    try: # pfr search works
        fp = urllib.request.urlopen(url)
        mybytes = fp.read()
        mystr = mybytes.decode("utf8", errors='ignore')
        fp.close()
        start = mystr.index('<h1>Search Results</h1>')
        mystr = mystr[start:]
        end = mystr.index('class="search-pagination"')
        mystr = mystr[:end]      
    except ValueError: # pfr search does not work
        all_urls = []
        for i in search(name + ' pro football reference', num=5, stop=5, pause=1):
            if re.search(r"www\.pro-football-reference\.com/players/[A-Z]/", i):
                all_urls.append(i)
        mystr = '\n'.join(all_urls)
        
    links = re.findall(pattern, mystr)
    pids = []
    for link in links:
        link = link.split('/')
        pid = link[-1].replace('.htm', '')
        if pid not in pids:
            pids.append(pid)
    
    return pids

def getNameFromPid(pid):
    
    p_char = pid[0].upper()
    url = 'https://www.pro-football-reference.com/players/' + p_char +'/' + pid + '.htm'
    
    fp = urllib.request.urlopen(url)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8", errors='ignore')
    fp.close()
    start = mystr.index('class="nothumb">')
    mystr = mystr[start:]
    end = mystr.index('</h1>')
    mystr = mystr[:end]
   
    return getNames(mystr, False)[0]

def addNames():
    
    df = pd.read_csv("%s.csv" % "../playByPlay/data/rawTrain")
    ndf = pd.read_csv("%s.csv" % "playerNames")
    
    df.fillna('', inplace=True)
    
    new_df = pd.DataFrame(columns=['name', 'pids'])
    
    for index, row in df.iterrows():
        print(index, len(df.index))
        line = row['detail']
        try:
            names = getNames(line, False)
            if 'Timeout' not in line and 'Overtime' not in line:
                for name in names:
                    if name not in ndf['name'].values and name not in new_df['name'].values:
                        pfr_name = name.lower().replace(' ', '+')
                        url = 'https://www.pro-football-reference.com/search/search.fcgi?hint=' + pfr_name + '&search=' + pfr_name
                        pids = getContent(name, url)
                        new_df.loc[len(new_df.index)] = [name, '|'.join(pids)]
                        time.sleep(2)
        except HTTPError as err:
            print('ERROR:', err)
    
    new_df.to_csv("%s.csv" % "playerNames_pbp", index=False)
    
    return

def cleanNames():
    
    df = pd.read_csv("%s.csv" % "playerNames_pbp")
    df.fillna('', inplace=True)
    
    cd = pd.read_csv("%s.csv" % "../playByPlay/data/rawTrain")
    cd.fillna('', inplace=True)
    
    new_names = []
    
    for index, row in df.iterrows():
        name = row['name']
        f_names = getNames(name, False)
        if 'Interception' in name:
            detail = cd.loc[cd['detail'].str.contains(name), 'detail'].values[0]
            start = detail.index('Interception by')
            detail = detail[start+len('Interception by'):]
            new_names = getNames(detail, False)
            for n1 in new_names:
                n1_split = n1.split(" ")
                for n2 in n1_split:
                    if n2 in name:
                        new_names.append(n1)
        else:
            new_names.append(np.NaN)
            
    
    
    return

def validatePids():
    
    df = pd.read_csv("%s.csv" % "playerNames_pbp")
    df.fillna('', inplace=True)
    
    # df = df.head(10)
    
    valid_pids = []
    
    for index, row in df.iterrows():
        # print(index, max(df.index))
        name = row['name']
        firstName = name.split(" ")[0]
        lastName = name.split(" ")[-1]
        pids = row['pids'].split("|")
        if len(pids) > 1:
            temp_pids = []
            test_pid = lastName[:4] + firstName[:2]
            for pid in pids:
                if test_pid in pid:
                    temp_pids.append(pid)
            valid_pids.append('|'.join(temp_pids))
        else:
            valid_pids.append('|'.join(pids))
            
    df['valid_pids'] = valid_pids
    
    df.to_csv("%s.csv" % "playerNames_pbp-temp", index=False)
    
    return

def getNewNames():
    
    df = pd.read_csv("%s.csv" % "rawPlayerNames_pbp")
    df.fillna('', inplace=True)
    
    cd = pd.read_csv("%s.csv" % "../playByPlay/data/rawTrain")
    cd.fillna('', inplace=True)
    
    # for index, row in df.iterrows():
    #     name = row['name']
    #     name = re.sub(r"\bJr\b", '', name)
    #     if 'Interception' in name:
    #         df.drop(index, inplace=True)
    #     sentence = TEST_SENTENCES[0].replace('@', name)
    #     f_names = getNames(sentence, False)
    #     try:
    #         if name.strip() != f_names[0]:
    #             print(name, ':', f_names[0])
    #     except IndexError:
    #         print(name, f_names)
    
    all_new_names = []
    
    for index, row in df.iterrows():
        print(index, len(df.index))
        name = row['name']
        line = cd.loc[cd['detail'].str.contains(name), 'detail'].values[0]
        new_names = getNames(line, False)
        if 'Interception' in name:
            name = re.sub(r"\bInterception by\s\b", '', name)
        name = name.strip()
        valid_names = []
        for n in new_names:
            if name in n:
                valid_names.append(n)
        try:
            all_new_names.append(valid_names[0])
        except IndexError:
            all_new_names.append('')
            
    df.insert(1, 'new_name', all_new_names)
    
    df.to_csv("%s.csv" % "playerNames_pbp", index=False)
    
    return

def getNewPids():
    
    df = pd.read_csv("%s.csv" % "playerNames_pbp")
    df.fillna('', inplace=True)
    
    all_new_pids = []
    
    for index, row in df.iterrows():
        print(index, len(df.index))
        name = row['name']
        new_name = row['new_name']
        pids = row['pids']
        if name.strip() != new_name:
            pfr_name = new_name.lower().replace(' ', '+')
            url = 'https://www.pro-football-reference.com/search/search.fcgi?hint=' + pfr_name + '&search=' + pfr_name
            new_pids = getContent(new_name, url)
            all_new_pids.append('|'.join(new_pids))
            time.sleep(2)
        else:
            all_new_pids.append(pids)
        if len(name) == 0:
            df.drop(index, inplace=True)
            
    df['new_pids'] = all_new_pids
            
    df.to_csv("%s.csv" % "playerNames_pbp", index=False)
    
    return

def createFinalNames():
    
    df = pd.read_csv("%s.csv" % "playerNames_pbp")
    df.dropna(inplace=True)
    
    new_df = pd.DataFrame(columns=['name', 'pids'])
    
    for index, row in df.iterrows():
        print(index, len(df.index))
        pids = row['pids']
        name = row['new_name']
        if re.search(r'\b\sJr\b', name):
            name = re.sub(r'\b\sJr\b', '', name)
        firstName = name.split(" ")[0]
        lastName = name.split(" ")[-1]
        new_pids = row['pids']
        all_pids = pids + '|' + new_pids
        all_pids = list(set(all_pids.split('|')))
        valid_pids = []
        if re.search(r"\.", name):
            firstName = firstName.replace('.', '')
        if len(lastName) < 4:
            lastName += 'x'
        test_pid = lastName[:4] + firstName[:2]
        for pid in all_pids:
            if test_pid in pid:
                valid_pids.append(pid)
        if len(valid_pids) == 0:
            pfr_name = name.lower().replace(' ', '+')
            url = 'https://www.pro-football-reference.com/search/search.fcgi?hint=' + pfr_name + '&search=' + pfr_name
            try:
                f_pids = getContent(name, url)
                [valid_pids.append(p) for p in f_pids]
                time.sleep(5)
            except HTTPError as error:
                valid_pids.append('UNK')
                print(name, ':', error)
        new_df.loc[len(new_df.index)] = [name, '|'.join(valid_pids)]
    
    new_df.to_csv("%s.csv" % "finalPlayerNames_pbp", index=False)
    
    return

def findUnkPids():
    
    df = pd.read_csv("%s.csv" % "finalPlayerNames_pbp")
    df = df.loc[df['pids']=='UNK']
    df.reset_index(drop=True, inplace=True)
    
    new_df = pd.DataFrame(columns=['name', 'pids'])
    
    past_df = pd.read_csv("%s.csv" % "unkPlayerNames_pbp-1")
    
    df = df.loc[~df['name'].isin(past_df['name'])]
    df.reset_index(drop=True, inplace=True)
    
    for index, row in df.iterrows():
        print(index, len(df.index))
        name = row['name']
        pfr_name = name.lower().replace(' ', '+')
        url = 'https://www.pro-football-reference.com/search/search.fcgi?hint=' + pfr_name + '&search=' + pfr_name
        try:
            pids = getContent(name, url)
            if len(pids) == 0:
                pids = ['UNK']
            new_df.loc[len(new_df.index)] = [name, '|'.join(pids)]
        except HTTPError as error:
            break
        time.sleep(2)
        
    new_df.to_csv("%s.csv" % "unkPlayerNames_pbp-2", index=False)
    
    return

def addUnkPids():
    
    df = pd.read_csv("%s.csv" % "finalPlayerNames_pbp")
    
    udf = pd.concat([
        pd.read_csv("%s.csv" % "unkPlayerNames_pbp-1"),
        pd.read_csv("%s.csv" % "unkPlayerNames_pbp-2")
    ])
    
    for index, row in df.iterrows():
        name = row['name']
        pids = row['pids']
        if 'UNK' in pids:
            new_pids = udf.loc[udf['name']==name, 'pids'].values[0]
            df.iloc[index]['pids'] = new_pids
    
    df.to_csv("%s.csv" % "finalPlayerNames_pbp", index=False)
    
    return

def filterPids1():
    
    df = pd.read_csv("%s.csv" % "finalPlayerNames_pbp")
    odf = pd.read_csv("%s.csv" % "playerNames")
    
    for index, row in df.iterrows():
        # print(index, len(df.index))
        pids = row['pids'].split("|")
        name = row['name']
        if re.search(r'\b\sJr\b', name):
            name = re.sub(r'\b\sJr\b', '', name)
        firstName = name.split(" ")[0]
        lastName = name.split(" ")[-1]
        valid_pids = []
        if re.search(r"\.", name):
            firstName = firstName.replace('.', '')
        if len(lastName) < 4:
            lastName += 'x'
        test_pid = lastName[:4] + firstName[:2]
        for pid in pids:
            if test_pid in pid:
                valid_pids.append(pid)
        if len(valid_pids) == 0 and len(pids) >= 2:
            if len(firstName) == 2:
                test_pid = lastName[:4] + firstName[0] + '.'
                for pid in pids:
                    if test_pid in pid:
                        valid_pids.append(pid)
        if len(valid_pids) == 0 and re.search(r"\'", name):
            name = re.sub(r"\'", '', name)
            firstName = name.split(" ")[0]
            lastName = name.split(" ")[-1]
            test_pid = lastName[:4] + firstName[:2]
            for pid in pids:
                if test_pid in pid:
                    valid_pids.append(pid)
        if len(valid_pids) == 0:
            test_pid = lastName[:4] + firstName[0] + '.'
            for pid in pids:
                if test_pid in pid:
                    valid_pids.append(pid)
        if len(valid_pids) == 0:
            df.iloc[index]['pids'] = '|'.join(pids)
        else:
            df.iloc[index]['pids'] = '|'.join(valid_pids)
            
    df.to_csv("%s.csv" % "finalPlayerNames_pbp-1", index=False)
    
    return

def expand():
    
    df = pd.read_csv("%s.csv" % "finalPlayerNames_pbp")
    
    new_df = pd.DataFrame(columns=['p_id', 'name'])
    
    for index, row in df.iterrows():
        name = row['name']
        pids = row['pids'].split("|")
        for pid in pids:
            new_df.loc[len(new_df.index)] = [pid, name]
            
    new_df.to_csv("%s.csv" % "finalPlayerNames_pbp", index=False)
    
    return

######################

# addNames()

# validatePids()

# cleanNames()

# getNewNames()

# getNewPids()

# createFinalNames()

# findUnkPids()

# addUnkPids()

# filterPids1()

# filterPids2()

# expand()