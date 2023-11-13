import pandas as pd
import numpy as np
import os
import time
import urllib.request
from urllib.error import HTTPError
import regex as re
from googlesearch import search

abbrs = {
    'afc_east': ['BUF', 'MIA', 'NWE', 'NYJ'],
    'afc_north': ['RAV', 'CIN', 'CLE', 'PIT'],
    'afc_south': ['HTX', 'CLT', 'JAX', 'OTI'],
    'afc_west': ['DEN', 'KAN', 'RAI', 'SDG'],
    'nfc_east': ['DAL', 'NYG', 'PHI', 'WAS'],
    'nfc_north': ['CHI', 'DET', 'GNB', 'MIN'],
    'nfc_south': ['ATL', 'CAR', 'NOR', 'TAM'],
    'nfc_west': ['CRD', 'RAM', 'SFO', 'SEA']
}

urls = {
    'afc_east': 'https://www.nfl.com/news/afc-east-projected-starters-for-2023-nfl-season-bills-dolphins-jets-all-contende',
    'afc_north': 'https://www.nfl.com/news/afc-north-projected-starters-for-2023-nfl-season-lamar-flanked-by-playmakers-ben',
    'afc_south': 'https://www.nfl.com/news/afc-south-projected-starters-for-2023-nfl-season-jaguars-clear-division-favorite',
    'afc_west': 'https://www.nfl.com/news/afc-west-projected-starters-for-2023-nfl-season-chiefs-built-to-contend-playoffs',
    'nfc_east': 'https://www.nfl.com/news/nfc-east-projected-starters-for-2023-nfl-season-eagles-cowboys-boast-conference-',
    'nfc_north': 'https://www.nfl.com/news/nfc-north-projected-starters-for-2023-nfl-season-lions-look-like-division-s-best',
    'nfc_south': 'https://www.nfl.com/news/nfc-south-projected-starters-for-2023-nfl-season-falcons-go-old-school-bryce-you',
    'nfc_west': 'https://www.nfl.com/news/nfc-west-projected-starters-for-2023-nfl-season-seahawks-closing-in-on-49ers-car',
}

simple_positions = {
    'DE': 'DL', 'CB/S': 'DB', 'RT': 'OL',
    'LG': 'OL', 'RG': 'OL', 'NT': 'DL',
    'Slot': 'DB', 'ILB': 'LB', 'Edge': 'DL',
    'WLB': 'LB', 'LT': 'OL', 'MLB': 'LB',
    'SLB': 'LB', 'OLB': 'LB', 'S': 'DB',
    'C': 'OL', 'CB': 'DB', 'DT': 'DL'
}

def saveRawStarters():
    df_list = []
    for conf in urls:
        print(conf)
        url = urls[conf]
        curr_abbrs = abbrs[conf]
        tables = pd.read_html(url)
        for i, df in enumerate(tables):
            df.columns = ['o_position', 'o_name', 'd_position', 'd_name']
            new_df = pd.DataFrame()
            new_df['name'] = np.concatenate([df['o_name'].values, df['d_name'].values])
            new_df['position'] = np.concatenate([df['o_position'].values, df['d_position'].values])
            new_df.insert(0, 'abbr', curr_abbrs[i])
            df_list.append(new_df)
        time.sleep(2)
    pd.concat(df_list).to_csv("%s.csv" % "rawStarters_w1", index=False)
    return

def getPids(name, url):
    
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

def cleanStarters():
    names = pd.read_csv("%s.csv" % "../../playerNames/finalPlayerInfo")
    df = pd.read_csv("%s.csv" % "rawStarters_w1")
    new_df = pd.DataFrame(columns=['key', 'abbr', 'wy', 'starters'])
    abbrs = list(set(df['abbr'].values))
    for abbr in abbrs:
        temp_df: pd.DataFrame = df.loc[df['abbr']==abbr]
        starters = []
        for index, row in temp_df.iterrows():
            abbr = row['abbr']
            name = row['name']
            position = row['position']
            s_position = simple_positions[position] if position in simple_positions else position
            pids = names.loc[
                ((names['name'].str.contains(name))|(names['aka'].str.contains(name)))&
                (names['position']==s_position),
                'p_id'
            ].values
            if len(pids) == 0:
                pfr_name = name.lower().replace(' ', '+')
                url = 'https://www.pro-football-reference.com/search/search.fcgi?hint=' + pfr_name + '&search=' + pfr_name
                try:
                    pids = getPids(name, url)
                except HTTPError as error:
                    print(error)
                    print(name, abbr, position)
                    temp_pid = input('Enter pid: ')
                    pids = [temp_pid]
            if len(pids) > 1: # same names - user input select
                print(name, abbr, position, pids)
                idx = input('Enter pids index: ')
                pid = pids[int(idx)]
            else: # unique name
                pid = pids[0]
            starters.append(pid + ':' + s_position)
        new_df.loc[len(new_df.index)] = ['UNK', abbr, '1 | 2023', '|'.join(starters)]
    new_df.to_csv("%s.csv" % "cleanStarters_w1", index=False)
    return

#################################

# saveRawStarters()

cleanStarters()


