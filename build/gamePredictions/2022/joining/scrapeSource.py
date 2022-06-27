from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np

def buildInfo(df):
    
    newCols = ['time', 'week', 'season', 'seasonWins',
               'seasonLoses', 'seasonWinPercentage', 'isHome',
               'isPlayoffs', 'isWeekBeforePlayoffs', 'isWeekAfterBye']
    cols = list(df.columns) + newCols
    
    new_df = pd.DataFrame(columns=cols)
    
    for index, row in df.iterrows():
        time = 0
        week = int(row['wy'].split(" | ")[0])
        season = int(row['wy'].split(" | ")[1])
        seasonWins = 0
        seasonLoses = 0
        seasonWinPercentage = 0
        if index % 2 == 0:
            isHome = 1
        else:
            isHome = 0
        isPlayoffs = 0
        isWeekBeforePlayoffs = 0
        isWeekAfterBye = 0
        new_df.loc[len(new_df.index)] = [row['key'], row['opp_abbr'], row['wy'],
                                         time, week, season, seasonWins, seasonLoses,
                                         seasonWinPercentage, isHome, isPlayoffs, isWeekBeforePlayoffs,
                                         isWeekAfterBye]
    
    new_df.to_csv("%s.csv" % ("sourceInfo_w" + wy.split(" | ")[0]), index=False)
    
    return

def build(url, wy):

    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")

    links = []

    # find links
    for link in soup.find_all('a'):
        l = link.get('href')
        if ('/boxscores/' in l or '/teams/' in l) and '2022' in l:
            links.append(l)

    df = pd.DataFrame()

    keys, opp_abbrs = [], []

    # parse links   
    for i in range(0, len(links), 3):
        away_abbr = links[i].split("/")[2].upper()
        home_abbr = links[i+2].split("/")[2].upper()
        key = links[i+1].split("/")[2].replace(".htm","")
        keys.append(home_abbr + "-" + key)
        opp_abbrs.append(away_abbr)
        keys.append(away_abbr + "-" + key)
        opp_abbrs.append(home_abbr)
        
    df['key'] = keys
    df['opp_abbr'] = opp_abbrs
    df['wy'] = [wy for i in range(len(keys))]
    
    df.to_csv("%s.csv" % ("source_w" + wy.split(" | ")[0]), index=False)
    
    # source info
    # base -> manually alter some columns
    # ! time (0: morning, 1: afternoon, 2: night) !
    buildInfo(df)
    
    return

################################

url = 'https://www.pro-football-reference.com/years/2022/week_1.htm'
wy = '1 | 2022'

build(url, wy)