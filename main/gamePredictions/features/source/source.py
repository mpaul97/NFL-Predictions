import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
import requests
import regex as re

def buildSource(cd: pd.DataFrame, _dir):
    
    # source exists
    if 'source.csv' in os.listdir(_dir):
        print('source already exists in: ' + _dir)
        return pd.read_csv("%s.csv" % (_dir + 'source'))
        
    # source does not exist 
    print('Creating source...')
    
    new_df = pd.DataFrame(columns=['key', 'wy', 'home_abbr', 'away_abbr'])
    
    for index, row in cd.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        new_df.loc[len(new_df.index)] = [key, wy, home_abbr, away_abbr]
        
    new_df.to_csv("%s.csv" % (_dir + "source"), index=False)
    
    return new_df

def buildSourceIndividual(cd: pd.DataFrame, _dir):
    
    # source exists
    if 'sourceIndividual.csv' in os.listdir(_dir):
        print('sourceIndividual already exists in: ' + _dir)
        return pd.read_csv("%s.csv" % (_dir + 'sourceIndividual'))
        
    # source does not exist 
    print('Creating sourceIndividual...')
    
    new_df = pd.DataFrame(columns=['key', 'abbr', 'opp_abbr', 'wy'])
    
    for index, row in cd.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        new_df.loc[len(new_df.index)] = [key, home_abbr, away_abbr, wy]
        new_df.loc[len(new_df.index)] = [key, away_abbr, home_abbr, wy]
        
    new_df.to_csv("%s.csv" % (_dir + "sourceIndividual"), index=False)
    
    return

def buildNewSource(week, year, _dir):
    
    url = 'https://www.pro-football-reference.com/years/' + str(year) + '/week_' + str(week) + '.htm'
    
    wy = str(week) + " | " + str(year)
    
    print('Building new source: ' + wy + ' ...')

    page = requests.get(url)

    soup = BeautifulSoup(page.content, "html.parser")

    links = []

    # find links
    for link in soup.find_all('a'):
        l = link.get('href')
        if (re.search(r"boxscores\/[0-9]{9}", l) or re.search(r"teams\/[a-z]{3}", l)):
            links.append(l)
            
    if 'teams' in links[0] and 'teams' in links[1]:
        links.pop(0)

    df = pd.DataFrame(columns=['key', 'wy', 'home_abbr', 'away_abbr'])

    # parse links   
    for i in range(0, len(links)-2, 3):
        away_abbr = links[i].split("/")[2].upper()
        home_abbr = links[i+2].split("/")[2].upper()
        key = links[i+1].split("/")[2].replace(".htm","")
        if re.search(r"[0-9]{9}[a-z]{3}", key):
            df.loc[len(df.index)] = [key, wy, home_abbr, away_abbr]
    
    df.to_csv("%s.csv" % (_dir + "new_source"), index=False)
    
    return df

# buildNewSource(1, 2023, './')