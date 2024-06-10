import os
import pandas as pd
from bs4 import BeautifulSoup
import requests
import regex as re

class Source:
    def __init__(self, gd: pd.DataFrame = None):
        # paths
        self._dir: str = os.getcwd()[:os.getcwd().index("main")+len("main")]+"/"
        self.root_dir = self._dir + "../"
        self.data_dir = self._dir + "games/features/data/"
        # make dir
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        # load gd (gameData) if not passed
        self.gd: pd.DataFrame = pd.read_csv("%s.csv" % (self.root_dir + "data/gameData")) if not gd else gd
        return
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" %  name, index=False)
        return
    def build(self):
        if 'source.csv' in os.listdir(self.data_dir): # source exists
            print('source already exists in: ' + self.data_dir)
            return pd.read_csv("%s.csv" % (self.data_dir + 'source'))
        # source does not exist -> create
        print('Creating source...')
        cols = ['key', 'wy', 'home_abbr', 'away_abbr']
        new_df = self.gd[cols]
        self.save_frame(new_df, (self.data_dir + "source"))
        return new_df
    def build_individual(self):
        if 'source_individual.csv' in os.listdir(self.data_dir): # source_individual exists
            print('source_individual already exists in: ' + self.data_dir)
            return pd.read_csv("%s.csv" % (self.data_dir + 'source_individual'))
        # source_individual does not exist -> create
        print('Creating source_individual...')
        new_df = self.gd.melt(id_vars=['key', 'wy'], value_vars=['home_abbr', 'away_abbr'], var_name='variable', value_name='abbr')[['key', 'wy', 'abbr']]
        new_df.sort_values(by=['key'], inplace=True)
        self.save_frame(new_df, (self.data_dir + "source_individual"))
        return new_df
    def build_new(self, week: int, year: int):
        wy = str(week) + " | " + str(year)
        if 'source.csv' in os.listdir(self.data_dir): # source exists
            source: pd.DataFrame = pd.read_csv("%s.csv" % (self.data_dir + 'source'))
            if wy in source['wy'].values:
                print(f'Extracting new source from train for {wy}')
                source: pd.DataFrame = source.loc[source['wy']==wy]
                self.save_frame(source, (self.data_dir + "source_new"))
                return source
        url = 'https://www.pro-football-reference.com/years/' + str(year) + '/week_' + str(week) + '.htm'
        print(f'Building new source: {wy}')
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
        self.save_frame(df, (self.data_dir + "source_new"))
        return df
    # END Source

#######################

# s = Source()
# s.build_new(1, 2024)