import pandas as pd
import numpy as np
import os
import random
import string

class TeamElos:
    def __init__(self, df: pd.DataFrame, _dir: str):
        self.df = df
        self._dir = _dir
        self.team_elos_raw = pd.DataFrame()
    def build(self):
        """
        Creates team_elos_raw.csv using gameData.csv, getElo function, and getEndElo function.
        Contains elo for each abbr going into next week. (elo calculated from 1 | 2022 corresponds to 2 | 2022).
        Output file is team_elos_raw.csv.
        """
        if 'team_elos_raw.csv' in os.listdir(self._dir):
            print('team_elos_raw.csv already built. Setting class attribute team_elos_raw.')
            self.setRawElos(pd.read_csv("%s.csv" % (self._dir + "team_elos_raw")))
            return
        abbrs = list(set(self.df['home_abbr'].values))
        new_df = pd.DataFrame(columns=['wy', 'abbr', 'elo'])
        first_wy = self.df['wy'].values[0]
        for a in abbrs:
            new_df.loc[len(new_df.index)] = [first_wy, a, 1500]
        end_abbrs = abbrs.copy()
        # self.df = self.df.loc[(self.df['wy'].str.contains('2021'))|(self.df['wy'].str.contains('2022'))]
        # self.df.reset_index(drop=True, inplace=True)
        for index, row in self.df.iterrows():
            self.printProgressBar(index, len(self.df.index), 'team_elos_raw Progress')
            wy = row['wy']
            home_abbr, away_abbr = row['home_abbr'], row['away_abbr']
            home_points, away_points = row['home_points'], row['away_points']
            winning_abbr = row['winning_abbr']
            home_curr_elo = new_df.loc[new_df['abbr']==home_abbr, 'elo'].values[-1]
            away_curr_elo = new_df.loc[new_df['abbr']==away_abbr, 'elo'].values[-1]
            try: # add elo to next week - elo going into the week
                next_wy = self.df.iloc[self.df.loc[self.df['wy']==wy].index.values[-1]+1]['wy']
                year = int(wy.split(" | ")[1])
                next_year = int(next_wy.split(" | ")[1])
                new_home_elo = self.getElo(home_abbr, winning_abbr, home_curr_elo, away_curr_elo, home_points, away_points)
                new_away_elo = self.getElo(away_abbr, winning_abbr, away_curr_elo, home_curr_elo, away_points, home_points)
                if year == next_year: # same year
                    new_df.loc[len(new_df.index)] = [next_wy, home_abbr, new_home_elo]
                    new_df.loc[len(new_df.index)] = [next_wy, away_abbr, new_away_elo]
                else: # new year
                    new_df.loc[len(new_df.index)] = [next_wy, home_abbr, self.getEndElo(new_home_elo)]
                    new_df.loc[len(new_df.index)] = [next_wy, away_abbr, self.getEndElo(new_away_elo)]
                    end_abbrs.remove(home_abbr)
                    end_abbrs.remove(away_abbr)
                    if wy != self.df.iloc[index+1]['wy']: # add missing abbrs - didnt play last week of prev year
                        for abbr1 in end_abbrs:
                            last_elo = new_df.loc[new_df['abbr']==abbr1, 'elo'].values[-1]
                            new_df.loc[(len(new_df.index))] = [next_wy, abbr1, self.getEndElo(last_elo)]
                            end_abbrs = abbrs.copy()
            except IndexError: # add elo end (END OF DATA) - getElo and then getEndElo for END(year)
                end_year = int(wy.split(" | ")[1])
                new_wy = '1 | ' + str(end_year+1)
                new_df.loc[len(new_df.index)] = [new_wy, home_abbr, self.getEndElo(self.getElo(home_abbr, winning_abbr, home_curr_elo, away_curr_elo, home_points, away_points))]
                new_df.loc[len(new_df.index)] = [new_wy, away_abbr, self.getEndElo(self.getElo(away_abbr, winning_abbr, away_curr_elo, home_curr_elo, away_points, home_points))]
                end_abbrs.remove(home_abbr)
                end_abbrs.remove(away_abbr)
        for abbr1 in end_abbrs: # add abbrs not present in last week of season
            last_elo = new_df.loc[new_df['abbr']==abbr1, 'elo'].values[-1]
            new_df.loc[(len(new_df.index))] = [new_wy, abbr1, self.getEndElo(last_elo)]
        self.saveFrame(new_df, 'team_elos_raw')
        self.setRawElos(new_df)
        return
    def createBoth(self, source: pd.DataFrame, isNew: bool = False):
        """
        Merges team_elos_raw with source to create both (head-to-head) elos.
        Output file is team_elos.csv
        @params:
            source   - Required  : source or new_source (DataFrame)
            isNew    - Required  : condition for if source is original or for new week
        """
        if 'team_elos.csv' in os.listdir(self._dir) and not isNew:
            print('team_elos.csv already built.')
            return
        print('Creating team_elos...')
        self.setRawElos()
        new_df = pd.DataFrame(columns=list(source.columns)+['home_elo', 'away_elo'])
        for index, row in source.iterrows():
            wy = row['wy']
            home_abbr, away_abbr = row['home_abbr'], row['away_abbr']
            home_elo = self.getRawElo(home_abbr, wy)
            away_elo = self.getRawElo(away_abbr, wy)
            new_df.loc[len(new_df.index)] = list(row.values) + [home_elo, away_elo]
        if not isNew: # all elos
            self.saveFrame(new_df, 'team_elos')
        else: # new week elos
            self.saveFrame(new_df, 'team_elos_new')
            return new_df
        return
    def update(self):
        """
        Updates team_elos_raw.csv when gameData contains new/unseen wys
        @params:
            isNewYear   - Required  : determines if new wy will increment week or year (bool)
        """
        self.setRawElos()
        abbrs = list(set(self.df['home_abbr'].values))
        df_wy: str = self.df['wy'].values[-1]
        elo_wy: str = self.team_elos_raw['wy'].values[-1]
        elo_week: int = int(elo_wy.split(" | ")[0])
        elo_year: int = int(elo_wy.split(" | ")[-1])
        new_last_wy: str = str(int(df_wy.split(" | ")[0])+1) + " | " + df_wy.split(" | ")[-1]
        if elo_week == 1:
            new_last_wy: str = elo_wy
        # gameData-wy matches team_elos_raw-wy => not updated
        if new_last_wy != self.team_elos_raw['wy'].values[-1]:
            min_wy = self.team_elos_raw['wy'].values[-1]
            max_wy = self.df['wy'].values[-1]
            missing_wys = [(str(w) + " | " + str(elo_year)) for w in range(int(min_wy.split(" | ")[0]), int(max_wy.split(" | ")[0])+1)]
            for last_wy in missing_wys:
                abbrs = list(set(self.df['home_abbr'].values))
                new_df = pd.DataFrame(columns=['wy', 'abbr', 'elo'])
                week = int(last_wy.split(" | ")[0])
                year = int(last_wy.split(" | ")[1])
                # new_wy = "1 | " + str(year+1) if isNewYear else str(week+1) + " | " + str(elo_year)
                new_wy: str = str(week+1) + " | " + str(elo_year)
                for index, row in self.df.loc[self.df['wy']==last_wy].iterrows():
                    home_abbr, away_abbr = row['home_abbr'], row['away_abbr']
                    abbrs.remove(home_abbr) # remove abbrs to add missing teams
                    abbrs.remove(away_abbr) # remove abbrs to add missing teams
                    home_points, away_points = row['home_points'], row['away_points']
                    winning_abbr = row['winning_abbr']
                    home_last_elo = self.team_elos_raw.loc[self.team_elos_raw['abbr']==home_abbr, 'elo'].values[-1]
                    away_last_elo = self.team_elos_raw.loc[self.team_elos_raw['abbr']==away_abbr, 'elo'].values[-1]
                    home_elo = self.getElo(home_abbr, winning_abbr, home_last_elo, away_last_elo, home_points, away_points)
                    away_elo = self.getElo(away_abbr, winning_abbr, away_last_elo, home_last_elo, away_points, home_points)
                    # if isNewYear:
                    #     home_elo = self.getEndElo(home_elo)
                    #     away_elo = self.getEndElo(away_elo)
                    new_df.loc[len(new_df.index)] = [new_wy, home_abbr, home_elo]
                    new_df.loc[len(new_df.index)] = [new_wy, away_abbr, away_elo]
                for abbr in abbrs:
                    last_elo = self.team_elos_raw.loc[self.team_elos_raw['abbr']==abbr, 'elo'].values[-1]
                    new_df.loc[len(new_df.index)] = [new_wy, abbr, last_elo]
                self.team_elos_raw = pd.concat([self.team_elos_raw, new_df])
                self.saveFrame(self.team_elos_raw, 'team_elos_raw')
            print(f"team_elos_raw.csv updated. Added elos for wys: {missing_wys}")
        else:
            print(f'team_elos_raw.csv already up-to-date. New last wy: {new_last_wy}')
        return
    def getK(self, mov, eloDif):
        n = (mov + 3) ** 0.8
        d = 7.5 + (0.006 * eloDif)
        return 20 * (n/d)
    def getExpected(self, teamElo, oppElo):
        dif = oppElo - teamElo
        x = 10 ** (dif/400)
        return 1 / (1 + x)
    def getElo(self, abbr, winning_abbr, teamElo, oppElo, teamPoints, oppPoints):
        mov = abs(teamPoints - oppPoints)
        eloDif = abs(teamElo - oppElo)
        actual = 1 if abbr == winning_abbr else 0
        k = self.getK(mov, eloDif)
        expected = self.getExpected(teamElo, oppElo)
        return k * (actual - expected) + teamElo
    def getEndElo(self, teamElo):
        return 1500 if teamElo == 1500 else ((teamElo * 0.75) + (0.25 * 1505))
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % (self._dir + name), index=False)
        return
    def setRawElos(self, df: pd.DataFrame = None):
        self.team_elos_raw = df if df is not None else pd.read_csv("%s.csv" % (self._dir + "team_elos_raw")) 
        return
    def getRawElo(self, abbr, wy):
        try:
            return self.team_elos_raw.loc[(self.team_elos_raw['abbr']==abbr)&(self.team_elos_raw['wy']==wy), 'elo'].values[0]
        except IndexError:
            # get most recent elo if curr elo not present
            start = self.team_elos_raw.loc[self.team_elos_raw['wy']==wy].index.values[0]
            return self.team_elos_raw.loc[(self.team_elos_raw.index<start)&(self.team_elos_raw['abbr']==abbr), 'elo'].values[-1]
        return

# / end class
    
def createMockData(wy: str, cd: pd.DataFrame):
    abbrs = list(set(cd['home_abbr'].values))
    num_games = 14
    h_abbrs = abbrs[:num_games]
    a_abbrs = abbrs[num_games:]
    new_df = pd.DataFrame(columns=['key', 'wy', 'home_abbr', 'away_abbr', 'home_points', 'away_points', 'winning_abbr'])
    for i in range(num_games):
        home_abbr, away_abbr = h_abbrs[i], a_abbrs[i]
        home_points = random.randrange(0, 50)
        away_points = random.randrange(0, 50)
        winning_abbr = home_abbr if home_points >= away_points else away_abbr
        key = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
        new_df.loc[len(new_df.index)] = [key, wy, home_abbr, away_abbr, home_points, away_points, winning_abbr]
    new_df.to_csv("%s.csv" % ("mockData_" + wy.replace(" | ", "-")), index=False)
    return

# createMockData('1 | 2024', pd.read_csv("%s.csv" % "../../../data/gameData"))

# TeamElos(pd.read_csv("%s.csv" % "../../../data/gameData"), 'data/').update()

