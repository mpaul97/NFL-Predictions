import pandas as pd
import numpy as np
import os
import random
import string

class CoachElos:
    def __init__(self, df: pd.DataFrame, cdf: pd.DataFrame, _dir: str):
        self.df = df
        self.cdf = cdf
        self._dir = _dir
        self.raw_elos = pd.DataFrame()
    def build(self):
        """
        Creates rawCoachElos.csv using gameData.csv, getElo function, and getEndElo function.
        Contains elo for each abbr going into next week. (elo calculated from 1 | 2022 corresponds to 2 | 2022).
        Output file is rawCoachElos.csv.
        """
        if 'rawCoachElos.csv' in os.listdir(self._dir):
            print('rawCoachElos.csv already built. Setting class attribute raw_elos.')
            self.setRawElos(pd.read_csv("%s.csv" % (self._dir + "rawCoachElos")))
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
            self.printProgressBar(index, len(self.df.index), 'rawCoachElos Progress')
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
                    home_isNewCoach = self.getIsNewCoach(home_abbr, next_year)
                    away_isNewCoach = self.getIsNewCoach(away_abbr, next_year)
                    new_home_elo = 1500 if home_isNewCoach == 1 else new_home_elo
                    new_away_elo = 1500 if away_isNewCoach == 1 else new_away_elo
                    new_df.loc[len(new_df.index)] = [next_wy, home_abbr, new_home_elo]
                    new_df.loc[len(new_df.index)] = [next_wy, away_abbr, new_away_elo]
                    end_abbrs.remove(home_abbr)
                    end_abbrs.remove(away_abbr)
                    # add missing abbrs - didnt play last week of prev year
                    if wy != self.df.iloc[index+1]['wy']:
                        for abbr1 in end_abbrs:
                            isNewCoach = self.getIsNewCoach(abbr1, next_year)
                            last_elo = new_df.loc[new_df['abbr']==abbr1, 'elo'].values[-1]
                            elo = 1500 if isNewCoach == 1 else last_elo
                            new_df.loc[(len(new_df.index))] = [next_wy, abbr1, elo]
                            end_abbrs = abbrs.copy()
            except IndexError: # add elo end (END OF DATA) - getElo and then getEndElo for END(year)
                end_year = int(wy.split(" | ")[1])
                next_year = end_year + 1
                new_wy = '1 | ' + str(next_year)
                home_isNewCoach = self.getIsNewCoach(home_abbr, next_year)
                away_isNewCoach = self.getIsNewCoach(away_abbr, next_year)
                home_curr_elo = self.getElo(home_abbr, winning_abbr, home_curr_elo, away_curr_elo, home_points, away_points)
                away_curr_elo = self.getElo(away_abbr, winning_abbr, away_curr_elo, home_curr_elo, away_points, home_points)
                home_curr_elo = 1500 if home_isNewCoach == 1 else home_curr_elo
                away_curr_elo = 1500 if away_isNewCoach == 1 else away_curr_elo
                new_df.loc[len(new_df.index)] = [new_wy, home_abbr, home_curr_elo]
                new_df.loc[len(new_df.index)] = [new_wy, away_abbr, away_curr_elo]
                end_abbrs.remove(home_abbr)
                end_abbrs.remove(away_abbr)
        for abbr1 in end_abbrs: # add abbrs not present in last week of season
            isNewCoach = self.getIsNewCoach(abbr1, next_year)
            last_elo = new_df.loc[new_df['abbr']==abbr1, 'elo'].values[-1]
            elo = 1500 if isNewCoach == 1 else last_elo
            new_df.loc[(len(new_df.index))] = [new_wy, abbr1, elo]
        self.saveFrame(new_df, 'rawCoachElos')
        self.setRawElos(new_df)
        return
    def createBoth(self, source: pd.DataFrame, isNew: bool):
        """
        Merges rawCoachElos with source to create both (head-to-head) elos.
        Output file is coachElos.csv
        @params:
            source   - Required  : source or new_source (DataFrame)
            isNew    - Required  : condition for if source is original or for new week
        """
        if 'coachElos.csv' in os.listdir(self._dir) and not isNew:
            print('coachElos.csv already built.')
            return
        new_df = pd.DataFrame(columns=list(source.columns)+['coach_home_elo', 'coach_away_elo'])
        for index, row in source.iterrows():
            self.printProgressBar(index, len(source.index), 'coachElos Progress')
            wy = row['wy']
            home_abbr, away_abbr = row['home_abbr'], row['away_abbr']
            home_elo = self.getRawElo(home_abbr, wy)
            away_elo = self.getRawElo(away_abbr, wy)
            new_df.loc[len(new_df.index)] = list(row.values) + [home_elo, away_elo]
        if not isNew: # all elos
            self.saveFrame(new_df, 'coachElos')
        else: # new week elos
            self.saveFrame(new_df, 'newCoachElos')
            return new_df
        return
    def update(self, isNewYear: bool):
        """
        Updates rawCoachElos.csv when gameData contains new/unseen wy
        @params:
            isNewYear   - Required  : determines if new wy will increment week or year (bool)
        """
        abbrs = list(set(self.df['home_abbr'].values))
        last_wy = self.df['wy'].values[-1]
        # gameData-wy matches raw_elos-wy => not updated
        if last_wy == self.raw_elos['wy'].values[-1]:
            new_df = pd.DataFrame(columns=['wy', 'abbr', 'elo'])
            week = int(last_wy.split(" | ")[0])
            year = int(last_wy.split(" | ")[1])
            next_year = year + 1
            new_wy = "1 | " + str(next_year) if isNewYear else str(week+1) + " | 2023"
            for index, row in self.df.loc[self.df['wy']==last_wy].iterrows():
                home_abbr, away_abbr = row['home_abbr'], row['away_abbr']
                abbrs.remove(home_abbr) # remove abbrs to add missing teams
                abbrs.remove(away_abbr) # remove abbrs to add missing teams
                home_points, away_points = row['home_points'], row['away_points']
                winning_abbr = row['winning_abbr']
                home_last_elo = self.raw_elos.loc[self.raw_elos['abbr']==home_abbr, 'elo'].values[-1]
                away_last_elo = self.raw_elos.loc[self.raw_elos['abbr']==away_abbr, 'elo'].values[-1]
                home_elo = self.getElo(home_abbr, winning_abbr, home_last_elo, away_last_elo, home_points, away_points)
                away_elo = self.getElo(away_abbr, winning_abbr, away_last_elo, home_last_elo, away_points, home_points)
                if isNewYear:
                    home_isNewCoach = self.getIsNewCoach(home_abbr, next_year)
                    away_isNewCoach = self.getIsNewCoach(away_abbr, next_year)
                    home_elo = 1500 if home_isNewCoach == 1 else home_elo
                    away_elo = 1500 if away_isNewCoach == 1 else away_elo
                    home_elo = self.getEndElo(home_elo)
                    away_elo = self.getEndElo(away_elo)
                new_df.loc[len(new_df.index)] = [new_wy, home_abbr, home_elo]
                new_df.loc[len(new_df.index)] = [new_wy, away_abbr, away_elo]
            for abbr in abbrs:
                last_elo = self.raw_elos.loc[self.raw_elos['abbr']==abbr, 'elo'].values[-1]
                new_df.loc[len(new_df.index)] = [new_wy, abbr, last_elo]
            self.raw_elos = pd.concat([self.raw_elos, new_df])
            self.saveFrame(self.raw_elos, 'rawCoachElos')
            print(f"rawCoachElos.csv updated. Added elos for wy: {new_wy}")
        else:
            print(f'rawCoachElos.csv already up-to-date. Last wy: {last_wy}')
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
    def setRawElos(self, df: pd.DataFrame):
        if df is not None:
            self.raw_elos = df
        else:
            self.raw_elos = pd.read_csv("%s.csv" % (self._dir + "rawCoachElos"))
            print("Class attribute raw_elos set from local dir.")
        return
    def getRawElo(self, abbr, wy):
        try:
            return self.raw_elos.loc[(self.raw_elos['abbr']==abbr)&(self.raw_elos['wy']==wy), 'elo'].values[0]
        except IndexError:
            # get most recent elo if curr elo not present
            start = self.raw_elos.loc[self.raw_elos['wy']==wy].index.values[0]
            return self.raw_elos.loc[(self.raw_elos.index<start)&(self.raw_elos['abbr']==abbr), 'elo'].values[-1]
        return
    def getIsNewCoach(self, abbr, next_year):
        try:
            isNewCoach = self.cdf.loc[(self.cdf['abbr']==abbr)&(self.cdf['year']==next_year), 'isNewCoach'].values[0]
        except IndexError:
            isNewCoach = 0
        return isNewCoach
    # Print iterations progress
    def printProgressBar (self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
        filledLength = int(length * iteration // total)
        bar = fill * filledLength + '-' * (length - filledLength)
        print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        # Print New Line on Complete
        if iteration == total: 
            print()
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
    
######################

# cd = pd.read_csv("%s.csv" % "../../../../data/gameData")
# cdf = pd.read_csv("%s.csv" % "../../../../coaches/coachInfo")
# # source = pd.read_csv("%s.csv" % "../source/source")
# # new_source = pd.read_csv("%s.csv" % "../source/new_source")

# # # cd = pd.concat([cd, pd.read_csv("%s.csv" % "mockData_1-2023")])

# e = CoachElos(cd, cdf, './')

# # e.build()

# e.setRawElos(None)

# # # e.createBoth(source, False)

# e.update(False)

# e.createBoth(new_source, True)

# createMockData('1 | 2023', cd)