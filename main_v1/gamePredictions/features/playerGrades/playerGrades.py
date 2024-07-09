import pandas as pd
import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
import statsmodels.api as sm
import random
import matplotlib.pyplot as plt
import time
import multiprocessing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class Elo:
    def __init__(self, pid: str, elo: float):
        self.pid = pid
        self.elo = elo
        return

class PlayerGrades:
    def __init__(self, position, _dir):
        self.position = position
        self.game_grades: pd.DataFrame = None
        self.elos: pd.DataFrame = None
        self._dir = _dir
        self.data_dir = _dir + 'data/' + position + '/'
        self.grade_funcs = {
            'qb': self.getGrade_qb, 'rb': self.getGrade_rb, 'wr': self.getGrade_wr
        }
        self.start: float = 0
        self.end: float = 0
        return
    def buildElos(self):
        self.start = time.time()
        """
        Creates and writes player elos to data_dir.
        """
        if (self.position + '_elos.csv') in os.listdir(self.data_dir):
            print((self.position + '_elos.csv') + ' already built. Proceeding to createBoth.')
            return
        self.setGameGrades()
        df: pd.DataFrame = self.game_grades
        # df = df.loc[df['wy'].str.contains('2022')]
        # df.reset_index(drop=True, inplace=True)
        first_wy = df['wy'].values[0]
        all_pids = list(set(df['p_id']))
        all_pids.sort()
        end_pids = all_pids.copy()
        new_df = pd.DataFrame(columns=['p_id', 'wy', 'elo'])
        # init
        for pid in all_pids:
            new_df.loc[len(new_df.index)] = [pid, first_wy, 100]
        total_mean = np.mean(df['grade'].values)
        for index, vals in enumerate(df[['p_id', 'wy', 'grade']].values):
            self.printProgressBar(index, len(df.index), 'Elos progress')
            pid, wy, grade = vals
            year = int(wy.split(" | ")[1])
            mean = self.getMeanGrade(df, pid, year)
            # grade = (grade - 1) if grade < mean else grade
            # grade = (((grade - mean)*1.5) + (mean - total_mean))
            grade = self.grade_funcs[self.position](grade, mean, total_mean)
            curr_elo = new_df.loc[new_df['p_id']==pid, 'elo'].values[-1]
            try:# add elo to next week - elo going into the week
                next_wy = df.iloc[df.loc[df['wy']==wy].index.values[-1]+1]['wy']
                year = int(wy.split(" | ")[1])
                next_year = int(next_wy.split(" | ")[1])
                new_elo = curr_elo + grade
                if year == next_year: # same year
                    new_df.loc[len(new_df.index)] = [pid, next_wy, new_elo]
                else: # new year
                    new_df.loc[len(new_df.index)] = [pid, next_wy, self.getEndElo(new_elo)]
                    try:
                        end_pids.remove(pid)
                    except ValueError:
                        pass
                    if wy != df.iloc[index+1]['wy']: # add missing abbrs - didnt play last week of prev year
                        for pid1 in end_pids:
                            last_elo = new_df.loc[new_df['p_id']==pid1, 'elo'].values[-1]
                            new_df.loc[(len(new_df.index))] = [pid1, next_wy, self.getEndElo(last_elo)]
                            end_pids = all_pids.copy()
            except IndexError: # add elo end (END OF DATA) - getElo and then getEndElo for END(year)
                end_year = int(wy.split(" | ")[1])
                new_wy = '1 | ' + str(end_year+1)
                new_df.loc[len(new_df.index)] = [pid, new_wy, self.getEndElo(curr_elo + grade)]
                try:
                    end_pids.remove(pid)
                except ValueError:
                    pass
            # # remove pids, no games in future
            # future_pids = df.loc[df.index>index, 'p_id'].values
            # if pid not in future_pids:
            #     try:
            #         all_pids.remove(pid)
            #     except ValueError:
            #         pass
        for pid1 in end_pids: # add pids not present in last week of season
            last_elo = new_df.loc[new_df['p_id']==pid1, 'elo'].values[-1]
            new_df.loc[(len(new_df.index))] = [pid1, new_wy, self.getEndElo(last_elo)]
        self.saveFrame(new_df, (self.data_dir + self.position + "_elos"))
        self.setElos()
        self.end = time.time()
        elapsed = self.end - self.start
        print(f"Time elapsed: {elapsed}")
        return
    def buildElosV2(self):
        """
        Creates and writes player elos to data_dir.
        """
        # if (self.position + '_elos.csv') in os.listdir(self.data_dir):
        #     print((self.position + '_elos.csv') + ' already built. Proceeding to createBoth.')
        #     return
        self.setGameGrades()
        df: pd.DataFrame = self.game_grades
        # df = df.loc[df['p_id']=='McCaCh01']
        df = df.loc[df['wy'].str.contains('2022')]
        df.reset_index(drop=True, inplace=True)
        all_pids = list(set(df['p_id']))
        all_pids.sort()
        total_mean = np.mean(df['grade'].values)
        new_df = pd.DataFrame(columns=['p_id', 'wy', 'elo'])
        for pid in all_pids:
            data: pd.DataFrame = df.loc[df['p_id']==pid]
            data.reset_index(drop=True, inplace=True)
            elo = 100
            for index, row in data.iterrows():
                wy = row['wy']
                year = int(wy.split(" | ")[1])
                grade = row['grade']
                mean = self.getMeanGrade(data, pid, year)
                grade = self.grade_funcs[self.position](grade, mean, total_mean)
                elo += grade
                if index != data.index[-1]:
                    next_wy = data.iloc[index+1]['wy']
                    next_year = int(next_wy.split(" | ")[1])
                    if year != next_year:
                        elo = self.getEndElo(elo)
                    new_df.loc[len(new_df.index)] = [pid, next_wy, elo]
            # add first row - elo
            first_wy = data['wy'].values[0]
            first_row = [{'p_id': pid, 'wy': first_wy, 'elo': 100}]
            new_df = pd.concat([pd.DataFrame(first_row), new_df])
            # add last row - elo
            last_wy = data['wy'].values[-1]
            last_year = int(last_wy.split(" | ")[1])
            if last_year == 2022:
                new_wy = '1 | ' + str(last_year+1)
                new_df.loc[len(new_df.index)] = [pid, new_wy, self.getEndElo(elo)]
        self.saveFrame(new_df, (self.data_dir + self.position + '_elos_v2'))
        return
    def getMeanGrade(self, df: pd.DataFrame, pid: str, year: int):
        data = df.loc[(df['p_id']==pid)&(df['wy'].str.contains(str(year))), 'grade'].values
        return np.mean(data) if len(data) != 0 else np.mean(df['grade'].values)
    def getGrade_qb(self, grade: float, mean: float, total_mean: float):
        grade = (grade - 1) if grade < mean else grade
        grade = (((grade - mean)*1.5) + (mean - total_mean))
        return grade
    def getGrade_rb(self, grade: float, mean: float, total_mean: float):
        return grade
    def getGrade_wr(self, grade: float, mean: float, total_mean: float):
        return grade
    def graphElos(self, pids: list, year: int):
        """
        Plots player elos.
        @params:
            pids   - Required  : pids to be plotted (list)
            year   - Required  : year for pids to be plotted (int)
        """
        self.setElos()
        df = self.elos.loc[(self.elos['wy'].str.contains(str(year)))&(self.elos['p_id'].isin(pids))]
        for pid in pids:
            plt.plot(df.loc[df['p_id']==pid, 'elo'].values)
        plt.legend(pids)
        plt.show()
        return
    def testEloFunction(self):
        """
        Test/find good function for elo calculation.
        """
        n = 5
        mean = 3.3694273127753305
        a_grades = [4.5, 3.3, 4.3, 4.4, 2.6, 4.7]
        b_grades = [2.5, 3.1, 2.8, 2.4, 3.2, 3.3]
        elos = [100]
        info = {
            'a': {
                'grades': [random.uniform(3.5, 4.1) for _ in range(n)],
                'elos': [100]
            },
            'b': {
                'grades': [random.uniform(2.3, 4) for _ in range(n)],
                'elos': [100]
            }
        }
        for key in info:
            grades = info[key]['grades']
            elos = info[key]['elos']
            for grade in grades:
                print(f"Normal grade: {grade}")
                # grade = (grade - 3) if grade < mean else grade
                grade = (grade - mean)*5
                print(f"New grade: {grade}")
                info[key]['elos'].append(elos[-1] + grade)
        #     plt.plot(info[key]['elos'])
        # plt.show()
        return
    def checkBestElos(self, wy: str):
        self.setElos()
        df = self.elos.loc[self.elos['wy']==wy, ['p_id', 'elo']]
        df.sort_values(by=['elo'], ascending=False, inplace=True)
        for index, row in df.iterrows():
            pid, elo = row[['p_id', 'elo']]
            print(f"p_id: {pid}, elo: {elo}")
        return
    def createBoth(self, source: pd.DataFrame, sdf: pd.DataFrame, isNew: bool):
        """
        Converts elos to home vs. away data. Using player with highest elo for given team.
        Returns and prints message if (position + "_playerGrades.csv) already exists.
        @params:
            source   - Required  : source info (DataFrame)
            sdf      - Required  : all starters (DataFrame)
            isNew    - Required  : determines all train or new week (bool)
        """
        if (self.position + "_playerGrades.csv") in os.listdir(self._dir) and not isNew:
            print(self.position + "_playerGrades.csv already built. Using exisiting.")
            return
        self.setElos()
        new_df = pd.DataFrame(columns=list(source.columns)+[('home_' + self.position + '_elo'), ('away_' + self.position + '_elo')])
        for index, row in source.iterrows():
            if not isNew:
                self.printProgressBar(index, len(source.index), (self.position + 'playerGrades Progress'))
            key = row['key']
            wy = row['wy']
            home_abbr, away_abbr = row['home_abbr'], row['away_abbr']
            try:
                home_starters = (sdf.loc[(sdf['key']==key)&(sdf['abbr']==home_abbr), 'starters'].values[0]).split("|")
                away_starters = (sdf.loc[(sdf['key']==key)&(sdf['abbr']==away_abbr), 'starters'].values[0]).split("|")
            except IndexError:
                home_starters = []
                away_starters = []
            if len(home_starters) != 0:
                try:
                    home_starters = self.getStarterElos(home_starters, wy)
                    home_elo = list(home_starters.values())[0]
                except IndexError:
                    home_elo = 100
            if len(away_starters) != 0:
                try:
                    away_starters = self.getStarterElos(away_starters, wy)
                    away_elo = list(away_starters.values())[0]
                except IndexError:
                    away_elo = 100
            new_df.loc[len(new_df.index)] = list(row.values) + [home_elo, away_elo]
        if not isNew:
            self.saveFrame(new_df, (self._dir + self.position + "_playerGrades"))
        return new_df
    def getStarterElos(self, starters: list, wy: str):
        starters = self.getPositionStarters(starters)
        df = self.elos
        info = {}
        for s in starters:
            try:
                elo = df.loc[(df['p_id']==s)&(df['wy']==wy), 'elo'].values[0]
            except IndexError: # last elo if not present for wy
                elo = df.loc[df['p_id']==s, 'elo'].values[-1]
            info[s] = elo
        info = dict(sorted(info.items(), key=lambda item: item[1], reverse=True))
        return info
    def getPositionStarters(self, starters: list):
        return [s.split(":")[0] for s in starters if s.split(":")[1] == self.position.upper()]
    def update(self, isNewYear: bool):
        """
        Updates elos for new week.
        @params:
            isNewYear   - Required  : determines if week or year will be incremented (bool)
        """
        self.setGameGrades()
        self.setElos()
        last_wy_gg = self.game_grades['wy'].values[-1]
        new_wy_gg = self.getNewWy(last_wy_gg, isNewYear)
        last_wy_e = self.elos['wy'].values[-1]
        new_wy_e = self.getNewWy(last_wy_e, isNewYear)
        if new_wy_gg == new_wy_e:
            df: pd.DataFrame = self.game_grades.loc[self.game_grades['wy']==last_wy_gg]
            total_mean = np.mean(self.game_grades['grade'].values)
            new_df = pd.DataFrame(columns=self.elos.columns)
            for index, row in df.iterrows():
                pid = row['p_id']
                grade = row['grade']
                wy = row['wy']
                year = int(wy.split(" | ")[1])
                mean = self.getMeanGrade(self.game_grades, pid, year)
                grade = (grade - 1) if grade < mean else grade
                grade = (((grade - mean)*1.5) + (mean - total_mean))
                try:
                    prev_elo = self.elos.loc[self.elos['p_id']==pid, 'elo'].values[-1]
                except IndexError: # unseen p_id (new player)
                    prev_elo = 100
                elo = (prev_elo + grade) if not isNewYear else self.getEndElo(prev_elo + grade)
                new_df.loc[len(new_df.index)] = [pid, new_wy_e, elo]
            self.elos = pd.concat([self.elos, new_df])
            self.saveFrame(self.elos, (self.data_dir + self.position + "_elos"))
        else:
            print(f"{self.position}_elos already up-to-date.")
        return
    def getNewWy(self, wy: str, isNewYear: str):
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        return (str(week+1) + " | " + str(year)) if not isNewYear else ("1 | " + str(year+1))
    def getEndElo(self, elo):
        # elo = ((elo * 0.75) + (0.25 * 20))
        return 100 + ((elo - 100)/10) if self.position == 'qb' else 100
        # return 100
    def setGameGrades(self):
        self.game_grades = pd.read_csv("%s.csv" % (self.data_dir + self.position + "_gameGrades"))
        return
    def setElos(self):
        self.elos = pd.read_csv("%s.csv" % (self.data_dir + self.position + "_elos"))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    
############################

pg = PlayerGrades('qb', './')

# pg.buildElos()

pg.checkBestElos('18 | 2022')

# pg.graphElos(['JeffJu00', 'AdamDa01', 'LazaAl00', 'WatsCh00'], 2022)

# source = pd.read_csv("%s.csv" % "../source/source")
# sdf = pd.read_csv("%s.csv" % "../../../../starters/allStarters")

# pg.createBoth(source, sdf, False)

# pg.update(isNewYear=False)

# source = pd.read_csv("%s.csv" % "../source/new_source")
# sdf = pd.read_csv("%s.csv" % "../../../../data/starters_23/starters_w2")

# pg.createBoth(source, sdf, True)

# arr = np.arange(58, 123, 5)

# for val in arr:
#     new_val = 100 + ((val - 100)/10)
#     print(f"val, new_val: {val}, {new_val}")