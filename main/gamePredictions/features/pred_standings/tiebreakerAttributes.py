import pandas as pd
import numpy as np
import os

pd.options.mode.chained_assignment = None

class Standing:
    def __init__(self, abbr, year):
        self.abbr = abbr
        self.year = year
        self.wins = 0
        self.loses = 0
        self.ties = 0
        self.wl = 0
        self.confWins = 0
        self.confLoses = 0
        self.confTies = 0
        self.confWl = 0
        self.divWins = 0
        self.divLoses = 0
        self.divTies = 0
        self.divWl = 0
        self.sov = 0 # strength of victory -> combined win lose ratio of teams defeated
        self.sos = 0 # strength of schedule -> combined win lose ratio of all teams played against
        self.pointsFor = 0
        self.pointsAgainst = 0
        self.tdsFor = 0
        self.tdsAgainst = 0
        self.division = ''
        self.conference = ''
    def getObjKey(self):
        return self.abbr + "-" + str(self.year)
    def show(self):
        print(self.__dict__)
        return
    def asDict(self):
        return vars(self)
    def zeroDivision(self, n, d):
        return n/d if d else 0
    def wlFunc(self, wins, loses, ties):
        return round(self.zeroDivision(((2 * wins) + ties), (2 * (wins + loses + ties))), 4)
    def updateWlRatios(self):
        self.wl = self.wlFunc(self.wins, self.loses, self.ties)
        self.confWl = self.wlFunc(self.confWins, self.confLoses, self.confTies)
        self.divWl = self.wlFunc(self.divWins, self.divLoses, self.divTies)
        return
# / END Standings

class TiebreakerAttributes:
    def __init__(self, df: pd.DataFrame, sl: pd.DataFrame, _dir):
        """
        @params
            source_indiv    - Required : individual source, each abbr and key (DataFrame)
            df              - Required : game data (DataFrame)
            sl              - Required : season length data (DataFrame)
        """
        self._dir = _dir
        self.div_dir = _dir + '/divisionData/'
        self.div_78 = pd.read_csv("%s.csv" % (self.div_dir + "divisions_78-01"))
        self.div_02 = pd.read_csv("%s.csv" % (self.div_dir + "divisions_02-22"))
        self.div_df: pd.DataFrame = None
        self.df: pd.DataFrame = df
        self.sl: pd.DataFrame = sl
        self.tb: pd.DataFrame = None
        self.game: pd.DataFrame = None
        self.standings = []
        return
    def setStandingsData(self, abbr, opp_abbr, year):
        # use same year obj or create new for current year
        st: Standing = self.getStanding(abbr, year) if self.objInStandings(abbr, year) else Standing(abbr, year)
        pointsFor, pointsAgainst = self.getPointsForPointsAgainst(abbr)
        # compute features
        conf = self.getConference(abbr, year)
        opp_conf = self.getConference(opp_abbr, year)
        division = self.getDivision(abbr, year)
        opp_division = self.getDivision(opp_abbr, year)
        # set division and conference tags
        st.division = division
        st.conference = conf
        # get wins loses and ties for all games, in conference, and in division
        st = self.calcWinLoseTie(st, pointsFor, pointsAgainst, conf, opp_conf, division, opp_division)
        # calc win lose ratios for all, conference, and division
        st.updateWlRatios()
        # calc strength of victory and strength of schedule
        opp_wl = self.getOppWl(opp_abbr, year)
        st = self.calcSOVandSOS(st, opp_wl, pointsFor, pointsAgainst)
        # pointsFor and pointsAgainst rankings
        st = self.updatePointRanks(st, pointsFor, pointsAgainst)
        # tdsFor and tdsAgainst rankings
        tdsFor, tdsAgainst = self.getTdsForandTdsAgainst(abbr)
        st = self.updateTdsRank(st, tdsFor, tdsAgainst)
        # append obj if it does not exist
        if not self.objInStandings(abbr, year):
            self.standings.append(st)
        return
    def getTiebreakerAttributes(self):
        allStandings = []
        # self.df = self.df.loc[self.df['wy'].str.contains('2022')]
        # self.df.reset_index(drop=True, inplace=True)
        for index, row in self.df.iterrows():
            self.printProgressBar(index, len(self.df.index), 'TiebreakerAttributes Progress')
            key = row['key']
            wy = row['wy']
            week = int(wy.split(" | ")[0])
            year = int(wy.split(" | ")[1])
            home_abbr, away_abbr = row[['home_abbr', 'away_abbr']]
            self.div_df = self.div_02 if year >= 2002 else self.div_78
            # only compute regular season attributes
            if self.isRegularSeason(week, year):
                self.game = self.df.loc[self.df['key']==key]
                self.setStandingsData(home_abbr, away_abbr, year)
                self.setStandingsData(away_abbr, home_abbr, year)
                # week change
                if index < max(self.df.index):
                    next_wy = self.df.loc[self.df.index==index+1, 'wy'].values[0]
                    if wy != next_wy:
                        yearStandings = [s for s in self.standings if s.year == year]
                        temp_df = pd.DataFrame([s.asDict() for s in yearStandings])
                        # final season rankings => wy = last week of season
                        seasonWeeks = self.sl.loc[self.sl['year']==year, 'weeks'].values[0]
                        if week <= seasonWeeks:
                            temp_df.insert(1, 'wy', [wy for _ in range(len(yearStandings))])
                            temp_df.drop(columns=['year'], inplace=True)
                            temp_df.sort_values(by=['conference', 'division'], inplace=True)
                            allStandings.append(temp_df)
                else: # no week change
                    yearStandings = [s for s in self.standings if s.year == year]
                    temp_df = pd.DataFrame([s.asDict() for s in yearStandings])
                    # final season rankings => wy = last week of season
                    seasonWeeks = self.sl.loc[self.sl['year']==year, 'weeks'].values[0]
                    if week <= seasonWeeks:
                        temp_df.insert(1, 'wy', [wy for _ in range(len(yearStandings))])
                        temp_df.drop(columns=['year'], inplace=True)
                        temp_df.sort_values(by=['conference', 'division'], inplace=True)
                        allStandings.append(temp_df)
        new_df = pd.concat(allStandings)
        return new_df
    def buildAllTiebreakerAttributes(self):
        if 'tiebreakerAttributes.csv' in os.listdir(self._dir):
            print('tiebreakerAttributes.csv already built. Proceeding.')
            return
        new_df = self.getTiebreakerAttributes()
        self.saveFrame(new_df, (self._dir + "tiebreakerAttributes"))
        return
    def isRegularSeason(self, week: int, year: int):
        seasonWeeks = self.sl.loc[self.sl['year']==year, 'weeks'].values[0]
        if week <= seasonWeeks + 1:
            return True
        return False
    def objInStandings(self, abbr, year):
        objKey = abbr + "-" + str(year)
        if objKey in [s.getObjKey() for s in self.standings]:
            return True
        return False
    def getStanding(self, abbr, year):
        return [s for s in self.standings if s.abbr == abbr and s.year == year][0]
    def getOppWl(self, abbr, year):
        temp = [s for s in self.standings if s.abbr == abbr and s.year == year]
        if len(temp) > 0:
            return temp[0].wl
        return 0
    def getPointsForPointsAgainst(self, abbr):
        home_abbr = self.game['home_abbr'].values[0]
        if abbr == home_abbr:
            pointsFor = self.game['home_points'].values[0]
            pointsAgainst = self.game['away_points'].values[0]
        else:
            pointsFor = self.game['away_points'].values[0]
            pointsAgainst = self.game['home_points'].values[0]
        return pointsFor, pointsAgainst
    def calcWinLoseTie(self, st: Standing, pointsFor, pointsAgainst, conf, opp_conf, division, opp_division):
        if pointsFor > pointsAgainst:
            st.wins += 1
            if conf == opp_conf:
                st.confWins += 1
            if division == opp_division:
                st.divWins += 1
        elif pointsFor < pointsAgainst:
            st.loses += 1
            if conf == opp_conf:
                st.confLoses += 1
            if division == opp_division:
                st.divLoses += 1
        elif pointsFor == pointsAgainst:
            st.ties += 1
            if conf == opp_conf:
                st.confTies += 1
            if division == opp_division:
                st.divTies += 1
        return st
    def getConference(self, abbr, year):
        afc = self.div_df.loc[self.div_df['year']==year, 'afc'].values[0]
        if abbr in afc:
            return 'afc'
        return 'nfc'
    def getDivision(self, abbr, year):
        cols = list(self.div_df.columns)
        cols.remove('year')
        cols.remove('afc')
        cols.remove('nfc')
        temp = self.div_df.loc[self.div_df['year']==year]
        for col in cols:
            abbrs = temp[col].values[0]
            if abbr in abbrs:
                return col
        return None
    def calcSOVandSOS(self, st, opp_wl, pointsFor, pointsAgainst):
        if pointsFor > pointsAgainst:
            st.sov += opp_wl
        st.sos += opp_wl
        return st
    def updatePointRanks(self, st, pointsFor, pointsAgainst):
        st.pointsFor += pointsFor
        st.pointsAgainst += pointsAgainst
        return st
    def calcOppVal(self, pointsFor, pointsAgainst):
        if pointsFor > pointsAgainst:
            return 1 # win
        elif pointsFor < pointsAgainst:
            return 0 # lose
        return 2 # tie
    def getTdsForandTdsAgainst(self, abbr):
        game = self.game
        home_abbr = game['home_abbr'].values[0]
        if abbr == home_abbr:
            tdsFor = game['home_pass_touchdowns'].values[0] + game['home_rush_touchdowns'].values[0]
            tdsAgainst = game['away_pass_touchdowns'].values[0] + game['away_rush_touchdowns'].values[0]
        else:
            tdsFor = game['away_pass_touchdowns'].values[0] + game['away_rush_touchdowns'].values[0]
            tdsAgainst = game['home_pass_touchdowns'].values[0] + game['home_rush_touchdowns'].values[0]
        return tdsFor, tdsAgainst
    def updateTdsRank(self, st, tdsFor, tdsAgainst):
        st.tdsFor += tdsFor
        st.tdsAgainst += tdsAgainst
        return st
    def buildMockData(self, wy: str):
        df = self.df.loc[self.df['wy']=='5 | 2022']
        df['wy'] = wy
        df['key'] = [('MOCK_' + str(i) + '_' + wy.replace(" | ", "-")) for i in range(len(df.index))]
        self.saveFrame(df, (self._dir + "mockGameData_" + wy.replace(" | ","-")))
        return
    def update(self):
        self.setTb()
        last_wy_tb = self.tb['wy'].values[-1]
        last_wy_df = self.df['wy'].values[-1]
        week = int(last_wy_df.split(" | ")[0])
        year = int(last_wy_df.split(" | ")[1])
        if last_wy_tb != last_wy_df and self.isRegularSeason(week, year):
            self.df = self.df.loc[self.df['wy'].str.contains(str(year))]
            new_df = self.getTiebreakerAttributes()
            new_df = new_df.loc[new_df['wy']==last_wy_df]
            self.tb = pd.concat([self.tb, new_df])
            self.saveFrame(self.tb, (self._dir + "tiebreakerAttributes"))
            print(f"tiebreakerAttributes updated for wy: {last_wy_df}")
        else:
            print("tiebreakerAttributes.csv already up-to-date.")
        return
    def setTb(self):
        self.tb = pd.read_csv("%s.csv" % (self._dir + "tiebreakerAttributes"))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
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
    
# / END TiebreakerAttributes
    
############################

# source = pd.read_csv("%s.csv" % "../source/sourceIndividual")
# df = pd.read_csv("%s.csv" % "../../../../data/gameData")
# # df = pd.concat([df, pd.read_csv("%s.csv" % "mockGameData_1-2023")])
# # df = pd.concat([df, pd.read_csv("%s.csv" % "mockGameData_2-2023")])
# # df.reset_index(drop=True, inplace=True)
# sl = pd.read_csv("%s.csv" % "../../../../data/seasonLength")

# tb = TiebreakerAttributes(df, sl, './')

# tb.buildAllTiebreakerAttributes()

# tb.buildMockData('2 | 2023')

# tb.update()