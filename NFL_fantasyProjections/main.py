import pandas as pd
import numpy as np
import os
import random
import urllib.request
from urllib.error import HTTPError
import regex as re
from googlesearch import search
import time
import math
from ordered_set import OrderedSet
import json

pd.options.mode.chained_assignment = None

class Main:
    def __init__(self, _dir):
        self.positions = ['QB', 'RB', 'WR', 'TE']
        self.merge_cols = ['p_id', 'year', 'position']
        self._dir = _dir
        self.data_dir = _dir + 'data/'
        self.json_dir = _dir + 'data/jsons/'
        self.all_ranks: pd.DataFrame = None
        self.indiv_dir = _dir + 'projections_indiv_stats/data/'
        self.indiv_pred_dirs = { pos: (self.indiv_dir + pos + '/') for pos in self.positions }
        self.names_df = pd.read_csv("%s.csv" % (self._dir + "../playerNames/finalPlayerInfo"))
        self.team_names_df = pd.read_csv("%s.csv" % (self._dir + "../teamNames/teamNames_line"))
        self.indiv_frame: pd.DataFrame = None
        self._types = ['std', 'ppr', 'half']
        self.raw_frames = { _type: pd.read_csv("%s.csv" % (self.data_dir + "raw_ranks_" + _type)) for _type in self._types }
        self.clean_frames = { _type: pd.read_csv("%s.csv" % (self.data_dir + "clean_ranks_" + _type)) for _type in self._types }
        self.info_frames = { _type: pd.read_csv("%s.csv" % (self.data_dir + "raw_ranks_info_" + _type)) for _type in self._types }
        self.unk_positions: pd.DataFrame = None
        self.indiv_cols = {
            'QB': ['passing_yards', 'passing_touchdowns', 'rush_yards', 'rush_touchdowns', 'interceptions_thrown'],
            'RB': ['yards_from_scrimmage', 'total_touchdowns', 'receptions'],
            'WR': ['yards_from_scrimmage', 'total_touchdowns', 'receptions'],
            'TE': ['yards_from_scrimmage', 'total_touchdowns', 'receptions'],
        }
        self.extra_indiv_cols = {
            'QB': ['over_300_passing_yards', 'over_100_rush_yards'],
            'RB': ['over_100_rush_yards', 'over_100_receiving_yards'],
            'WR': ['over_100_rush_yards', 'over_100_receiving_yards'],
            'TE': ['over_100_rush_yards', 'over_100_receiving_yards']
        }
        self.json_cols = [
            'overallRanking', 'name', 'team',
            'position', 'projections', 'lastSeasonPoints',
            'positionRanking', 'flexRanking'
        ]
        self.flex_positions = ['RB', 'WR', 'TE']
        return
    def cleanRawRanks(self):
        for fn in os.listdir(self.data_dir):
            fn = fn.replace('.csv', '')
            df = pd.read_csv("%s.csv" % (self.data_dir + fn))
            df = df[['RK', 'PLAYER NAME', 'FAN PTS']]
            df.columns = ['rank', 'name', '2022_pts']
            df.insert(1, 'p_id', [self.getPid(name) for name in df['name'].values])
            self.saveFrame(df, (self.data_dir + fn.replace('raw', 'clean')))
        return
    def getPid(self, name: str):
        try:
            pid = self.names_df.loc[
                ((self.names_df['name'].str.contains(name))|
                (self.names_df['aka'].str.contains(name)))&
                (self.names_df['position'].isin(['QB', 'RB', 'WR', 'TE']))&
                (self.names_df['info'].str.contains('2022')),
                'p_id'
            ].values[0]
        except IndexError:
            print(f"Name ({name}) not found.")
            return 'UNK'
        return pid
    def combineRanks(self):
        """
        Combines all clean ranks, so all pids together for 2023
        """
        fns = [fn for fn in os.listdir(self.data_dir) if 'clean' in fn]
        df_list = []
        for fn in fns:
            df = pd.read_csv(self.data_dir + fn)
            df = df[['p_id', 'name']]
            df_list.append(df)
        new_df = pd.concat(df_list)
        new_df.drop_duplicates(inplace=True)
        self.saveFrame(new_df, (self.data_dir + 'all_ranks'))
        return
    def getAllPoints(self, pos: str, vals):
        """
        Returns standard, ppr, and half points
        Args:
            pos (str): position
            vals (np.array): stats

        Returns:
            tuple: (standard_points, ppr_points, half_points)
        """
        points = 0
        if pos == 'QB':
            points += vals[0]*0.04 # passing_yards
            points += vals[1]*4 # passing_touchdowns
            points += vals[2]*0.1 # rush_yards
            points += vals[3]*6 # rush_touchdowns
            points -= vals[4] # interceptions_thrown
            points += vals[5]*3 # over 300 passing
            points += vals[6]*3 # over 100 rush
            return points, points, points
        # skill positions
        points += vals[0]*0.1 # yards_from_scrimmage
        points += vals[1]*6 # total_touchdowns
        points += vals[2] # receptions
        points += vals[3]*3 # over 100 rush
        points += vals[4]*3 # over 100 receiving
        std = points - vals[2] # no points per reception
        ppr = points
        half = points - (vals[2]/2) # half per reception
        return std, ppr, half
    def indivProjToPoints(self):
        df_list = []
        for pos in self.indiv_pred_dirs:
            _dir = self.indiv_pred_dirs[pos]
            df = pd.read_csv("%s.csv" % (_dir + 'predictions'))
            cols = self.merge_cols + self.indiv_cols[pos] + self.extra_indiv_cols[pos]
            str_cols = self.merge_cols.copy()
            str_cols.insert(1, 'name')
            new_df = pd.DataFrame(columns=str_cols+['std_points', 'ppr_points', 'half_points'])
            for index, vals in enumerate(df[cols].values):
                self.printProgressBar(index, len(df.index), 'Indiv to Points')
                pid, year, position = vals[:3]
                num_vals = vals[3:]
                std, ppr, half = self.getAllPoints(pos, num_vals)
                name = self.names_df.loc[self.names_df['p_id']==pid, 'name'].values[0]
                new_df.loc[len(new_df.index)] = [pid, name, year, position, std, ppr, half]
            df_list.append(new_df)
        all_df = pd.concat(df_list)
        all_df.sort_values(by=['position', 'std_points'], ascending=False, inplace=True)
        all_df = all_df.round(2)
        self.saveFrame(all_df, (self._dir + "indiv_projections"))
        return
    def nameToPid_content(self, name, url):
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
    def filterPids(self, name, pids):
        f_name = name.split(" ")[0]
        l_name = name.split(" ")[1]
        pid_key = l_name[:4] + f_name[:2]
        try:
            pid = [pid for pid in pids if pid_key in pid][0]
        except IndexError:
            pid = 'UNK'
            print(f"{name} not found.")
        return pid
    def getUnkPids(self):
        self.setAllRanks()
        df = self.all_ranks.loc[self.all_ranks['p_id']=='UNK']
        new_df = pd.DataFrame(columns=['name', 'p_id'])
        for index, name in enumerate(df['name'].values):
            print(index, len(df.index), '\n')
            pfr_name = name.lower().replace(' ', '+')
            url = 'https://www.pro-football-reference.com/search/search.fcgi?hint=' + pfr_name + '&search=' + pfr_name
            pids = self.nameToPid_content(name, url)
            if len(pids) == 1:
                pid = pids[0]
            elif len(pids) > 1:
                pid = self.filterPids(name, pids)
            else:
                pid = 'UNK'
                print(f"{name} not found.")
            new_df.loc[len(new_df.index)] = [name, pid]
            time.sleep(2)
        self.saveFrame(new_df, (self.data_dir + 'unkPids'))
        return
    def getUnkPositions(self):
        self.setAllRanks()
        df = self.all_ranks.loc[self.all_ranks['p_id']=='UNK']
        info = pd.concat([self.info_frames[_type] for _type in self._types])
        new_df = pd.DataFrame(columns=['name', 'position'])
        for index, name in enumerate(df['name'].values):
            try:
                pos_key = info.loc[info['PLAYER NAME']==name, 'POS'].values[0]
                pos = re.findall(r"[A-Z]+", pos_key)[0]
            except IndexError:
                pos = input(f"{name} not found.")
            new_df.loc[len(new_df.index)] = [name, pos]
        self.saveFrame(new_df, (self.data_dir + 'unkPositions'))
        return
    def addPositionsToCleanFrames(self):
        self.setIndivFrame()
        self.setUnkPositions()
        for _type in self.clean_frames:
            df = self.clean_frames[_type]
            positions = []
            for pid, name in df[['p_id', 'name']].values:
                try:
                    if pid == 'UNK':
                        pos = self.unk_positions.loc[self.unk_positions['name']==name, 'position'].values[0]
                    else:
                        pos = self.indiv_frame.loc[self.indiv_frame['p_id']==pid, 'position'].values[0]
                except IndexError:
                    pos = input(f"{name} not found, enter position: ")
                positions.append(pos)
            df['position'] = positions
            self.saveFrame(df, (self.data_dir + 'clean_ranks_' + _type))
        return
    def addProjections_indiv(self):
        self.setIndivFrame()
        for _type in self.clean_frames:
            print(_type)
            df = self.clean_frames[_type]
            target_col = _type + "_points"
            projections = []
            for pid in df['p_id'].values:
                try:
                    points = self.indiv_frame.loc[self.indiv_frame['p_id']==pid, target_col].values[0]
                except IndexError:
                    points = 0
                projections.append(points)
            df['projections'] = projections
            self.saveFrame(df, (self.data_dir + "clean_ranks_" + _type))
        return
    def reRankCleanFrames(self, _type):
        df = self.clean_frames[_type]
        df['rank'] = df.index + 1
        self.saveFrame(df, (self.data_dir + "clean_ranks_" + _type))
        return
    def toJsonFrame(self):
        raw_df = pd.concat([self.raw_frames[_type] for _type in self.raw_frames])
        for _type in self.clean_frames:
            print(_type)
            df = self.clean_frames[_type]
            new_df = pd.DataFrame(columns=self.json_cols)
            for index, vals in enumerate(df[['rank', 'name', '2022_pts', 'position', 'projections']].values):
                self.printProgressBar(index, len(df.index), 'JSON Frame')
                rank, name, lastSeasonPoints, position, projections = vals
                # position rank
                temp_df: pd.DataFrame = df.loc[df['position']==position]
                temp_df.reset_index(drop=True, inplace=True)
                temp_df['position_rank'] = temp_df.index + 1
                position_rank = temp_df.loc[temp_df['name']==name, 'position_rank'].values[0]
                # flex_rank
                if position in self.flex_positions:
                    temp_df: pd.DataFrame = df.loc[df['position'].isin(self.flex_positions)]
                    temp_df.reset_index(drop=True, inplace=True)
                    temp_df['flex_rank'] = temp_df.index + 1
                    flex_rank = temp_df.loc[temp_df['name']==name, 'flex_rank'].values[0]
                else: # QB, K, DST - no flex rank
                    flex_rank = 0
                # 0.0 projections - fill with surrounding projections
                if projections == 0:
                    if position not in ['K', 'DST']:
                        temp_df: pd.DataFrame = df.loc[df['position']==position]
                        temp_df.reset_index(drop=True, inplace=True)
                        idx = temp_df.loc[temp_df['name']==name].index.values[0]
                        upper_points = temp_df.loc[temp_df.index==idx-1, 'projections'].values[0]
                        projections = random.uniform(upper_points-10, upper_points)
                    else:
                        idx = df.loc[df['name']==name].index.values[0]
                        upper_points = df.loc[df.index==idx-1, 'projections'].values[0]
                        projections = random.uniform(upper_points-10, upper_points)
                # team
                team = raw_df.loc[raw_df['PLAYER NAME'].str.contains(name), 'TEAM'].values[0]
                new_df.loc[len(new_df.index)] = [rank, name, team, position, projections, lastSeasonPoints, position_rank, flex_rank]
            self.saveFrame(new_df, (self.data_dir + "json_frame_" + _type))
        return
    def framesToJson(self):
        for _type in self._types:
            df = pd.read_csv("%s.csv" % (self.data_dir + 'json_frame_' + _type))
            _json = df.to_json(orient='records')
            with open((self.json_dir + 'jsonData_' + _type + '.json'), 'w') as f:
                json.dump(_json, f)
        return
    def setUnkPositions(self):
        self.unk_positions = pd.read_csv("%s.csv" % (self.data_dir + "unkPositions"))
        return
    def setAllRanks(self):
        self.all_ranks = pd.read_csv("%s.csv" % (self.data_dir + "all_ranks"))
        return
    def setIndivFrame(self):
        self.indiv_frame = pd.read_csv("%s.csv" % (self._dir + 'indiv_projections'))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def printProgressBar(self, iteration, total, prefix = 'Progress', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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
    
# / END Main
    
############################

m = Main('./')

# m.cleanRawRanks()

# m.combineRanks()

# m.indivProjToPoints()

# m.getUnkPids()

# m.getUnkPositions()

# m.addPositionsToCleanFrames()

# m.addProjections_indiv()

# m.reRankCleanFrames('std')

# m.toJsonFrame()

m.framesToJson()