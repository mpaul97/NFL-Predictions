import pandas as pd
import numpy as np
import os
import regex as re
import datetime
from ordered_set import OrderedSet
import random
import itertools

pd.options.mode.chained_assignment = None

class MockTiers:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir_22 = _dir + 'data_2022/'
        self.data_dir = _dir + 'data/'
        self.info_dir = _dir + 'info/'
        # info
        self.positions = ['QB', 'RB', 'WR', 'TE']
        self.skill_positions = ['RB', 'WR', 'TE']
        self._types = ['short', 'long']
        self.tiers = {
            'short': [('T' + str(i)) for i in range(1, 7)],
            'long': [('T' + str(i)) for i in range(1, 9)]
        }
        self.source_cols = ['wy', 'p_id', 'pos_type', 'position', 'tier']
        # frames
        self.names_df = pd.read_csv("%s.csv" % (self._dir + "../playerNames/finalPlayerInfo"))
        self.gd: pd.DataFrame = pd.read_csv("%s.csv" % (_dir + "../data/gameData_regOnly"))
        self.fdf: pd.DataFrame = pd.read_csv("%s.csv" % (_dir + "../data/fantasyData"))
        self.fdf: pd.DataFrame = self.addDatetimeColumns(self.fdf)
        self.sr: pd.DataFrame = None
        self.long_tiers: pd.DataFrame = None
        self.short_tiers: pd.DataFrame = None
        self.short_info_frames: dict = {}
        self.long_info_frames: dict = {}
        self.tiers_info: dict = {}
        self.source_frames: dict = {}
        return
    def most_common(self, lst: list):
        return max(set(lst), key=lst.count)
    def getDatetime(self, week: int, year: int):
        return datetime.datetime.strptime(f'{year}-W{week}-1', "%Y-W%W-%w")
    def addDatetimeColumns(self, df: pd.DataFrame):
        df['week'] = [int(wy.split(" | ")[0]) for wy in df['wy'].values]
        df['year'] = [int(wy.split(" | ")[1]) for wy in df['wy'].values]
        df['datetime'] = [self.getDatetime(week, year) for week, year in df[['week', 'year']].values]
        return df
    def combine(self):
        fns = os.listdir(self.data_dir_22)
        long_tiers, short_tiers = [], []
        for fn in fns:
            wy = (re.findall(r"[0-9]{2}-[0-9]{4}", fn)[0]).replace("-", " | ")
            # removing leading 0 for single digit week
            wy = wy[1:] if wy[0] == '0' else wy
            df = pd.read_csv(self.data_dir_22 + fn)
            key = fn.replace('.csv','')
            df = df[['Position', 'Name', 'Roster Position', 'AvgPointsPerGame']]
            df.columns = ['position', 'name', 'tier', 'points_per_game']
            df.insert(0, 'p_id', [self.getPid(row['name'], row['position']) for _, row in df.iterrows()])
            df.drop(columns=['name'], inplace=True)
            df.insert(0, 'key', key)
            df.insert(1, 'wy', wy)
            self.fdf = self.fdf[['p_id', 'wy', 'points', 'week_rank']]
            df = df.merge(self.fdf, on=['p_id', 'wy'])
            long_tiers.append(df) if len(set(df['tier'])) == 8 else short_tiers.append(df)
        long_df = pd.concat(long_tiers)
        short_df = pd.concat(short_tiers)
        self.saveFrame(long_df, (self.info_dir + 'long_tiers'))
        self.saveFrame(short_df, (self.info_dir + 'short_tiers'))
        return
    def getPid(self, name: str, position: str):
        try:
            pid = self.names_df.loc[
                ((self.names_df['name'].str.contains(name))|
                (self.names_df['aka'].str.contains(name)))&
                (self.names_df['position']==position), 
                'p_id'
            ].values[0]
        except IndexError:
            print(f"Name ({name}) not found.")
            return None
        return pid
    def createSeasonRanks(self):
        """
        Creates season ranks per week for each player
        """
        df = self.fdf
        new_df = pd.DataFrame(columns=['p_id', 'position', 'wy', 'season_points'])
        for index, (pid, position, week, year, datetime) in enumerate(df[['p_id', 'position', 'week', 'year', 'datetime']].values):
            self.printProgressBar(index, len(df.index), 'Season Points')
            if week == 1:
                stats = df.loc[(df['p_id']==pid)&(df['year']==(year-1)), 'points'].values
                total = sum(stats)
            else:
                stats = df.loc[(df['p_id']==pid)&(df['datetime']<datetime)&(df['year']==year), 'points'].values
                total = sum(stats)
            new_df.loc[len(new_df.index)] = [pid, position, (str(week) + " | "  + str(year)), total]
        # add ranks
        df_list = []
        wys = list(OrderedSet(new_df['wy'].values))
        for index, wy in enumerate(wys):
            self.printProgressBar(index, len(wys), 'Season Ranks')
            for position in self.positions:
                temp_df: pd.DataFrame = new_df.loc[(new_df['wy']==wy)&(new_df['position']==position)]
                temp_df.sort_values(by=['season_points'], ascending=False, inplace=True)
                temp_df.reset_index(drop=True, inplace=True)
                temp_df['season_rank'] = temp_df.index
                df_list.append(temp_df)
        new_df = pd.concat(df_list)
        self.saveFrame(new_df, 'seasonRanks')
        return
    def createInfo_currRanks(self):
        """
        Get current week_rank and season rank for each played in tiers
        """
        cd = self.fdf
        self.setSeasonRanks()
        self.sr = self.addDatetimeColumns(self.sr)
        self.setShortTiers()
        self.setLongTiers()
        for df, _type  in [(self.short_tiers, 'short'), (self.long_tiers, 'long')]:
            df = self.addDatetimeColumns(df)
            new_df = pd.DataFrame(columns=['key', 'p_id', 'wy', 'position', 'tier', 'week_rank', 'season_rank'])
            for index, (key, pid, wy, position, tier, datetime) in enumerate(df[['key', 'p_id', 'wy', 'position', 'tier', 'datetime']].values):
                self.printProgressBar(index, len(df.index), 'Target')
                w_rank = cd.loc[(cd['p_id']==pid)&(cd['datetime']==datetime), 'week_rank'].values[0]
                s_rank = self.sr.loc[(self.sr['p_id']==pid)&(self.sr['datetime']==datetime), 'season_rank'].values[0]
                new_df.loc[len(new_df.index)] = [key, pid, wy, position, tier, w_rank, s_rank]
            tiers = list(set(df['tier'].values))
            for tier in tiers:
                temp_df = new_df.loc[new_df['tier']==tier]
                self.saveFrame(temp_df, (self.info_dir + _type + '_info_' + tier))
        return
    def createTiersInfo(self):
        """
        Creates long and short tiers info; tier size, min, and max season rank
        """
        self.setAllTierInfoFrames()
        for _type in self._types:
            new_df = pd.DataFrame(columns=['tier', 'position', 'size', 'min_rank', 'max_rank'])
            tiers = self.tiers[_type]
            for tier in tiers:
                print(_type, tier)
                df: pd.DataFrame = self.short_info_frames[tier] if _type == 'short' else self.long_info_frames[tier]
                keys = list(OrderedSet(df['key'].values))
                all_sizes, all_mins, all_maxes = [], [], []
                for key in keys:
                    temp_df: pd.DataFrame = df.loc[df['key']==key]
                    position = 'qb' if list(set(temp_df['position'].values))[0] == 'QB' else 'skill'
                    size = len(temp_df.index)
                    min_rank = min(temp_df['season_rank'].values)
                    max_rank = max(temp_df['season_rank'].values)
                    all_sizes.append(size)
                    all_mins.append(min_rank)
                    all_maxes.append(max_rank)
                new_df.loc[len(new_df.index)] = [tier, position, self.most_common(all_sizes), min(all_mins), int(np.mean(all_maxes))]
            self.saveFrame(new_df, (self.info_dir + _type + '_tiersInfo'))
        return
    def getMinMax_skills(self, all_counts: dict, position: str):
        lst = all_counts[position]
        if len(lst) != 0:
            return min(lst), max(lst)
        return np.nan, np.nan
    def createSkillsInfo(self):
        self.setAllTierInfoFrames()
        for _type in self._types:
            new_df = pd.DataFrame(
                columns=[
                    'tier', 'position', 'min_rb', 
                    'max_rb', 'min_wr', 'max_wr',
                    'min_te', 'max_te'
                ]
            )
            tiers = self.tiers[_type]
            for tier in tiers:
                print(_type, tier)
                df: pd.DataFrame = self.short_info_frames[tier] if _type == 'short' else self.long_info_frames[tier]
                keys = list(OrderedSet(df['key'].values))
                all_counts = { pos: [] for pos in self.skill_positions }
                for key in keys:
                    temp_df: pd.DataFrame = df.loc[df['key']==key]
                    if list(set(temp_df['position'].values))[0] != 'QB':
                        position = 'skill'
                        counts = temp_df['position'].value_counts()
                        for pos in self.skill_positions:
                            try:
                                all_counts[pos].append(counts.loc[counts.index==pos].values[0])
                            except IndexError:
                                all_counts[pos].append(0)
                    else:
                        position = 'qb'
                r_min, r_max = self.getMinMax_skills(all_counts, 'RB')
                w_min, w_max = self.getMinMax_skills(all_counts, 'WR')
                t_min, t_max = self.getMinMax_skills(all_counts, 'TE')
                new_df.loc[len(new_df.index)] = [
                    tier, position, r_min,
                    r_max, w_min, w_max,
                    t_min, t_max
                ]
            new_df = new_df.round(2)
            self.saveFrame(new_df, (self.info_dir + _type + "_skillsInfo"))
        return
    def getRandRanks(self, used: list, min_rank, max_rank, size):
        if len(used) == 0:
            rands = random.sample(range(min_rank, max_rank), size)
            [used.append(r) for r in rands]
        else:
            rands = []
            while len(rands) < size:
                r = random.randrange(min_rank, max_rank)
                if r not in used and r not in rands:
                    rands.append(r)
                    used.append(r)
        return used, rands
    def getSkillSizes(self, arr: list, size):
        [arr.remove(a) for a in arr if len(a) == 1 and a[0] == 0]
        combos = [c for c in list(itertools.product(*arr)) if sum(c) == size]
        return combos[random.randrange(0, len(combos))]
    def createMockTiers(self):
        """
        Creates source/mock tiers
        """
        self.setAllInfo()
        self.setSeasonRanks()
        cd = self.sr
        df = self.gd
        df = self.addDatetimeColumns(df)
        df = df.loc[df['datetime']>=self.getDatetime(1, 2000)]
        wys = list(OrderedSet(df['wy'].values))
        for _type in self._types:
            print(_type)
            new_df = pd.DataFrame(columns=['wy', 'p_id', 'pos_type', 'position', 'tier'])
            t_info: pd.DataFrame = self.tiers_info[_type]['tiers_info']
            s_info: pd.DataFrame = self.tiers_info[_type]['skills_info']
            for index, wy in enumerate(wys):
                self.printProgressBar(index, len(wys), 'Mock Tiers')
                pdf: pd.DataFrame = cd.loc[cd['wy']==wy]
                used_vals = { pos: [] for pos in self.positions }
                for index, row in t_info.iterrows():
                    tier, pos_type, size, min_rank, max_rank = row[['tier', 'position', 'size', 'min_rank', 'max_rank']]
                    if pos_type == 'qb':
                        temp_df: pd.DataFrame = pdf.loc[pdf['position']=='QB']
                        max_rank = max_rank if tier != 'T1' else (size + 3)
                        used_vals['QB'], rands = self.getRandRanks(used_vals['QB'], min_rank, max_rank, size)
                        data = temp_df.loc[temp_df['season_rank'].isin(rands), ['p_id', 'position']].values
                        for pid, position in data:
                            new_df.loc[len(new_df.index)] = [wy, pid, pos_type, position, tier]
                    else:
                        vals = s_info.loc[s_info['tier']==tier].values[0, 2:]
                        pos_sizes = [(int(vals[i]), int(vals[i+1])) for i in range(0, len(vals)-1, 2)]
                        possible_sizes = [[i for i in range(min_size, max_size+1)] for min_size, max_size in pos_sizes]
                        skill_sizes = self.getSkillSizes(possible_sizes, size)
                        for i in range(len(skill_sizes)):
                            pos = self.skill_positions[i]
                            pos_size = skill_sizes[i]
                            temp_df: pd.DataFrame = pdf.loc[pdf['position']==pos]
                            num_tier = int(tier.replace('T',''))
                            max_rank = size*num_tier
                            max_rank = max_rank if pos != 'TE' else 15 # top 15 for TEs, no more than 11 used
                            used_vals[pos], rands = self.getRandRanks(used_vals[pos], min_rank, max_rank, pos_size)
                            # print(tier, pos, rands, max_rank)
                            data = temp_df.loc[temp_df['season_rank'].isin(rands), ['p_id', 'position']].values
                            for pid, position in data:
                                new_df.loc[len(new_df.index)] = [wy, pid, pos_type, position, tier]
            self.saveFrame(new_df, (self.data_dir + _type + '_mockTiers'))
        return
    def createTargets(self):
        self.setSourceFrames()
        for _type in self._types:
            tiers = self.tiers[_type]
            df: pd.DataFrame = self.source_frames[_type]
            wys = list(OrderedSet(df['wy'].values))
            df_list = []
            for index, wy in enumerate(wys):
                self.printProgressBar(index, len(wys), 'Targets')
                for tier in tiers:
                    temp_df: pd.DataFrame = df.loc[(df['wy']==wy)&(df['tier']==tier), ['wy', 'p_id']]
                    temp_df = temp_df.merge(self.fdf, on=['wy', 'p_id'])
                    temp_df = temp_df[['wy', 'p_id', 'points']]
                    temp_df.sort_values(by=['points'], ascending=False, inplace=True)
                    temp_df.reset_index(drop=True, inplace=True)
                    temp_df['tier_rank'] = temp_df.index
                    temp_df.drop(columns=['points'], inplace=True)
                    df_list.append(temp_df)
            new_df = pd.concat(df_list)
            df = df.merge(new_df, on=['wy', 'p_id'])
            self.saveFrame(df, (self.data_dir + _type + '_target'))
        return
    def concat2022Tiers(self):
        self.setLongTiers()
        print(self.long_tiers)
        return
    def setSourceFrames(self):
        for _type in self._types:
            self.source_frames[_type] = pd.read_csv("%s.csv" % (self.data_dir + _type + '_mockTiers'))
        return
    def setAllInfo(self):
        self.tiers_info = {
            _type: {
                'tiers_info': pd.read_csv("%s.csv" % (self.info_dir + _type + "_tiersInfo")),
                'skills_info': pd.read_csv("%s.csv" % (self.info_dir + _type + "_skillsInfo"))
            }
            for _type in self._types
        }
        return
    def setAllTierInfoFrames(self):
        self.short_info_frames = {tier: pd.read_csv("%s.csv" % (self.info_dir + 'short_info_' + tier)) for tier in self.tiers['short']}
        self.long_info_frames = {tier: pd.read_csv("%s.csv" % (self.info_dir + 'long_info_' + tier)) for tier in self.tiers['long']}
        return
    def setSeasonRanks(self):
        self.sr = pd.read_csv("%s.csv" % (self._dir + "seasonRanks"))
        return
    def setLongTiers(self):
        self.long_tiers = pd.read_csv("%s.csv" % (self.info_dir + "long_tiers"))
        return
    def setShortTiers(self):
        self.short_tiers = pd.read_csv("%s.csv" % (self.info_dir + "short_tiers"))
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
    
# END / MockTiers
    
########################

mt = MockTiers('./')

# mt.combine()

# mt.createSeasonRanks()

# mt.createInfo_currRanks()

# mt.createTiersInfo()

# mt.createSkillsInfo()

# mt.createMockTiers() 

# mt.createTargets() # targets rank of players in that tier

mt.concat2022Tiers()