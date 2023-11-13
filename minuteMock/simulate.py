import pandas as pd
import numpy as np
import os
import random

class Need:
    def __init__(self, position: str, weight: float):
        self.position = position
        self.weight = weight
        return
    def show(self):
        print(vars(self))
        return

class Player:
    def __init__(
        self, overallRanking: int, name: str,
        team: str, position: str, projections: float,
        lastSeasonPoints: float, positionRanking: int, flexRanking: int
    ):
        self.overallRanking = overallRanking
        self.name = name
        self.team = team
        self.position = position
        self.projections = projections
        self.lastSeasonPoints = lastSeasonPoints
        self.positionRanking = positionRanking
        self.flexRanking = flexRanking
        return
    def show(self):
        print(vars(self))
        return

class Simulate:
    def __init__(self, _dir):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.sims_dir = self.data_dir + "sims/"
        self.json_dir = self._dir + "../NFL_fantasyProjections/data/"
        self._types = ['std', 'ppr', 'half']
        self.frames = { t: pd.read_csv("%s.csv" % (self.json_dir + "json_frame_" + t)) for t in self._types }
        self.flex_positions = ['RB', 'WR', 'TE']
        # sim options
        self.league_size: int = 0
        self.position_sizes: dict = {}
        self.df: pd.DataFrame = None
        return
    def flatten(self, l: list[list]):
        return [item for sublist in l for item in sublist]
    def initTeams(self):
        teams = {}
        for i in range(1, (self.league_size + 1)):
            p_obj = {}
            for key in self.position_sizes:
                p_obj[key] = ['' for _ in range(self.position_sizes[key])]
            teams[i] = p_obj
        return teams
    def initNeeds(self):
        needs = {}
        for i in range(1, (self.league_size + 1)):
            needs[i] = [
                Need('QB', random.uniform(0.1, 0.3)),
                Need('RB', random.uniform(0.24, 0.57)),
                Need('WR', random.uniform(0.2, 0.5)),
                Need('TE', random.uniform(0.1, 0.25)),
                Need('K', random.uniform(0.02, 0.12)),
                Need('DST', random.uniform(0.02, 0.15))
            ]
        return needs
    def getQueueArr(self):
        queueArr = []
        total_size = sum([self.position_sizes[pos] for pos in self.position_sizes])
        for round in range(1, (total_size + 1)):
            if round % 2 != 0:
                for i in range(1, (self.league_size + 1)):
                    queueArr.append((round, i))
            else:
                for i in range(self.league_size, 0, -1):
                    queueArr.append((round, i))
        return queueArr
    def nameToPlayer(self, name: str):
        vals = list(self.df.loc[self.df['name']==name].values[0])
        return Player(*vals)
    # computer funcs
    def getPlayerR1_Top3(self):
        temp_df: pd.DataFrame = self.df.head(3)
        name = random.choices(temp_df['name'].values, [0.7, 0.2, 0.1])[0]
        return self.nameToPlayer(name)
    def getPlayerR1(self):
        temp_df: pd.DataFrame = self.df.head(10)
        name = random.choices(temp_df['name'].values, [0.3, 0.2, 0.115, 0.1, 0.085, 0.065, 0.055, 0.045, 0.025, 0.01])[0]
        return self.nameToPlayer(name)
    def getMaxNeed(self, needs: list[Need]):
        return [n.position for n in needs if n.weight == max([n.weight for n in needs])][0]
    def getPlayerRest(self, needs: list[Need]):
        pos = self.getMaxNeed(needs)
        name = self.df.loc[self.df['position']==pos, 'name'].values[0]
        return self.nameToPlayer(name)
    def getPlayer(self, needs: list[Need], round: int, currDrafter: int):
        if round == 1:
            if currDrafter <= 3:
                return self.getPlayerR1_Top3()
            else:
                return self.getPlayerR1()
        return self.getPlayerRest(needs)
    # team funcs
    def insertPlayer(self, team: dict, position: str, player: Player):
        for i in range(len(team[position])):
            if team[position][i] == '':
                team[position][i] = player.name
                break
        return
    def addPlayer(self, team: dict, player: Player):
        if '' in team[player.position]:
            self.insertPlayer(team, player.position, player)
        elif '' in team['FLEX'] and player.position in self.flex_positions:
            self.insertPlayer(team, 'FLEX', player)
        elif '' in team['BEN'] and player.position:
            self.insertPlayer(team, 'BEN', player)
        else:
            return False
        return True
    def getMissingStarters(self, team: dict):
        return [(pos, team[pos].count('')) for pos in team]
    def updateNeeds(self, team: dict, needs: list[Need], selectedPosition: str, round: int):
        earlyRoundPositions = ['QB', 'RB', 'WR', 'TE'];
        for i in range(len(needs)):
            n_position, n_weight = needs[i].position, needs[i].weight
            if n_position == selectedPosition:
                needs[i] = Need(n_position, (n_weight - random.uniform(0, 0.1)))
            else:
                if round <= 5:
                    if n_position in earlyRoundPositions:
                        needs[i] = Need(n_position, (n_weight + random.uniform(0, 0.1)))
                else:
                    needs[i] = Need(n_position, (n_weight + random.uniform(0, 0.1)))
        if '' not in team['BEN']:
            # set last selected position weight to 0
            idx = [i for i in range(len(needs)) if needs[i].position == selectedPosition][0]
            needs[idx] = Need(selectedPosition, 0)
            missing = self.getMissingStarters(team)
            for pos, count in missing:
                if count != 0:
                    idx = [i for i in range(len(needs)) if needs[i].position == pos][0]
                    needs[idx] = Need(pos, 5)
        return needs
    def simulate(self, league_size: int, league_type: int, position_sizes: dict):
        self.league_size, self.position_sizes = league_size, position_sizes
        self.df = self.frames[league_type]
        allTeams = self.initTeams()
        allNeeds = self.initNeeds()
        queueArr = self.getQueueArr()
        t_cols = [(pos.lower() + '_' + str(i)) for pos in self.position_sizes for i in range(1, (self.position_sizes[pos] + 1))]
        df = pd.DataFrame(columns=['drafter', 'round', 'name'] + t_cols)
        for round, currDrafter in queueArr:
            team, needs = allTeams[currDrafter], allNeeds[currDrafter]
            player: Player = self.getPlayer(needs, round, currDrafter)
            self.df = self.df.loc[self.df['name']!=player.name]
            added: bool = self.addPlayer(team, player)
            allNeeds[currDrafter] = self.updateNeeds(team, needs, player.position, round)
            team_vals = self.flatten([team[pos] for pos in team])
            df.loc[len(df.index)] = [currDrafter, round, player.name] + team_vals
        fns = [fn for fn in os.listdir(self.sims_dir) if 'sim' in fn]
        if len(fns) == 0:
            self.saveFrame(df, (self.sims_dir + 'sim_' + league_type + '_0'))
        else:
            max_num = max([int(fn.replace('sim_','').replace((league_type + '_'), '').replace('.csv','')) for fn in fns])
            self.saveFrame(df, (self.sims_dir + 'sim_' + league_type + '_' + str(max_num + 1)))
        return
    def saveFrame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    
# / END Main

##########################

s = Simulate("./")

s.simulate(
    league_size=8, 
    league_type='std',
    position_sizes={
        'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1,
        'FLEX': 1, 'K': 1, 'DST': 1, 'BEN': 7
    }
)