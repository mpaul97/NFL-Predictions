import pandas as pd
import numpy as np
import os
import regex as re
import random

class PenaltyObject:
    def __init__(self, _type: str, penalizer: str, yards: int, declined: bool, no_play: bool, offset: bool, against_possessing_team: bool):
        self._type = _type
        self.penalizer = penalizer
        self.yards = yards
        self.declined = declined
        self.no_play = no_play
        self.offset = offset
        self.against_possessing_team = against_possessing_team
        return
    def show(self):
        print(self.__dict__)
        return
    def __eq__(self, other):
        return self._type == other._type and self.penalizer == other.penalizer and self.yards == other.yards and self.no_play == other.no_play
    def __hash__(self):
        return hash(('_type', self._type, 'penalizer', self.penalizer, 'yards', self.yards, 'no_play', self.no_play))

# --------------------------------

ADF: pd.DataFrame = pd.read_csv("%s.csv" % "../data/allTables")[['primary_key', 'down', 'togo']]
ADF['at_index'] = ADF.index

class Penalties:
    def __init__(self):
        pass
    def save_frame(self, df: pd.DataFrame, name: str):
        df.to_csv("%s.csv" % name, index=False)
        return
    def save_types(self):
        """
        Get every penalty type and save to csv - add penalty info manually
        """
        df = pd.read_csv("%s.csv" % "../data/allTables_pids")
        df.dropna(inplace=True)
        df = df.loc[df['pids_detail'].str.contains('Penalty')]
        all_lines: list[str] = list(df['pids_detail'].values)
        all_types = []
        for line in all_lines:
            line = line.replace('),', ').')
            sentences = line.split(". ")
            for s in sentences:
                if 'Penalty' in s:
                    arr = s.split(":")
                    try:
                        _type = (arr[1].split(',')[0]).lstrip().rstrip()
                        _type = re.sub(r"Penalty\son\s[A-Z]{3}", '', _type)
                        _type = re.sub(r'\([^)]*\)', '', _type).rstrip()
                        all_types.append(_type)
                    except IndexError:
                        continue
        new_df = pd.DataFrame()
        new_df['type'] = list(set(all_types))
        new_df.sort_values(by=['type'], inplace=True)
        self.save_frame(new_df, "./data/penalty_types")
        return
    def penalty_sentences_regex_testing(self):
        """
        Save all penalty sentences for analysis
        """
        df = pd.read_csv("%s.csv" % "../data/allTables_pids")
        df.dropna(inplace=True)
        df = df.loc[df['pids_detail'].str.contains('Penalty')]
        all_lines: list[str] = list(df['pids_detail'].values)
        for line in all_lines[:100]:
            # convert comma seperated penalties to period seperated
            line = re.sub(r"\,\sPenalty", '. Penalty', line)
            # get substrings starting with Penalty and ending with period or end of sentence
            pens = re.findall(r"Penalty.*?\.\s|\bPenalty.*$", line)
            # found penalties do not match Penalty count
            if len(pens) != line.count('Penalty'):
                print(line)
                print(pens)
                print()
        return
    def get(self, row: pd.Series):
        """
        Get each penalty_type, penalizer (team_abbr or pid), penalty_yards, accepted, and is_no_play
        Args:
            row (pd.Series): pids_detail
        """
        at_index, line, togo = row[['at_index', 'pids_detail', 'togo']]
        # convert comma seperated penalties to period seperated
        line = re.sub(r"\,\sPenalty", '. Penalty', line)
        # convert [) + space] seperated penalties to period seperated
        line = re.sub(r"\)\sPenalty", '). Penalty', line)
        # get substrings starting with Penalty and ending with period or end of sentence
        pens: list[str] = re.findall(r"Penalty.*?\.\s|\bPenalty.*$", line)
        objs = []
        for p in pens:
            try:
                _type = (re.findall(r":\s.*?[,(]|\b:\s.*$", p)[0]).replace(':','').replace(',','').replace('(','').replace('.','').lstrip().rstrip()
                penalizer = ([s for s in p.split(' ') if ':' in s][0]).replace(':','')
                yards = 0
                if '1 yard' in p: # 1 yard (NOT YARDS)
                    yards = 1
                if 'yards' in p:
                    yards = int((re.findall(r"[0-9]{1,2}\syards", p)[0]).replace(' yards', ''))
                declined = ('declined' in p.lower())
                no_play = ('(no play)' in p)
                offset = ('(offset)' in p or '(Offsetting)' in p)
                try:
                    next_togo = ADF.loc[ADF.index==at_index+1, 'togo'].values[0]
                    against_possessing_team = False
                    if not pd.isna(togo) and not pd.isna(next_togo):
                        against_possessing_team = (next_togo > togo)
                except IndexError:
                    against_possessing_team = False
                objs.append(PenaltyObject(_type, penalizer, yards, declined, no_play, offset, against_possessing_team))
            except Exception as e:
                print(f"Error: {e} for line: {line}")
        return objs

###########################

# Penalties().penalty_sentences_regex_testing()

# df = pd.read_csv("%s.csv" % "../data/allTables_pids")
# df = df.merge(ADF, on=['primary_key'])
# # df.dropna(inplace=True)
# df = df.loc[df['primary_key']=='201909220buf-33']
# print(df)

# all_types = []

# for index, row in df.iterrows():
#     pens: list[PenaltyObject] = Penalties().get(row)
#     print(row['pids_detail'])
#     [p.show() for p in pens]
#     print()
#     [all_types.append(p._type) for p in pens]
    
# all_types = list(set(all_types))
# all_types.sort()

# for t in all_types:
#     print(t)

    