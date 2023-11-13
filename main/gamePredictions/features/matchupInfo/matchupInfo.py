import pandas as pd
import numpy as np
import os

pd.options.mode.chained_assignment = None

def buildMatchupInfo(source: pd.DataFrame, cd: pd.DataFrame, sl: pd.DataFrame, _dir):
    if 'matchupInfo.csv' in os.listdir(_dir):
        print('Using existing matchInfo.')
        return
    print('Creating matchupInfo...')
    div_dir = _dir + '../pred_standings/divisionData/'
    div_78 = pd.read_csv("%s.csv" % (div_dir + "divisions_78-01"))
    div_02 = pd.read_csv("%s.csv" % (div_dir + "divisions_02-22"))
    isRival_div, isRival_conf = [], [] # 1 = same div, same conf
    isPlayoffs, isWeekBeforePlayoffs = [], []
    homeWonLast, homeWLPs, awayWLPs = [], [], []
    for index, row in source.iterrows():
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        div_df = div_78 if year <= 2001 else div_02
        info = div_df.loc[div_df['year']==year].values[0][1:]
        conf_info = info[-2:]
        div_info = info[:-2] # only divisions
        same_div, same_conf = 0, 0
        for div in div_info:
            if home_abbr in div and away_abbr in div:
                same_div = 1
                break
        for conf in conf_info:
            if home_abbr in conf and away_abbr in conf:
                same_conf = 1
                break
        isRival_div.append(same_div)
        isRival_conf.append(same_conf)
        # playoff info - season length
        weeks = sl.loc[sl['year']==year, 'weeks'].values[0]
        isPlayoffs.append(1 if week > weeks else 0)
        isWeekBeforePlayoffs.append(1 if week == weeks else 0)
        # past matchups
        start = cd.loc[cd['wy']==wy].index.values[0]
        stats = cd.loc[
            (cd.index<start)&
            (
                ((cd['home_abbr']==home_abbr)&(cd['away_abbr']==away_abbr))|
                ((cd['away_abbr']==home_abbr)&(cd['home_abbr']==away_abbr))
            ),
            'winning_abbr'
        ].values
        try:
            last_winner = stats[-1]
            homeWonLast.append(1 if last_winner == home_abbr else 0)
            home_won_count = list(stats).count(home_abbr)
            away_won_count = list(stats).count(away_abbr)
            home_wlp = home_won_count/len(stats)
            away_wlp = away_won_count/len(stats)
            homeWLPs.append(home_wlp)
            awayWLPs.append(away_wlp)
        except IndexError:
            homeWonLast.append(1)
            homeWLPs.append(0.5)
            awayWLPs.append(0.5)
    source['isDivisionRival'] = isRival_div
    source['isConferenceRival'] = isRival_conf
    source['isPlayoffs'] = isPlayoffs
    source['isWeekBeforePlayoffs'] = isWeekBeforePlayoffs
    source['homeWonLastMatchup'] = homeWonLast
    source['home_opp_wlp'] = homeWLPs
    source['away_opp_wlp'] = awayWLPs
    source.to_csv("%s.csv" % (_dir + "matchupInfo"), index=False)
    return

def buildNewMatchupInfo(source: pd.DataFrame, cd: pd.DataFrame, sl: pd.DataFrame, _dir):
    print('Creating new matchupInfo...')
    div_dir = _dir + '../pred_standings/divisionData/'
    div_78 = pd.read_csv("%s.csv" % (div_dir + "divisions_78-01"))
    div_02 = pd.read_csv("%s.csv" % (div_dir + "divisions_02-22"))
    isRival_div, isRival_conf = [], [] # 1 = same div, same conf
    isPlayoffs, isWeekBeforePlayoffs = [], []
    homeWonLast, homeWLPs, awayWLPs = [], [], []
    for index, row in source.iterrows():
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        div_df = div_78 if year <= 2001 else div_02
        info = div_df.loc[div_df['year']==year].values[0][1:]
        conf_info = info[-2:]
        div_info = info[:-2] # only divisions
        same_div, same_conf = 0, 0
        for div in div_info:
            if home_abbr in div and away_abbr in div:
                same_div = 1
                break
        for conf in conf_info:
            if home_abbr in conf and away_abbr in conf:
                same_conf = 1
                break
        isRival_div.append(same_div)
        isRival_conf.append(same_conf)
        # playoff info - season length
        weeks = sl.loc[sl['year']==year, 'weeks'].values[0]
        isPlayoffs.append(1 if week > weeks else 0)
        isWeekBeforePlayoffs.append(1 if week == weeks else 0)
        # past matchups
        if wy in cd['wy'].values:
            start = cd.loc[cd['wy']==wy].index.values[0]
            stats = cd.loc[
                (cd.index<start)&
                (
                    ((cd['home_abbr']==home_abbr)&(cd['away_abbr']==away_abbr))|
                    ((cd['away_abbr']==home_abbr)&(cd['home_abbr']==away_abbr))
                ),
                'winning_abbr'
            ].values
        else:
            stats = cd.loc[
                (
                    ((cd['home_abbr']==home_abbr)&(cd['away_abbr']==away_abbr))|
                    ((cd['away_abbr']==home_abbr)&(cd['home_abbr']==away_abbr))
                ),
                'winning_abbr'
            ].values
        try:
            last_winner = stats[-1]
            homeWonLast.append(1 if last_winner == home_abbr else 0)
            home_won_count = list(stats).count(home_abbr)
            away_won_count = list(stats).count(away_abbr)
            home_wlp = home_won_count/len(stats)
            away_wlp = away_won_count/len(stats)
            homeWLPs.append(home_wlp)
            awayWLPs.append(away_wlp)
        except IndexError:
            homeWonLast.append(1)
            homeWLPs.append(0.5)
            awayWLPs.append(0.5)
    source['isDivisionRival'] = isRival_div
    source['isConferenceRival'] = isRival_conf
    source['isPlayoffs'] = isPlayoffs
    source['isWeekBeforePlayoffs'] = isWeekBeforePlayoffs
    source['homeWonLastMatchup'] = homeWonLast
    source['home_opp_wlp'] = homeWLPs
    source['away_opp_wlp'] = awayWLPs
    
    wy = source['wy'].values[0]
    source.to_csv("%s.csv" % (_dir + "newMatchupInfo"), index=False)
    
    return source

##############################

# source = pd.read_csv("%s.csv" % "../source/new_source")
# cd = pd.read_csv("%s.csv" % "../../../../data/gameData")
# sl = pd.read_csv("%s.csv" % "../../../../data/seasonLength")

# buildNewMatchupInfo(source, cd, sl, './')