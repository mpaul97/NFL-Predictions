import pandas as pd
import numpy as np
import os

def getK(movWinner, eloDifWinner):
    top = (movWinner + 3) ** 0.8
    bottom = 7.5 + (0.006 * (eloDifWinner))
    return 20 * (top/bottom)

def getETeam(oppElo, teamElo):
	dif = oppElo - teamElo
	x = 10 ** (dif/400)
	return 1 / (1 + x)

def getNewElo(k, sTeam, eTeam, teamElo):
	return k * (sTeam - eTeam) + teamElo

def findElo(teamElo, oppElo, sTeam, movWinner, eloDifWinner):
	k = getK(movWinner, eloDifWinner)
	eTeam = getETeam(oppElo, teamElo)
	return k * (sTeam - eTeam) + teamElo

def findEndElo(teamElo):
	if teamElo != 1500:
		return (teamElo * 0.75) + (0.25 * 1505)
	return 1500

#-----------

def getLastElo(df: pd.DataFrame, abbr):
    return df.loc[df['abbr']==abbr, 'elo'].values[-1]

def getElo(df: pd.DataFrame, team_abbr, opp_abbr, movWinner, winning_abbr):
    try:
        team_elo = getLastElo(df, team_abbr)
        opp_elo = getLastElo(df, opp_abbr)
        sTeam = 0
        eloDif = opp_elo - team_elo
        if team_abbr == winning_abbr:
            eloDif = team_elo - opp_elo
            sTeam = 1
        return findElo(team_elo, opp_elo, sTeam, movWinner, eloDif)
    except IndexError:
        return 1500
    return

def mergeElos(source: pd.DataFrame, df: pd.DataFrame):
    
    new_df = pd.DataFrame(columns=list(source.columns)+['home_coach_elo', 'away_coach_elo'])
    
    for index, row in source.iterrows():
        key = row['key']
        wy = row['wy']
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        start = df.loc[df['wy']==wy].index.values[0]
        home_elo = df.loc[(df['abbr']==home_abbr)&(df.index<start), 'elo'].values[-1]
        away_elo = df.loc[(df['abbr']==away_abbr)&(df.index<start), 'elo'].values[-1]
        new_df.loc[len(new_df.index)] = np.concatenate([row.values, np.array([home_elo, away_elo])])
    
    return new_df

def buildCoachElos(source: pd.DataFrame, cd: pd.DataFrame, _dir):
    
    df = pd.read_csv("%s.csv" % (_dir + "../../../../coaches/newCoaches"))
    
    # cd = cd.loc[(cd['wy'].str.contains('2018'))|(cd['wy'].str.contains('2019'))|(cd['wy'].str.contains('2020'))]
    # cd.reset_index(drop=True, inplace=True)
    
    if 'coachElos.csv' in os.listdir(_dir):
        print('coachElos.csv already built.')
        return
    
    if 'rawCoachElos.csv' not in os.listdir(_dir):
        
        new_df = pd.DataFrame(columns=['key', 'wy', 'abbr', 'elo'])
        
        print('Creating rawCoachElos.csv...')
        
        # init
        for abbr in set(list(cd['home_abbr'].values)):
            new_df.loc[len(new_df.index)] = ['INIT', '0', abbr, 1500]
        
        for index, row in cd.iterrows():
            key = row['key']
            wy = row['wy']
            home_abbr = row['home_abbr']
            away_abbr = row['away_abbr']
            home_points = row['home_points']
            away_points = row['away_points']
            # home winner, movWinner, winning_abbr
            winning_abbr = away_abbr
            movWinner = away_points - home_points
            if home_points > away_points:
                winning_abbr = home_abbr
                movWinner = home_points - away_points
            # ------------------------------------
            home_elo = getElo(new_df, home_abbr, away_abbr, movWinner, winning_abbr)
            away_elo = getElo(new_df, away_abbr, home_abbr, movWinner, winning_abbr)
            new_df.loc[len(new_df.index)] = [key, wy, home_abbr, home_elo]
            new_df.loc[len(new_df.index)] = [key, wy, away_abbr, away_elo]
            # calc end year
            try:
                year = int(wy.split(" | ")[1])
                nextYear = int((cd.loc[cd.index==index+1, 'wy'].values[0]).split(" | ")[1])
                if year != nextYear:
                    abbrs = set(list(new_df['abbr'].values))
                    for abbr in abbrs:
                        try:
                            isNewCoach = df.loc[(df['abbr']==abbr)&(df['year']==nextYear), 'isNewCoach'].values[0]
                        except IndexError:
                            isNewCoach = 0
                        last_elo = new_df.loc[new_df['abbr']==abbr, 'elo'].values[-1]
                        if isNewCoach == 1:
                            team_elo = 1500
                        else:
                            team_elo = findEndElo(last_elo)
                        new_df.loc[len(new_df.index)] = ['END', str(year), abbr, team_elo]
            except IndexError:
                abbrs = set(list(new_df['abbr'].values))
                for abbr in abbrs:
                    last_elo = new_df.loc[new_df['abbr']==abbr, 'elo'].values[-1]
                    team_elo = findEndElo(last_elo)
                    new_df.loc[len(new_df.index)] = ['END', str(year), abbr, team_elo]
                print('rawCoachElos created - End of data.')
            
        new_df.to_csv("%s.csv" % (_dir + "rawCoachElos"), index=False)
        
    else:
        
        print('Using existing rawCoachElos.csv...')
        new_df = pd.read_csv("%s.csv" % (_dir + "rawCoachElos"))
    
    # --------------------------------
    print('Joining rawCoachElos.csv...')
    
    final_df = mergeElos(source, new_df)
    
    final_df.to_csv("%s.csv" % (_dir + "coachElos"), index=False)
        
    return

def buildNewCoachElos(source: pd.DataFrame, _dir):
    
    print('Creating newCoachElos...')
    
    df = pd.read_csv("%s.csv" % (_dir + "rawCoachElos"))
    
    new_df = pd.DataFrame(columns=list(source.columns)+['home_coach_elo', 'away_coach_elo'])
    
    for index, row in source.iterrows():
        wy = row['wy']
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        if wy in df['wy'].values:
            if week != 1:
                start = df.loc[df['wy']==wy].index.values[0]
                home_elo = df.loc[(df['abbr']==home_abbr)&(df.index<start), 'elo'].values[-1]
                away_elo = df.loc[(df['abbr']==away_abbr)&(df.index<start), 'elo'].values[-1]
            else:
                start = df.loc[df['wy']==str(year)].index.values[0]
                home_elo = df.loc[(df['abbr']==home_abbr)&(df.index<start), 'elo'].values[-1]
                away_elo = df.loc[(df['abbr']==away_abbr)&(df.index<start), 'elo'].values[-1]
        else:
            home_elo = df.loc[df['abbr']==home_abbr, 'elo'].values[-1]
            away_elo = df.loc[df['abbr']==away_abbr, 'elo'].values[-1]
        new_df.loc[len(new_df.index)] = list(row.values) + [home_elo, away_elo]
    
    wy = source['wy'].values[0]
    new_df.to_csv("%s.csv" % (_dir + 'newCoachElos_' + wy.replace(' | ', '-')), index=False)
    
    return new_df

###############################

# source = pd.read_csv("%s.csv" % "../source/source")
# cd = pd.read_csv("%s.csv" % "../../../../data/gameData")

# buildCoachElos(source, cd, './')

# df = pd.read_csv("%s.csv" % "rawCoachElos")

# mergeElos(source, df)

# source = pd.read_csv("%s.csv" % "../source/new_source")

# buildNewTeamElos(source, './')