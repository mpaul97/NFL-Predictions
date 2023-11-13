import pandas as pd
import numpy as np
import os

# try:
#     from features.pred_standings.StandingClasses import Standing
# except ModuleNotFoundError:
#     print('tiebreakerAttributes - Using local imports.')
#     from StandingClasses import Standing
from gamePredictions.features.pred_standings.StandingClasses import Standing

def isRegularSeason(week, year, sl):
    seasonWeeks = sl.loc[sl['year']==year, 'weeks'].values[0]
    if week <= seasonWeeks + 1:
        return True
    return False

def objInStandings(abbr, year, standings):
    objKey = abbr + "-" + str(year)
    if objKey in [s.getObjKey() for s in standings]:
        return True
    return False

def getStanding(abbr, year, standings):
    return [s for s in standings if s.abbr == abbr and s.year == year][0]

def getOppWl(abbr, year, standings):
    temp = [s for s in standings if s.abbr == abbr and s.year == year]
    if len(temp) > 0:
        return temp[0].wl
    return 0

def showStandings(standings, abbrs):
    [print(s.asDict()) for s in standings if s.abbr in abbrs]

def getPointsForPointsAgainst(abbr, game):
    home_abbr = game['home_abbr'].values[0]
    if abbr == home_abbr:
        pointsFor = game['home_points'].values[0]
        pointsAgainst = game['away_points'].values[0]
    else:
        pointsFor = game['away_points'].values[0]
        pointsAgainst = game['home_points'].values[0]
    return pointsFor, pointsAgainst

def calcWinLoseTie(st, pointsFor, pointsAgainst, conf, opp_conf, division, opp_division):
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

def getConference(abbr, year, df):
    afc = df.loc[df['year']==year, 'afc'].values[0]
    if abbr in afc:
        return 'afc'
    return 'nfc'

def getDivision(abbr, year, df):
    cols = list(df.columns)
    cols.remove('year')
    cols.remove('afc')
    cols.remove('nfc')
    temp = df.loc[df['year']==year]
    for col in cols:
        abbrs = temp[col].values[0]
        if abbr in abbrs:
            return col
    return None

def calcSOVandSOS(st, opp_wl, pointsFor, pointsAgainst):
    if pointsFor > pointsAgainst:
        st.sov += opp_wl
    st.sos += opp_wl
    return st

def updatePointRanks(st, pointsFor, pointsAgainst):
    st.pointsFor += pointsFor
    st.pointsAgainst += pointsAgainst
    return st

def calcOppVal(pointsFor, pointsAgainst):
    if pointsFor > pointsAgainst:
        return 1 # win
    elif pointsFor < pointsAgainst:
        return 0 # lose
    return 2 # tie

def getTdsForandTdsAgainst(abbr, game):
    home_abbr = game['home_abbr'].values[0]
    if abbr == home_abbr:
        tdsFor = game['home_pass_touchdowns'].values[0] + game['home_rush_touchdowns'].values[0]
        tdsAgainst = game['away_pass_touchdowns'].values[0] + game['away_rush_touchdowns'].values[0]
    else:
        tdsFor = game['away_pass_touchdowns'].values[0] + game['away_rush_touchdowns'].values[0]
        tdsAgainst = game['home_pass_touchdowns'].values[0] + game['home_rush_touchdowns'].values[0]
    return tdsFor, tdsAgainst

def updateTdsRank(st, tdsFor, tdsAgainst):
    st.tdsFor += tdsFor
    st.tdsAgainst += tdsAgainst
    return st

def buildTiebreakerAttributes(source: pd.DataFrame, cd: pd.DataFrame, sl: pd.DataFrame, _dir):
    
    if 'tiebreakerAttributes.csv' in os.listdir(_dir):
        print('tiebreakerAttributes.csv already created.')
        return pd.read_csv("%s.csv" % (_dir + 'tiebreakerAttributes'))
    
    div_dir = _dir + '/divisionData/'
    
    div_78 = pd.read_csv("%s.csv" % (div_dir + "divisions_78-01"))
    div_02 = pd.read_csv("%s.csv" % (div_dir + "divisions_02-22"))
    
    standings = []

    allStandings = []

    for index, row in source.iterrows():
        abbr = row['abbr']
        key = row['key']
        wy = row['wy']
        week = int(wy.split(" | ")[0])
        year = int(wy.split(" | ")[1])
        opp_abbr = row['opp_abbr']
        if year < 2002:
            div_df = div_78
        else:
            div_df = div_02
        # only compute regular season attributes
        if isRegularSeason(week, year, sl):
            game = cd.loc[cd['key']==key]
            pointsFor, pointsAgainst = getPointsForPointsAgainst(abbr, game)
            # use same year obj or create new for current year
            if objInStandings(abbr, year, standings):
                st = getStanding(abbr, year, standings)
            else:
                st = Standing(abbr, year)
            # compute features
            conf = getConference(abbr, year, div_df)
            opp_conf = getConference(opp_abbr, year, div_df)
            division = getDivision(abbr, year, div_df)
            opp_division = getDivision(opp_abbr, year, div_df)
            # set division and conference tags
            st.division = division
            st.conference = conf
            # get wins loses and ties for all games, in conference, and in division
            st = calcWinLoseTie(st, pointsFor, pointsAgainst, conf, opp_conf, division, opp_division)
            # calc win lose ratios for all, conference, and division
            st.updateWlRatios()
            # calc strength of victory and strength of schedule
            opp_wl = getOppWl(opp_abbr, year, standings)
            st = calcSOVandSOS(st, opp_wl, pointsFor, pointsAgainst)
            # pointsFor and pointsAgainst rankings
            st = updatePointRanks(st, pointsFor, pointsAgainst)
            # tdsFor and tdsAgainst rankings
            tdsFor, tdsAgainst = getTdsForandTdsAgainst(abbr, game)
            st = updateTdsRank(st, tdsFor, tdsAgainst)
            # append obj if it does not exist
            if not objInStandings(abbr, year, standings):
                standings.append(st)
            # week change
            if index < max(source.index):
                nextWy = source.loc[source.index==index+1, 'wy'].values[0]
                if wy != nextWy:
                    yearStandings = [s for s in standings if s.year == year]
                    temp_df = pd.DataFrame([s.asDict() for s in yearStandings])
                    # final season rankings => wy = last week of season
                    seasonWeeks = sl.loc[sl['year']==year, 'weeks'].values[0]
                    if week <= seasonWeeks:
                        temp_df.insert(1, 'wy', [wy for i in range(len(yearStandings))])
                        temp_df.drop(columns=['year'], inplace=True)
                        temp_df.sort_values(by=['conference', 'division'], inplace=True)
                        allStandings.append(temp_df)

    new_df = pd.concat(allStandings)

    new_df.to_csv("%s.csv" % (_dir + "tiebreakerAttributes"), index=False)
    
    return new_df

############################

# source = pd.read_csv("%s.csv" % "../source/sourceIndividual")
# cd = pd.read_csv("%s.csv" % "../../../../data/gameData")
# sl = pd.read_csv("%s.csv" % "../../../../data/seasonLength")

# buildTiebreakerAttributes(source, cd, sl)