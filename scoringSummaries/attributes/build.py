import pandas as pd
import numpy as np
import os

import sys
sys.path.append("../../")
from paths import DATA_PATH

def getPoints(info):
    new_info = '|'.join(info.split('|')[1:])
    try:
        new_info = new_info.replace('td','6')
        new_info = new_info.replace('fg','3')
        if 'nex' not in new_info and 'exrt' not in new_info:
            new_info = new_info.replace('ex','1')
        else:
            new_info = new_info.replace('nex','0')
        new_info = new_info.replace('two','2')
        new_info = new_info.replace('sf','2')
        new_info = new_info.replace('exrt','2')
        split = [int(i) for i in new_info.split('|')]
        return sum(split)
    except ValueError:
        print('Value not found.')
    return 0

def getMaxDif(points):
    return max(abs(ele1 - ele2) for ele1, ele2 in points)

def build():
    
    df = pd.read_csv("%s.csv" % "../all/infoTarget")
    cd = pd.read_csv("%s.csv" % (DATA_PATH + 'gameData'))
    
    test_keys = [
        '202211130gnb', '199409040clt', '202212170min',
        '202212190gnb'
    ]
    
    # cd = cd.head(10)
    # cd = cd.loc[cd['key'].isin(test_keys)]
    
    new_df = pd.DataFrame(
        columns=[
            'key', 'home_abbr', 'away_abbr',
            'isHomeBlowout', 'isAwayBlowout', 'isHomeComeback',
            'isAwayComeback', 'blowoutSize', 'comebackSize',
            'isClose', 'isHighScoring', 'homeNonOffensivePoints',
            'awayNonOffensivePoints'
        ]
    )
    
    for index, row in cd.iterrows():
        key = row['key']
        year = int(row['wy'].split(' | ')[1])
        print(row['wy'])
        home_abbr = row['home_abbr']
        away_abbr = row['away_abbr']
        data = df.loc[df['key']==key].sort_values(by=['num'])
        home_points, away_points = 0, 0
        home_no_points, away_no_points = 0, 0 # non-offensive points
        home_count, away_count = 0, 0
        possible_homeBlowout, possible_awayBlowout = False, False
        all_points = []
        for index1, row1 in data.iterrows():
            info = row1['info']
            isDefensiveTouchdown = row1['isDefensiveTouchdown']
            isSpecialTeamsTouchdown = row1['isSpecialTeamsTouchdown']
            points = getPoints(info)
            # home
            if home_abbr in info:
                home_points += points
                home_count += 1
                if isDefensiveTouchdown == 1 or isSpecialTeamsTouchdown == 1:
                    home_no_points += points
            # away
            if away_abbr in info:
                away_points += points
                away_count += 1
                if isDefensiveTouchdown == 1 or isSpecialTeamsTouchdown == 1:
                    away_no_points += points
            all_points.append((home_points, away_points))
            # home blowout flag
            if home_count - away_count >= 3 and home_points >= 21:
                possible_homeBlowout = True
            # away blowout flag
            if away_count - home_count >= 3 and away_points >= 21:
                possible_awayBlowout = True
            # end for
        # blowouts/comebacks
        isHomeBlowout, isAwayBlowout, isHomeComeback, isAwayComeback = 0, 0, 0, 0
        blowoutSize, comebackSize = 0, 0
        if possible_homeBlowout and home_points - away_points > 10:
            isHomeBlowout = 1
            blowoutSize = getMaxDif(all_points)
        elif possible_homeBlowout and home_points < away_points:
            isAwayComeback = 1
            comebackSize = getMaxDif(all_points)
        if possible_awayBlowout and away_points - home_points > 10:
            isAwayBlowout = 1
            blowoutSize = getMaxDif(all_points)
        elif possible_awayBlowout and away_points < home_points:
            isHomeComeback = 1
            comebackSize = getMaxDif(all_points)
        # ---------------------
        isClose = 0
        if abs(home_points - away_points) <= 7:
            isClose = 1
        # mean points for given year to compare for isHighScoring
        isHighScoring = 0
        year_points = cd.loc[cd['wy'].str.contains(str(year)), ['home_points', 'away_points']].values
        year_points = year_points.sum(axis=1)
        avg_points = np.mean(year_points)
        total_points = home_points + away_points
        if total_points >= avg_points:
            isHighScoring = 1
        new_df.loc[len(new_df.index)] = [
            key, home_abbr, away_abbr,
            isHomeBlowout, isAwayBlowout, isHomeComeback,
            isAwayComeback, blowoutSize, comebackSize,
            isClose, isHighScoring, home_no_points,
            away_no_points
        ]
        
    new_df.to_csv("%s.csv" % "attributes", index=False)
        
    return

#######################

# isHomeBlowout (home blewout away)
# isAwayBlowout
# isClose
# isHighScoring
# isHomeComeback (home cameback to win)
# isAwayComeback
# comebackSize (max for given game)
# blowoutSize (max for given game)
# non-offensize points (home + away)

build()