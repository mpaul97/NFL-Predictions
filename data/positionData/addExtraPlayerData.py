import pandas as pd
import os

pd.options.mode.chained_assignment = None

# 'p_id', 'completed_passes', 'attempted_passes', 'passing_yards',
#        'passing_touchdowns', 'interceptions_thrown', 'times_sacked',
#        'yards_lost_from_sacks', 'longest_pass', 'quarterback_rating',
#        'rush_attempts', 'rush_yards', 'rush_touchdowns', 'longest_rush',
#        'fumbles', 'fumbles_lost', 'abbr', 'wy', 'game_key', 'position'

PERC_COLS = {
    'QB': 'attempted_passes', 'RB': 'total_touches',
    'WR': 'times_pass_target', 'TE': 'times_pass_target'
}

def zeroDivision(n, d):
	return n / d if d else 0
 
def buildTe():
    
    td = pd.read_csv("%s.csv" % ("TEData"))
    pl = td.copy()

    rpaL = []
    rprL = []
    catchpL = []
    rptL = []
    touchL = []
    ysL = []
    sptL = []
    totalL = []
    tptL = []
    tprL = []
    tpruL = []
    tptaL = []

    for index, row in pl.iterrows():
        # rush yards per attempt
        ry = int(row['rush_yards'])
        ra = int(row['rush_attempts'])
        rypa = zeroDivision(ry, ra)
        rpaL.append(rypa)
        # receiving yards per reception
        recy = int(row['receiving_yards'])
        recs = int(row['receptions'])
        rypr = zeroDivision(recy, recs)
        rprL.append(rypr)
        # catch percentage
        target = int(row['times_pass_target'])
        cp = zeroDivision(recs, target)*100
        catchpL.append(cp)
        # receiving yards per target
        rect = zeroDivision(recy, target)
        rptL.append(rect)
        # total touches
        touch = ra + recs
        touchL.append(touch)
        # yards from scrimmage
        ys = ry + recy
        ysL.append(ys)
        # scrimmage yards per touch
        spt = zeroDivision(ys, touch)
        sptL.append(spt)
        # total touchdowns
        rut = int(row['rush_touchdowns'])
        rt = int(row['receiving_touchdowns'])
        total = rut + rt
        totalL.append(total)
        # touchdown per touch
        tpt = zeroDivision(total, touch)
        tptL.append(tpt)
        # touchdown per reception
        tpr = zeroDivision(rt, recs)
        tprL.append(tpr)
        # touchdown per rush
        tpru = zeroDivision(rut, ra)
        tpruL.append(tpru)
        # touchdown per target
        tpta = zeroDivision(rt, target)
        tptaL.append(tpta)

    pl['rush_yards_per_attempt'] = rpaL
    pl['receiving_yards_per_reception'] = rprL
    pl['catch_percentage'] = catchpL
    pl['receiving_yards_per_target'] = rptL
    pl['total_touches'] = touchL
    pl['yards_from_scrimmage'] = ysL
    pl['scrimmage_yards_per_touch'] = sptL
    pl['total_touchdowns'] = totalL
    pl['touchdown_per_touch'] = tptL
    pl['touchdown_per_reception'] = tprL
    pl['touchdown_per_rush'] = tpruL
    pl['touchdown_per_target'] = tptaL

    pl = pl.round(3)

    pl.to_csv("%s.csv" % ("TEData-e"), index=False)
    #END BUILDTE

# add attempted_passes, rush_attempts, and targets percentages
def addPercentages():
    positions = ['QB', 'RB', 'WR', 'TE']
    for pos in positions:
        df = pd.read_csv("%s.csv" % (pos + 'Data'))
        keys = list(set(df['game_key'].values))
        print(pos)
        df_list = []
        for key in keys:
            data: pd.DataFrame = df.loc[df['game_key']==key]
            abbrs = list(set(data['abbr'].values))
            for abbr in abbrs:
                stats: pd.DataFrame = data.loc[data['abbr']==abbr]
                if stats.shape[0] > 1:
                    total = sum(stats[PERC_COLS[pos]].values)
                    stats['volume_percentage'] = stats[PERC_COLS[pos]] / total
                else:
                    stats['volume_percentage'] = 1.0
                df_list.append(stats)
        new_df = pd.concat(df_list)
        new_df = new_df.round(2)
        new_df.sort_values(by=['game_key', 'abbr'], inplace=True)
        new_df.to_csv("%s.csv" % (pos + 'Data'), index=False)
    return

#############################################################
# Function Calls

# buildQb()

# buildRw()

# buildTe()

# addPercentages()

# positions = ['QB', 'RB', 'WR', 'TE']
# for pos in positions:
#     df = pd.read_csv("%s.csv" % (pos + 'Data'))
#     df.sort_values(by=['game_key', 'abbr'], inplace=True)
#     df.to_csv("%s.csv" % (pos + 'Data'), index=False)