import pandas as pd

# 'p_id', 'completed_passes', 'attempted_passes', 'passing_yards',
#        'passing_touchdowns', 'interceptions_thrown', 'times_sacked',
#        'yards_lost_from_sacks', 'longest_pass', 'quarterback_rating',
#        'rush_attempts', 'rush_yards', 'rush_touchdowns', 'longest_rush',
#        'fumbles', 'fumbles_lost', 'abbr', 'wy', 'game_key', 'position'

POSITION_PATH = "positionData/"

def zeroDivision(n, d):
	return n / d if d else 0

def buildQb():

	pl = pd.read_csv("%s.csv" % (POSITION_PATH + "QBData"))

	compL = []
	tdL = []
	intL = []
	ypL = []
	aypL = []
	ycL = []
	spL = []
	ngL = []
	anyL = []
	rpaL = []

	for index, row in pl.iterrows():
		# week = int(row['wy'].split(" | ")[0])
		# year = row['wy'].split(" | ")[1]
		# pid = row['p_id']
		# temp = pl.loc[(pl['wy'].str.contains(year))&(pl['p_id']==pid)&(pl.index<index)]
		# stats = temp.sum(numeric_only=True).to_frame().transpose()
		# if week == 1 and year != '2002':
		# 	lastYear = str(int(year)-1)
		# 	temp = pl.loc[(pl['wy'].str.contains(lastYear))&(pl['p_id']==pid)]
		# 	if temp.empty:
		# 		temp = pl.loc[pl['wy'].str.contains(lastYear)]
		# 	stats = temp.sum(numeric_only=True).to_frame().transpose()
		# completion percentage
		comp = int(row['completed_passes'])
		att = int(row['attempted_passes'])
		compPerc = (zeroDivision(comp, att))*100
		compL.append(compPerc)
		# td percentage
		tds = int(row['passing_touchdowns'])
		tdPerc = (zeroDivision(tds, att))*100
		tdL.append(tdPerc)
		# interception percentage
		ints = int(row['interceptions_thrown'])
		intPerc = (zeroDivision(ints, att))*100
		intL.append(intPerc)
		# yards per attempt
		yards = int(row['passing_yards'])
		ypa = (zeroDivision(yards, att))
		ypL.append(ypa)
		# adjusted yards per attempt
		apa = zeroDivision((yards + 20 * tds - 45 * ints), att)
		aypL.append(apa)
		# yards per completion
		ypc = zeroDivision(yards, comp)
		ycL.append(ypc)
		# sack percentage
		tsk = int(row['times_sacked'])
		skp = zeroDivision(tsk, (att + tsk))*100
		spL.append(skp)
		# net gained per pass attempt
		sky = int(row['yards_lost_from_sacks'])
		ng = zeroDivision((yards - sky), (att + tsk))
		ngL.append(ng)
		# adjusted net yards per pass attempt
		anya = zeroDivision((yards - sky + (20 * tds) - (45 * ints)), (att + tsk))
		anyL.append(anya)
		# rush yards per attempt
		ry = int(row['rush_yards'])
		ra = int(row['rush_attempts'])
		rypa = zeroDivision(ry, ra)
		rpaL.append(rypa)

	pl['completion_percentage'] = compL
	pl['td_percentage'] = tdL
	pl['interception_percentage'] = intL
	pl['yards_per_attempt'] = ypL
	pl['adjusted_yards_per_attempt'] = aypL
	pl['yards_per_completion'] = ycL
	pl['sack_percentage'] = spL
	pl['net_gained_per_pass_attempt'] = ngL
	pl['adjusted_net_yards_per_pass_attempt'] = anyL
	pl['rush_yards_per_attempt'] = rpaL

	pl = pl.round(2)

	pl.to_csv((POSITION_PATH + "QBData.csv"), index=False)

# p_id	rush_attempts	rush_yards	rush_touchdowns	longest_rush	
# times_pass_target	receptions	receiving_yards	receiving_touchdowns	
# longest_reception	fumbles	fumbles_lost	abbr	wy	game_key	position

def buildRw():

	rd = pd.read_csv("%s.csv" % (POSITION_PATH + "RBData"))
	wd = pd.read_csv("%s.csv" % (POSITION_PATH + "WRData"))
	pl = pd.concat([rd, wd])

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

	rbs = pl.loc[pl['position']=='RB']
	wrs = pl.loc[pl['position']=='WR']

	rbs.to_csv("%s.csv" % (POSITION_PATH + "RBData"), index=False)
	wrs.to_csv("%s.csv" % (POSITION_PATH + "WRData"), index=False)
	#END BUILDRW

#############################################################
# Function Calls

buildQb()

buildRw()