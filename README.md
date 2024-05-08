# Directory Descriptions
- ### NFL_FantasyPredictions (IN PROGRESS)
  - predict TOTAL season points for QB, RB, WR, TE, FLEX, K and DST's
- ### coaches
  - scrape Pro-Football-Reference for all coaches of all NFL teams and seasons contained in gameData.csv
- ### data
  - allVolumeProjections
    - player stats at 100% volume (e.i. playerX plays 50% of snaps -> stats * 2 = 100% snaps)
  - positionData
    - all player data seperated by positions (QBData, RBData, etc.)
  - starters_23
    - scrape all starters for W1 from Pro-Football-Reference
    - all starters for previous weeks collected from sportsipy
  - allNewData.py
    - updates/processes all player data, position data, starters, etc.
  - approximateValues.py
    - sportsipy advancedStats missing in-season approximate values (season player ranks)
    - ML models to predict missing in-season approximate values
  - data.py
    - get various types of data from sportsipy & scraping Pro-Football-Reference
  - fantasyData.py
    - convert/update player stats to fantasy points (PPR, Half-PPR, and Standard points)
  - info.txt -> IGNORE
  - missingAdvancedStats.txt -> player ids for players with NO advanced stats
  - newStarters.py
    - scrape starters from sportsipy
    - duplicate starters from previous week
    - simpleStarters used to easily edit starters
  - olStatsData.py
    - get stats for offensive lineman when active (e.i. yards_lost_from_sacks, sack_percentage, rush_yards_per_attempt, etc.)
  - toKings.py
    - convert stats to DraftKings points
- ### maddenRatings
  - scrape player Madden ratings from maddenratings.weebly.com
  - predict new Madden ratings using player career average stats as training data
- ### main
  - fantasyPredictions (player stats & fantasy points predictions)
    -  features -> stores all feature build files ({featureName}.py) and train data
    -  testing -> various test scripts
    -  build.py
      - build all train data
      - train models
      - build all test data
      - create predictions
    - tfPredict.py (UNUSED) -> TensorFlow predictions
  - gamePredictions (game winner & team points predictions)
    -  features -> stores all feature build files ({featureName}.py) and train data
    -  testing -> various test scripts
    -  build.py
      - build all train data
      - train models
      - build all test data
      - create predictions
    - tfPredict.py (UNUSED) -> TensorFlow predictions
  - bestModels.py -> compares various models accuracy
  - database.py -> updates Firebase database with new week predictions
  - main.py
    - given a WEEK & YEAR, stores predictions for the specified arguments
    - calls fantasyPredictions build.py
    - calls gamePredictions build.py
    - calls database.py
- ### minuteMock (IN PROGRESS)
- ### myRegex
  - various Regex scripts for getting player names, team names, etc. from sentences
- ### old
  - old scripts used for easy references
- ### pbp_custom_ners
  - 
