import pandas as pd
import regex as re
from random import randrange
import itertools

TEST_SENTENCES = [
    '@ pass complete right to @ for 16 yards (tackle by @ and @)',
    '@ left for no gain. @ fumbles, recovered by @. @ fumbles, recovered by @ and returned for 12 yards, touchdown, touchdown'
    
]

TEST_TEAM_SENTENCES = [
    'Timeout #2 by @'
]

TEST_KICKOFF_SENTENCES = [
    '@ won the coin toss and deferred, @ to receive the opening kickoff.'
]

PENALTY_TOKENS = list(pd.read_csv("%s.csv" % "D:/NFLPredictions3/myRegex/penaltyTokens")['name'].values)

ndf = pd.read_csv("%s.csv" % "D:/NFLPredictions3/playerNames/finalPlayerInfo")
pdf = pd.read_csv("%s.csv" % "D:/NFLPredictions3/playerNames/teamPidsByWeek")
tdf = pd.read_csv("%s.csv" % "D:/NFLPredictions3/teamNames/teamNames_pbp")
tndf = pd.read_csv("%s.csv" % "D:/NFLPredictions3/teamNames/teamNames")
cd = pd.read_csv("%s.csv" % "D:/NFLPredictions3/data/gameData")

# ['QB', 'RB', 'K', 'LS', 'TE', 'WR', 'UNK', 'KR', 'OL', 'DB', 'P', 'LB', 'DL']
OD_POSITIONS = {
    'QB': 'off', 'RB': 'off', 'K': 'st',
    'LS': 'st', 'TE': 'off', 'WR': 'off',
    'UNK': 'UNK', 'KR': 'st', 'OL': 'off',
    'DB': 'def', 'P': 'st', 'LB': 'def',
    'DL': 'def'
}

# convert penalties to lowercase
def convertPenalties(line):
    for token in PENALTY_TOKENS:
        p1 = r"\s" + re.escape(token) + r"\s"
        p2 = r"\s" + re.escape(token) + r"\,"
        p3 = re.escape(token) + r"\s"
        if re.search(p1, line):
            line = re.sub(p1, ' ' + token.lower() + ' ', line)
        if re.search(p2, line):
            line = re.sub(p2, ' ' + token.lower() + ',', line)
        if re.search(p3, line):
            line = re.sub(p3, token.lower() + ' ', line)
    return line

# convert team name to abbr
def teamNameToAbbr(name):
    try:
        abbr = tdf.loc[tdf['names'].str.contains(name), 'abbr'].values[0]
    except IndexError:
        abbr = 'UNK'
    return abbr

# convert kickoff team name to abbr
def kickoffNameToAbbr(name):
    try:
        abbr = tndf.loc[tndf['name'].str.contains(name), 'abbr'].values[0]
    except IndexError:
        abbr = 'UNK'
    return abbr

# alt abbr to abbr
def convertAltAbbr(test_abbr):
    try:
        abbr = tndf.loc[tndf['alt_abbr']==test_abbr, 'abbr'].values[0]
    except IndexError:
        abbr = test_abbr
    return abbr

#-----------------------
# player names regex functions

# Aaron Rodgers
def getR0(line):
    # remove (Ja'Marr)
    line = re.sub(r"[A-Z][a-z]+\'[A-Z]", '', line)
    # remove (CeeDee)
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+", '', line)
    # remove (Jeremiah Owusu-Koramoah)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # remove (D'Ernest Johnson)
    line = re.sub(r"[A-Z]\'[A-Z][a-z]+", '', line)
    # remove (By'not'e)
    line = re.sub(r"[A-Z][a-z]+\'[a-z]+\'[a-z]+", '', line)
    # remove (St. Brown)
    line = re.sub(r"[A-Z][a-z]+\sSt.\sBrown", '', line)
    # r0 - normal names (Aaron Rodgers)
    r0 = re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+", line)
    return r0

# Ja'Marr Chase
def getR1(line):
    # remove (Ja'Quan McMillian)
    line = re.sub(r"[A-Z][a-z][A-Z][a-z]+", '', line)
    # remove (Owusu-Koramoah)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # r1 - apostrophe first names (Ja'Marr Chase)
    r1 = re.findall(r"[A-Z][a-z]+\'[A-Z][a-z]+\s[A-Z][a-z]+", line)
    return r1

# J.K. Dobbins
def getR2(line):
    # remove (C.J. Gardner-Johnson)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # remove (McMillian)
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+", '', line)
    r2 = re.findall(r"[A-Z]\.[A-Z]\.\s[A-Z][a-z]+", line)
    return r2

# AJ Dillon
def getR3(line):
    # replace (Owusu-Koramoah)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # replace (McCloud)
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+", '', line)
    r3 = re.findall(r"[A-Z][A-Z]\s[A-Z][a-z]+", line)
    return r3

# CeeDee Lamb
def getR4(line):
    # remove (Owusu-Koramoah)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # remove ([A-Z][a-z]+ McCloud)
    line = re.sub(r"[A-Z][a-z]+\s[A-Z][a-z]+[A-Z][a-z]+", '', line)
    r4 = re.findall(r"[A-Z][a-z]+[A-Z][a-z]+\s[A-Z][a-z]+", line)
    return r4

# Jeremiah Owusu-Koramoah
def getR5(line):
    # remove (CeeDee)
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+", '', line)
    # remove (Ki-Jana)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+\s[A-Z]", '', line)
    # remove (D'Ernest)
    line = re.sub(r"[A-Z]\'[A-Z][a-z]+", '', line)
    # remove (Ja'Quan)
    line = re.sub(r"[A-Z][a-z]+\'[A-Z][a-z]+", '', line)
    r5 = re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+\-[A-Z][a-z]+", line)
    return r5

# Ki-Jana Carter
def getR6(line):
    # remove (Ray-Ray McCloud)
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+", '', line)
    # remove (Owusu-Koramoah)
    line = re.sub(r"[A-Z][a-z]+\s[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # remove (St. Brown)
    line = re.sub(r"[A-Z][a-z]+\sSt.\sBrown", '', line)
    r6 = re.findall(r"[A-Z][a-z]+\-[A-Z][a-z]+\s[A-Z][a-z]+", line)
    return r6

# Ray-Ray McCloud
def getR7(line):
    r7 = re.findall(r"[A-Z][a-z]+\-[A-Z][a-z]+\s[A-Z][a-z]+[A-Z][a-z]+", line)
    return r7

# D'Ernest Johnson
def getR8(line):
    # remove (Owusu-Koramoah)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # remove (McCloud)
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+", '', line)
    r8 = re.findall(r"[A-Z]+\'[A-Z][a-z]+\s[A-Z][a-z]+", line)
    return r8

# Uwe von Schamann
def getR9(line):
    # remove (KaVontae)
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+", '', line)
    # remove (Ki-Jana)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # remove (D'Ernest)
    line = re.sub(r"[A-Z]\'[A-Z][a-z]+", '', line)
    # remove (Ja'Quan)
    line = re.sub(r"[A-Z][a-z]+\'[A-Z][a-z]+", '', line)
    # remove (and)
    line = re.sub(r"\sand\s", ' ', line)
    r9 = re.findall(r"[A-Z][a-z]+\s[a-z]+\s[A-Z][a-z]+", line)
    return r9

# Neil O'Donnell
def getR10(line):
    # remove (CeeDee)
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+", '', line)
    # remove (Ki-Jana)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # remove (D'Ernest)
    line = re.sub(r"[A-Z]\'[A-Z][a-z]+\s[A-Z]", '', line)
    # remove (Ja'Quan [a-zA-Z])
    line = re.sub(r"[A-Z][a-z]+\'[A-Z][a-z]+\s[a-zA-Z]", '', line)
    r10 = re.findall(r"[A-Z][a-z]+\s[A-Z]+\'[A-Z][a-z]+", line)
    return r10

# Ka'imi Fairbairn
def getR11(line):
    # remove (Owusu-Koramoah)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # remove (McCloud)
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+", '', line)
    r11 = re.findall(r"[A-Z][a-z]+\'[a-z]+\s[A-Z][a-z]+", line)
    return r11

# Donte' Stallworth
def getR12(line):
    # remove (McCarthy)
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+", ' ', line)
    # remove (Owusu-Koramoah)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    r12 = re.findall(r"[A-Z][a-z]+\'\s[A-Z][a-z]+", line)
    return r12

# J.T. O'Sullivan
def getR13(line):
    r13 = re.findall(r"[A-Z]\.[A-Z]\.\s[A-Z]+\'[A-Z][a-z]+", line)
    return r13

# C.J. Gardner-Johnson
def getR14(line):
    r14 = re.findall(r"[A-Z]\.[A-Z]\.\s[A-Z][a-z]+\-[A-Z][a-z]+", line)
    return r14

# Ja'Quan McMillian
def getR15(line):
    r15 = re.findall(r"[A-Z][a-z]+\'[A-Z][a-z]+\s[A-Z][a-z]+[A-Z][a-z]+", line)
    return r15

# Aaron McCarthy
def getR16(line):
    # remove (Ja'Marr)
    line = re.sub(r"[A-Z][a-z]+\'[A-Z]", '', line)
    # remove (Jeremiah Owusu-Koramoah)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    # remove (D'Ernest Johnson)
    line = re.sub(r"[A-Z]\'[A-Z][a-z]+", '', line)
    # remove (CeeDee [A-Z])
    line = re.sub(r"[A-Z][a-z]+[A-Z][a-z]+\s[A-Z]", '', line)
    r16 = re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+[A-Z][a-z]+", line)
    return r16

# Donte' McCarthy
def getR17(line):
    r17 = re.findall(r"[A-Z][a-z]+\'\s[A-Z][a-z]+[A-Z][a-z]+", line)
    return r17

# C.J. McMillian
def getR18(line):
    r18 = re.findall(r"[A-Z]\.[A-Z]\.\s[A-Z][a-z]+[A-Z][a-z]+", line)
    return r18

# J.K. von Schamann
def getR19(line):
    # remove (and)
    line = re.sub(r"\sand\s", ' ', line)
    r19 = re.findall(r"[A-Z]\.[A-Z]\.\s[a-z]+\s[A-Z][a-z]+", line)
    return r19

# AJ Owusu-Koramoah
def getR20(line):
    r20 = re.findall(r"[A-Z][A-Z]\s[A-Z][a-z]+\-[A-Z][a-z]+", line)
    return r20

# AJ McCloud
def getR21(line):
    r21 = re.findall(r"[A-Z][A-Z]\s[A-Z][a-z]+[A-Z][a-z]+", line)
    return r21

# AJ O'Donnell
def getR22(line):
    r22 = re.findall(r"[A-Z][A-Z]\s[A-Z]\'[A-Z][a-z]+", line)
    return r22

# AJ von Schamann
def getR23(line):
    # remove (and)
    line = re.sub(r"\sand\s", ' ', line)
    r23 = re.findall(r"[A-Z][A-Z]\s[a-z]+\s[A-Z][a-z]+", line)
    return r23

# CeeDee Owusu-Koramoah
def getR24(line):
    r24 = re.findall(r"[A-Z][a-z]+[A-Z][a-z]+\s[A-Z][a-z]+\-[A-Z][a-z]+", line)
    return r24

# CeeDee McCloud
def getR25(line):
    r25 = re.findall(r"[A-Z][a-z]+[A-Z][a-z]+\s[A-Z][a-z]+[A-Z][a-z]+", line)
    return r25

# CeeDee O'Donnell
def getR26(line):
    r26 = re.findall(r"[A-Z][a-z]+[A-Z][a-z]+\s[A-Z]+\'[A-Z][a-z]+", line)
    return r26

# CeeDee von Schamann
def getR27(line):
    # remove (and)
    line = re.sub(r"\sand\s", ' ', line)
    r27 = re.findall(r"[A-Z][a-z]+[A-Z][a-z]+\s[a-z]+\s[A-Z][a-z]+", line)
    return r27

# Ki-Jana Owusu-Koramoah
def getR28(line):
    r28 = re.findall(r"[A-Z][a-z]+\-[A-Z][a-z]+\s[A-Z][a-z]+\-[A-Z][a-z]+", line)
    return r28

# Ki-Jana O'Donnell
def getR29(line):
    r29 = re.findall(r"[A-Z][a-z]+\-[A-Z][a-z]+\s[A-Z]\'[A-Z][a-z]+", line)
    return r29

# Ki-Jana von Schamann
def getR30(line):
    # remove (and)
    line = re.sub(r"\sand\s", ' ', line)
    r30 = re.findall(r"[A-Z][a-z]+\-[A-Z][a-z]+\s[a-z]+\s[A-Z][a-z]+", line)
    return r30

# D'Ernest Owusu-Koramoah
def getR31(line):
    r31 = re.findall(r"[A-Z]\'[A-Z][a-z]+\s[A-Z][a-z]+\-[A-Z][a-z]+", line)
    return r31

# D'Ernest McCloud
def getR32(line):
    r32 = re.findall(r"[A-Z]\'[A-Z][a-z]+\s[A-Z][a-z]+[A-Z][a-z]+", line)
    return r32

# D'Ernest O'Donnell
def getR33(line):
    r33 = re.findall(r"[A-Z]\'[A-Z][a-z]+\s[A-Z]\'[A-Z][a-z]+", line)
    return r33

# D'Ernest von Schamann
def getR34(line):
    # remove (and)
    line = re.sub(r"\sand\s", ' ', line)
    r34 = re.findall(r"[A-Z]\'[A-Z][a-z]+\s[a-z]+\s[A-Z][a-z]+", line)
    return r34

# Ka'imi Owusu-Koramoah
def getR35(line):
    r35 = re.findall(r"[A-Z][a-z]+\'[a-z]+\s[A-Z][a-z]+\-[A-Z][a-z]+", line)
    return r35

# Ka'imi McCloud
def getR36(line):
    r36 = re.findall(r"[A-Z][a-z]+\'[a-z]+\s[A-Z][a-z]+[A-Z][a-z]+", line)
    return r36

# Ka'imi O'Donnell
def getR37(line):
    r37 = re.findall(r"[A-Z][a-z]+\'[a-z]+\s[A-Z]\'[A-Z][a-z]+", line)
    return r37

# Ka'imi von Schamann
def getR38(line):
    # remove (and)
    line = re.sub(r"\sand\s", ' ', line)
    r38 = re.findall(r"[A-Z][a-z]+\'[a-z]+\s[a-z]+\s[A-Z][a-z]+", line)
    return r38

# Donte' Owusu-Koramoah
def getR39(line):
    r39 = re.findall(r"[A-Z][a-z]+\'\s[A-Z][a-z]+\-[A-Z][a-z]+", line)
    return r39

# Donte' O'Donnell
def getR40(line):
    r40 = re.findall(r"[A-Z][a-z]+\'\s[A-Z]\'[A-Z][a-z]+", line)
    return r40

# Donte' von Schamann
def getR41(line):
    # remove (and)
    line = re.sub(r"\sand\s", ' ', line)
    r41 = re.findall(r"[A-Z][a-z]+\'\s[a-z]+\s[A-Z][a-z]+", line)
    return r41

# Ja'Quan Owusu-Koramoah
def getR42(line):
    r42 = re.findall(r"[A-Z][a-z]+\'[A-Z][a-z]+\s[A-Z][a-z]+\-[A-Z][a-z]+", line)
    return r42

# Ja'Quan O'Donnell
def getR43(line):
    r43 = re.findall(r"[A-Z][a-z]+\'[A-Z][a-z]+\s[A-Z]\'[A-Z][a-z]+", line)
    return r43

# Ja'Quan von Schamann
def getR44(line):
    # remove (and)
    line = re.sub(r"\sand\s", ' ', line)
    r44 = re.findall(r"[A-Z][a-z]+\'[A-Z][a-z]+\s[a-z]+\s[A-Z][a-z]+", line)
    return r44

# Rob E. Smith
def getR45(line):
    r45 = re.findall(r"[A-Z][a-z]+\s[A-Z]\.\s[A-Z][a-z]+", line)
    return r45

# Butler By'not'e
def getR46(line):
    r46 = re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+\'[a-z]+\'[a-z]+", line)
    return r46

# Amon-Ra St. Brown
def getR47(line):
    r47 = re.findall(r"[A-Z][a-z]+\-[A-Z][a-z]+\sSt\.\sBrown", line)
    return r47

# Equanimeous St. Brown
def getR48(line):
    # remove (Amon-Ra)
    line = re.sub(r"[A-Z][a-z]+\-[A-Z][a-z]+", '', line)
    r48 = re.findall(r"[A-Z][a-z]+\sSt\.\sBrown", line)
    return r48

#--------------------

# get all names
def getNames(line, testing):
    line = convertPenalties(line)
    all_names = [
        getR0(line), getR1(line), getR2(line), getR3(line), getR4(line), getR5(line),
        getR6(line), getR7(line), getR8(line), getR9(line), getR10(line), getR11(line),
        getR12(line), getR13(line), getR14(line), getR15(line), getR16(line), getR17(line),
        getR18(line), getR19(line), getR20(line), getR21(line), getR22(line), getR23(line),
        getR24(line), getR25(line), getR26(line), getR27(line), getR28(line), getR29(line),
        getR30(line), getR31(line), getR32(line), getR33(line), getR34(line), getR35(line),
        getR36(line), getR37(line), getR38(line), getR39(line), getR40(line), getR41(line),
        getR42(line), getR43(line), getR44(line), getR45(line), getR46(line), getR47(line),
        getR48(line)
    ]
    found_names = []
    for index, names in enumerate(all_names):
        if len(names) != 0:
            found_names.append(index)
    if testing:
        return list(set([i for sublist in all_names for i in sublist])), found_names
    return list(set([i for sublist in all_names for i in sublist]))

# get team names
def getTeamNames(line):
    t0 = re.findall(r"\b[A-Z][a-z]+\s[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b", line)
    [t0.remove(t) for t in t0 if 'San Francisco' in t]
    t1 = re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+\s[0-9]+[a-z]+", line)
    return list(set(t0 + t1))

# get kickoff names
def getKickoffNames(line):
    k0 = re.findall(r"\b(?!Team\b)(?!Football\b)[A-Z][a-z]+\b", line)
    # Football Teams
    k1 = re.findall(r"\bFootball\sTeam\b", line)
    # 49ers
    k2 = re.findall(r"[0-9]+[a-z]+", line)
    return list(set(k0 + k1 + k2))

# convert name to pid and position - using teamPidsByWeek
def nameToInfo(name, key):
    # remove leading and trailing whitespace
    name = name.strip()
    # regex for names with middle intial, removes (Rod E. Jones)
    if re.search(r"[A-Z][a-z]+\s[A-Z].\s[A-Z][a-z]+", name):
        name_split = name.split(" ")
        name = name_split[0] + " " + name_split[-1]
    info = ndf.loc[(ndf['name'].str.contains(name))|(ndf['aka'].str.contains(name)), ['p_id', 'position']].values
    if info.shape[0] > 1:
        try:
            game_pids = pdf.loc[pdf['key']==key, 'pids'].values
            game_pids = '|'.join(game_pids)
            pid, position = [(pid, position) for pid, position in info if pid in game_pids][0]
        except IndexError:
            print(name, key, 'name with key not found.')
            pid = 'UNK'
            position = 'UNK'
    else:
        try:
            pid = info[0][0]
            position = info[0][1]
        except IndexError:
            print('!!!!!!!!!!!!!!!!!!!')
            print(name, 'missing.')
            print('!!!!!!!!!!!!!!!!!!!')
    return pid, position

# convert name to pid and position - using playerInfo
def nameToInfo2(name, isOff, p_abbr, all_abbrs, year):
    # get off and def abbrs
    off_abbr = p_abbr
    def_abbr = list(set(all_abbrs).difference(set([p_abbr])))[0]
    # info key combos
    off_info = off_abbr + "," + year
    def_info = def_abbr + "," + year
    # remove leading and trailing whitespace
    name = name.strip()
    # regex for names with middle intial, removes (Rod E. Jones)
    if re.search(r"[A-Z][a-z]+\s[A-Z].\s[A-Z][a-z]+", name):
        name_split = name.split(" ")
        name = name_split[0] + " " + name_split[-1]
    data = ndf.loc[(ndf['name'].str.contains(name))|(ndf['aka'].str.contains(name)), ['p_id', 'position', 'info']].values
    pid, pos = 'UNK', 'UNK'
    if data.shape[0] > 1:
        for row in data:
            if isOff and off_info in row[-1]: # pid, pos found using offensive abbr
                pid = row[0]
                pos = row[1]
            elif not isOff and def_info in row[-1]: # pid, pos found using defensive abbr
                pid = row[0]
                pos = row[1]
    else:
        pid, pos = data[0][0], data[0][1]
    return pid, pos

#--------------------
# testing
# -------------------

# test names
def test():
    
    file = open('testData.txt', 'r')
    names = file.read().split('\n')
    
    for name in names:
        for index, sentence in enumerate(TEST_SENTENCES):
            sentence = sentence.replace('@', name)
            f_names, f_indexes = getNames(sentence, True)
            # print(sentence)
            print(index, f_names, f_indexes)
    
    return

# test first and last name combinations
def testCombos():
    
    file = open('testDataCombo.txt', 'r')
    names = file.read().split('\n')
    
    empty_index = [i for i, name in enumerate(names) if len(name) == 0][0]
    
    firstNames = names[:empty_index]
    lastNames = names[empty_index + 1:]
    
    combos = list(itertools.product(firstNames, lastNames))
    
    names = [c[0] + " " + c[1] for c in combos]
        
    for name in names:
        # print(name)
        for index, sentence in enumerate(TEST_SENTENCES):
            sentence = sentence.replace('@', name)
            f_names, f_indexes = getNames(sentence, True)
            # print(sentence)
            print(index, f_names, f_indexes)
        print()
    
    return

# test team name to abbr
def testTeams():
    
    df = pd.read_csv("%s.csv" % "../teamNames/teamNames_pbp")
    
    for name in df['names'].values:
        if '|' in name:
            name_split = name.split('|')
            for index, sentence in enumerate(TEST_TEAM_SENTENCES):
                for n in name_split:
                    names = getTeamNames(sentence.replace('@', n))
                    print(n)
                    print(names)
        else:
            for index, sentence in enumerate(TEST_TEAM_SENTENCES):
                names = getTeamNames(sentence.replace('@', name))
                print(name)
                print(names)
    
    return

# test kickoff team names to abbr
def testKickoffTeams():
    
    df = pd.read_csv("%s.csv" % "../teamNames/teamNames")
    
    for name in df['name'].values:
        if '|' in name:
            name_split = name.split('|')
            for index, sentence in enumerate(TEST_KICKOFF_SENTENCES):
                for n in name_split:
                    names = getKickoffNames(sentence.replace('@', n))
                    print(n)
                    print(names)
        else:
            for index, sentence in enumerate(TEST_KICKOFF_SENTENCES):
                names = getKickoffNames(sentence.replace('@', name))
                print(name)
                print(names)
    
    return

######################

# name = 'Josh Allen'
# isOff = False
# p_abbr = 'BUF'
# all_abbrs = ['BUF', 'JAX']
# year = '2022'

# pid, position = nameToInfo2(name, isOff, p_abbr, all_abbrs, year)

# print(pid, position)

# print(pid, position)

# test()

# testCombos()

# testTeams()

# testKickoffTeams()

# print(getNames("Chris Warren middle for -1 yards (tackle by Jerry Ball and Greg Biekert)", True))