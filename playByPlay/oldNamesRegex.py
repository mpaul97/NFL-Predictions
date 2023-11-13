import regex as re
import pandas as pd

PENALTY_TOKENS = list(pd.read_csv("%s.csv" % "D:/NFLPredictions3/playByPlay/penaltyTokens")['name'].values)

ndf = pd.read_csv("%s.csv" % "D:/NFLPredictions3/playerNames/playerNames")
pdf = pd.read_csv("%s.csv" % "D:/NFLPredictions3/playerNames/teamPidsByWeek")
tdf = pd.read_csv("%s.csv" % "D:/NFLPredictions3/teamNames/teamNames_pbp")

# sort penalty tokens
def sortPenaltyTokens():
    df = pd.read_csv("%s.csv" % "penaltyTokens")
    df.sort_values(by=['name'], inplace=True)
    df.to_csv("%s.csv" % "penaltyTokens", index=False)
    return

# print each regex operation
def display(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14):
    print('r0:', r0)
    print('r1:', r1)
    print('r2:', r2)
    print('r3:', r3)
    print('r4:', r4)
    print('r5:', r5)
    print('r6:', r6)
    print('r7:', r7)
    print('r8:', r8)
    print('r9:', r9)
    print('r10:', r10)
    print('r11:', r11)
    print('r12:', r12)
    print('r13:', r13)
    print('r14:', r14)
    return

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

# find names
def getNames(line, show):
    # convert penalties to lower case so regex ignores them
    line = convertPenalties(line)
    # case 1
    r0 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", line)
    # case 1 - remove continuous names (Shawn Bouwens. Shawn)
    for index, name in enumerate(r0):
        name_split = name.split(" ")
        if len(name_split) > 2 and '.' in name_split[1] and len(name_split[1]) > 3:
            r0[index] = name_split[0] + " " + name_split[1].replace('.', '')
    # case 2 (Ja'Marr Chase)
    r1 = re.findall(r"[A-Z][a-z]+,?\s?'(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", line)
    # remove case 2 from case 1
    for n0 in r0:
        for n1 in r1:
            if n0 in n1:
                try:
                    r0.remove(n0)
                except ValueError:
                    print(n0 + ' already removed.')
    # short first name (J.K.)
    r2 = re.findall(r"[A-Z]\.[A-Z]\.+,?\s(?:[A-Z]*\.?\s*)?[A-Z][a-z]+\s?", line)
    # double capital first name (AJ Dillon)
    r3 = re.findall(r"[A-Z][A-Z]+,?\s?[A-Z][a-z]+", line)
    # capital lower lower capital lower... (CeeDee, JoJo)
    r4 = re.findall(r"[A-Z][a-z]+[A-Z][a-z]+,?\s?[A-Z][a-z]+", line)
    # remove case 5 from case 1
    for n0 in r0:
        for n1 in r4:
            if n0 in n1:
                try:
                    r0.remove(n0)
                except ValueError:
                    print(n0 + ' already removed.')
    # hyphen last names (Jeremiah Owusu-Koramoah)
    r5 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+?\-[A-Z][a-z]+", line)
    # case 5 - remove continuous names (Shawn Bouwens. Shawn)
    for index, name in enumerate(r5):
        name_split = name.split(" ")
        if len(name_split) > 2 and '.' in name_split[1] and len(name_split[1]) > 3:
            r5.remove(name)
    # remove case 6 from case 1
    for n0 in r0:
        for n1 in r5:
            if n0 in n1:
                try:
                    r0.remove(n0)
                except ValueError:
                    print(n0 + ' already removed.')
    # hyphen first names (Ray-Ray McCloud)
    r6 = re.findall(r"[A-Z][a-z]+?\-[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\s*)?[A-Z][a-z]+", line)
    # remove case 7 from case 1
    for n0 in r0:
        for n1 in r6:
            if n0 in n1:
                try:
                    r0.remove(n0)
                except ValueError:
                    print(n0 + ' already removed.')
    # case 8 (D'Ernest Johnson.)
    # r7 = re.findall(r"[A-Z]'[A-Z][a-z]+,?\s(?:[A-Z][a-z]?\s*)?[A-Z][a-z]+", line)
    r7 = re.findall(r"[A-Z]'[A-Z][a-z]+?\s[A-Z][a-z]+", line)
    # remove case 8 from case 1
    for n0 in r0:
        for n1 in r7:
            if n0 in n1:
                try:
                    r0.remove(n0)
                except ValueError:
                    print(n0 + ' already removed.')
    # case 9 (Uwe von Schamann)
    r8 = re.findall(r"[A-Z][a-z]+?\s[a-z]+?\s[A-Z][a-z]+", line)
    r8a = re.findall(r"[A-Z][a-z]+?\sand?\s[A-Z][a-z]+", line) # remove (White and Josiah)
    r8 = list(set(r8).difference(set(r8a)))
    # remove case 9 from case 1
    for n0 in r0:
        n0_arr = n0.split(" ")
        for word in n0_arr:
            for n8 in r8:
                if word in n8:
                    try:
                        r0.remove(n0)
                    except ValueError:
                        print(n0 + ' already removed.')
    # case 10 (Neil O'Donnell)
    r9 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?\'[A-Z][a-z]+", line)
    # remove case 10 from case 1
    for n0 in r0:
        n0_arr = n0.split(" ")
        for word in n0_arr:
            for n9 in r9:
                if word in n9:
                    try:
                        r0.remove(n0)
                    except ValueError:
                        print(n0 + ' already removed.')
    # case 11 (Ka'imi Fairbairn)
    r10 = re.findall(r"[A-Z][a-z]+?\'[a-z]+?\s[A-Z][a-z]+", line)
    # case 12 (Donte' Stallworth)
    r11 = re.findall(r"[A-Z][a-z]+'\s[A-Z][a-z]+", line)
    # case 13 (J.T. O'Sullivan)
    r12 = re.findall(r"[A-Z].[A-Z].?\s[A-Z]?\'[A-Z][a-z]+", line)
    # # case 14 (Robert Griffin III)
    # r13 = re.findall(r"[A-Z][a-z]+\s?[A-Z][a-z]+\s?[A-Z]+", line)
    # # remove case 14 from case 1
    # for n0 in r0:
    #     for n1 in r13:
    #         if n0 in n1:
    #             try:
    #                 r0.remove(n0)
    #             except ValueError:
    #                 print(n0 + ' already removed.')
    # case 15 (C.J. Gardner-Johnson)
    r13 = re.findall(r"[A-Z]\.[A-Z]\.+,?\s(?:[A-Z]*\.?\s*)?[A-Z][a-z]+-[A-Z][a-z]+", line)
    # case 16 (Ja'Quan McMillian)
    r14 = re.findall(r"[A-Z][a-z]+\'[A-Z][a-z]+\s[A-Z][a-z]+[A-Z][a-z]+", line)
    for n0 in r0: # remove from r0
        for n1 in r14:
            if n0 in n1:
                try:
                    r0.remove(n0)
                except ValueError:
                    print(n0 + ' already removed.')
    for n1 in r1: # remove from r1
        for n2 in r14:
            if n1 in n2:
                try:
                    r1.remove(n1)
                except ValueError:
                    print(n1 + ' already removed.')
    if show:
        display(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14)
    return list(set(r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12 + r13 + r14))

# convert name to pid and position
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

# convert team name to abbr
def teamNameToAbbr(name):
    try:
        abbr = tdf.loc[tdf['names'].str.contains(name), 'abbr'].values[0]
    except IndexError:
        abbr = 'UNK'
    return abbr

##############################

# pid, position = nameToPid('Aaron Jones', '202301080gnb')

# print(pid, position)

# sortPenaltyTokens()

# test = 'Jordan Love pass complete short middle to Allen Lazard for 17 yards (tackle by Kyzir White and Josiah Scott). Uwe von Schamann'

# test = "Matt Stover kicks off 66 yards, returned by Randy Jordan for 16 yards. Penalty on Le'Shai Maston: Illegal Block, 10 yards. Donte' Stallworth tackled by Clay Matthews."

test = 'J.K. Dobbins rushed for 5 yards. tackle by Ki-Jana Carter. Amon-Ra St. Brown'

names = getNames(test, True)

print(names)

# test = 'Jalen Hurts pass complete short right to A.J. Brown for 23 yards (tackle by Rasul Douglas)'

# for name in getNames(test, False):
#     pid, position = nameToInfo(name)

# tokens = ['Penalty', 'Illegal', 'Block']
# test = " Penalty on Le'Shai Maston: Illegal Block, 10 yards. "

# for token in tokens:
#     pattern = r"\s" + re.escape(token) + r",|\s"
#     if re.search(pattern, test):
#         test = re.sub(token, token.lower(), test)
        
# print(test)

# test = 'Rod E. Jones'

# if re.search(r"[A-Z][a-z]+\s[A-Z].\s[A-Z][a-z]+", test):
#     temp = test.split(" ")
#     temp = temp[0] + " " + temp[-1]
#     print(temp)

# --------------------------------------------

# PENALTY_TOKENS = [
#     'Penalty', 'False', 'Start', 'Offensive', 'Holding', 'Defensive',
#     'Neutral', 'Zone', 'Infraction', 'Offside', 'Illegal', 'Low',
#     'Block', 'Ineligible', 'Downfield', 'Pass', 'Unnecessary', 'Roughness',
#     'Defense', 'Offense', 'Unsportsmanlike', 'Conduct', 'Motion', 'Roughing',
#     'Passer', 'Use', 'Hands', 'Above', 'Waist', 'Tripping',
#     'Face', 'Mask', 'Short', 'Free', 'Kick', 'Contact',
#     'Interference', 'Encroachment', 'Invalid', 'Fair', 'Catch', 'Signal',
#     'Delay', 'Game', 'Shift', 'Formation', 'Ineligible', 'Kickoff',
#     'Out', 'Bounds', 'Intentional', 'Grounding', 'Interfere', 'Personal',
#     'Foul', 'Substitution', 'Forward', 'Into', 'Kicker', 'Horse',
#     'Collar', 'Tackle', 'Touch', 'Men', 'Field', 'Too',
#     'Many', 'Two', 'Point', 'Attempt', 'Crackback', 'Blindside',
# ]