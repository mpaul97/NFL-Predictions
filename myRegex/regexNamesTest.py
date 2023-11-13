import pandas as pd
import regex as re

def getNames(line):
    # case 1
    r0 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", line)
    # case 2 (Ja'Marr Chase)
    r1 = re.findall(r"[A-Z][a-z]+,?\s?'(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", line)
    # remove case 2 from case 1
    for n0 in r0:
        for n1 in r1:
            if n0 in n1:
                r0.remove(n0)
    # short first name (J.K.)
    r2 = re.findall(r"[A-Z]\.[A-Z]\.+,?\s(?:[A-Z]*\.?\s*)?[A-Z][a-z]+", line)
    # double capital first name (AJ Dillion)
    r3 = re.findall(r"[A-Z][A-Z]+,?\s?[A-Z][a-z]+", line)
    # capital lower lower capital lower... (CeeDee, JoJo)
    r4 = re.findall(r"[A-Z][a-z]+[A-Z][a-z]+,?\s?[A-Z][a-z]+", line)
    # remove case 5 from case 1
    for n0 in r0:
        for n1 in r4:
            if n0 in n1:
                r0.remove(n0)
    # hyphen last names (Jeremiah Owusu-Koramoah)
    r5 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+?\-[A-Z][a-z]+", line)
    # remove case 6 from case 1
    for n0 in r0:
        for n1 in r5:
            if n0 in n1:
                r0.remove(n0)
    # hyphen first names (Ray-Ray McCloud)
    r6 = re.findall(r"[A-Z][a-z]+?\-[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", line)
    # remove case 7 from case 1
    for n0 in r0:
        for n1 in r6:
            if n0 in n1:
                r0.remove(n0)
    # case 8 (D'Ernest Johnson.)
    # r7 = re.findall(r"[A-Z]'[A-Z][a-z]+,?\s(?:[A-Z][a-z]?\s*)?[A-Z][a-z]+", line)
    r7 = re.findall(r"[A-Z]'[A-Z][a-z]+?\s[A-Z][a-z]+", line)
    # remove case 8 from case 1
    for n0 in r0:
        for n1 in r7:
            if n0 in n1:
                r0.remove(n0)
    # case 9 (Uwe von Schamann)
    r8 = re.findall(r"[A-Z][a-z]+?\s[a-z]+?\s[A-Z][a-z]+", line)
    # remove case 9 from case 1
    for n0 in r0:
        n0_arr = n0.split(" ")
        for word in n0_arr:
            for n8 in r8:
                if word in n8:
                    r0.remove(n0)
    # case 10 (Neil O'Donnell)
    r9 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?\'[A-Z][a-z]+", line)
    # remove case 10 from case 1
    for n0 in r0:
        n0_arr = n0.split(" ")
        for word in n0_arr:
            for n9 in r9:
                if word in n9:
                    r0.remove(n0)
    return r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9

######################################

df = pd.read_csv("%s.csv" % "testLines")

# df = df.iloc[70:]

lines = df['line'].values

# for line in lines:
#     if 'Timeout' in line:
#         print(line)
#         print(getNames(line))
#         print()

# line = "Joe Burrow pass complete to Ja'Marr Chase. D'Ernest Johnson. AJ Dillion. J.K. Dobbins. Patrick Mahomes Jr. Jeremiah Owusu-Koramoah. Ray-Ray McCloud. A.J. Green."

# line = "Joe Burrow pass complete to Ja'Marr Chase. D'Ernest Johnson. Uwe von Schamann. JoJo Domann."

line = "Joe Burrow pass complete to Ja'Marr Chase. Neil O'Donnell"

print(getNames(line))