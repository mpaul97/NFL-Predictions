import regex as re

# find names
def getNames(line):
    # case 1
    r0 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", line)
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
    r2 = re.findall(r"[A-Z]\.[A-Z]\.+,?\s(?:[A-Z]*\.?\s*)?[A-Z][a-z]+", line)
    # double capital first name (AJ Dillion)
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
    # remove case 6 from case 1
    for n0 in r0:
        for n1 in r5:
            if n0 in n1:
                try:
                    r0.remove(n0)
                except ValueError:
                    print(n0 + ' already removed.')
    # hyphen first names (Ray-Ray McCloud)
    r6 = re.findall(r"[A-Z][a-z]+?\-[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", line)
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
    r11 = re.findall(r"[A-Z][a-z]+'\s?[A-Z][a-z]+", line)
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
    return r0 + r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8 + r9 + r10 + r11 + r12