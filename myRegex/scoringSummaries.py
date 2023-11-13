import regex as re

def getNames(sentence):
    if "'" in sentence:
        s_arr = sentence.split(" ")
        ap_words = [(index, s) for index, s in enumerate(s_arr) if "'" in s]
        for word in ap_words:
            s_arr[word[0]] = word[1].lower()
        s1 = ' '.join(s_arr)
        temp0 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", s1)
    else:
        # temp0 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", sentence)
        temp0 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+[^\s]*", sentence)
    # with apostrophe (Ja'Marr)
    temp1 = re.findall(r"[A-Z][a-z]+,?\s?'(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+", sentence)
    # short first name (J.K.)
    temp2 = re.findall(r"[A-Z]\.[A-Z]\.+,?\s(?:[A-Z]*\.?\s*)?[A-Z][a-z]+", sentence)
    # double capital first name (AJ Dillion)
    temp3 = re.findall(r"[A-Z][A-Z]+,?\s?[A-Z][a-z]+", sentence)
    # capital lower lower capital lower... (CeeDee)
    temp4 = re.findall(r"[A-Z][a-z]+[A-Z][a-z]+,?\s?[A-Z][a-z]+", sentence)
    for name in temp4:
        name_arr = name.split(" ")
        lastName = name_arr[-1]
        for name1 in temp0:
            if lastName in name1:
                temp0.remove(name1)
    print('0', temp0)
    print('1', temp1)
    print('2', temp2)
    print('3', temp3)
    print('4', temp4)
    temp = temp0 + temp1 + temp2 + temp3 + temp4
    return temp

def namesToInfo(lines, index):
    # get names
    if '(' in lines[index]:
        lines[index] = lines[index].replace("(",".").replace(")",".")
    if ',' in lines[index]:
        lines[index] = lines[index].replace(',','.')
    if 'II' in lines[index]:
        lines[index] = lines[index].replace('II','')
    if 'III' in lines[index]:
        lines[index] = lines[index].replace('III','')
    if 'Jr.' in lines[index]:
        lines[index] = lines[index].replace('Jr.','')
    if "'" in lines:
        lines[index] = lines[index].replace("'",'')
    # for abbr in abbrs:
    #     if abbr in lines[index]:
    #         lines[index] = lines[index].replace(abbr, abbr.lower())
    names = getNames(lines[index])
    res = [idx for idx in range(len(lines[index])) if lines[index][idx].isupper()]
    if len(names) == 0 or (len(res) % 2 == 0 and len(names) < (len(res) / 2)):
        for k in range(0, len(res), 2):
            temp = lines[index]
            firstName = temp[res[k]:res[k+1]]
            lastName = temp[res[k+1]:]
            if " " in lastName:
                lastName = lastName[:lastName.index(" ")]
            elif "." in lastName:
                lastName = lastName[:lastName.index(".")]
            if len(firstName) > 1 and len(firstName.split(" ")) <= 3 and 'and' not in firstName and '-' not in firstName:
                names.append(firstName + lastName)
    print(names)
    return

##########################

# x = ["C.J. Ham. Tony Wilks. Ja'Marr Chase Tom Brady", "Ja'Marr Chase", "Aaron Rodgers"]
# x = ["CeeDee Lamb. AJ Dillion. Matt McCling."]
x = ["AJ Dillon up the middle for 4 yards (tackle by Anthony Walker and Jeremiah Owusu-Koramoah)"]
# x = ['Aaron Rodgers pass complete short right to Marcedes Lewis for 1 yard (tackle by Denzel Ward)']

namesToInfo(x, 0)

# print(getNames(x[0]))

# temp5 = re.findall(r"[A-Z][a-z]+\s[A-Z][a-z]+?\-[A-Z][a-z]+", x[0])

# print(temp5)

# temp0 = re.findall(r"[A-Z][a-z]+,?\s+(?:[A-Z][a-z]*\.?\s*)?[A-Z][a-z]+[^\s]*", x[0])

# print(temp0)