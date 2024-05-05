import pandas as pd
import numpy as np
import os
import requests
from bs4 import BeautifulSoup
import time
import regex as re

class PidEnt:
    def __init__(self, name: str, start: int, end: int):
        self.name = name
        self.start = start
        self.end = end
        self.stripped_index = -1
        self.pid = ''
        return
    def show(self):
        print(self.__dict__)
        return

def get_pids_detail(key: str):
    url = "https://www.pro-football-reference.com/boxscores/" + key + ".htm"
    text = requests.get(url).text
    start = text.index('div_pbp')
    soup = BeautifulSoup(text[start:], 'html.parser')
    df = pd.DataFrame(columns=['primary_key', 'pids_detail'])
    for index, tag in enumerate(soup.find_all('td', {'data-stat': 'detail'})):
        line = str(tag)
        links = tag.findChildren('a')
        for link in links:
            link = str(link)
            if "href" in link: # replace pid links with raw pids
                p, t = '/players/', '/teams/'
                if p in link:
                    start = link.index(p)
                    end = link.index('.htm')
                    pid = link[start+len(p)+2:end]
                    line = line.replace(link, pid)
                if t in link:
                    abbr = re.findall(r"[A-Z]{3}", line)[0]
                    line = line.replace(link, abbr)
            else: # remove links without href tags
                line = line.replace(link, "")
        # remove style tags - <i>, <b>
        line = re.sub(r"<\/{0,1}i>", "", line)
        line = re.sub(r"<\/{0,1}b>", "", line)
        # content of td tag (detail)
        line = line[line.index(">")+1:]
        line = line[:line.index("<")]
        df.loc[len(df.index)] = [f"{key}-{index}", line]
    sample = pd.read_csv("%s.csv" % "../playByPlay_v2/data/sample")
    print(sample.shape, df.shape)
    # df.to_csv("%s.csv" % "temp", index=False)
    return

def get_passer_epa():
    df = pd.read_csv("%s.csv" % "../playByPlay_v2/data/sample")
    edf = pd.read_csv("%s.csv" % "../playByPlay_v2/data/sample_entities")
    pids = pd.read_csv("%s.csv" % "temp")
    df = df.merge(edf, on=['primary_key'])
    df = df.merge(pids, on=['primary_key'])
    print(df.columns)
    return

def show_start_end(s: str, name: str):
    print(s.index(name), s.index(name)+len(name))
    return

#################################

# get_pids_detail("202401070gnb")

# get_passer_epa()

pid_detail = "FielJu00 pass complete short middle to St.BEq00 for 18 yards (tackle by CampDe00 and SavaDa00)"
detail = "Justin Fields pass complete short middle to Equanimeous St. Brown for 18 yards (tackle by De'Vondre Campbell and Darnell Savage)"
ents = {
    'PASSER': 'Justin Fields:0:13', 'RECEIVER': 'Equanimeous St. Brown:44:65', 'TACKLER': "De'Vondre Campbell:90:108|Darnell Savage:113:127"
}

# ent_objs = []
# for val in ents.values():
#     for item in val.split("|"):
#         name = item.split(":")[0]
#         start, end = int(item.split(":")[1]), int(item.split(":")[2])
#         ent_objs.append(PidEnt(name, start, end))
        
# ent_objs.sort(key=lambda x: x.start)

# stripped_detail = detail
# rep_count = 0

# for obj in ent_objs:
#     name, start, end = obj.name, obj.start, obj.end
#     stripped_detail = stripped_detail[:start-rep_count] + name.replace(" ", "") + stripped_detail[end-rep_count:]
#     obj.stripped_index = stripped_detail.split(" ").index(stripped_detail[start-rep_count:end-rep_count].rstrip())
#     rep_count += len(name.split(" "))-1
        
# pid_arr = pid_detail.split(" ")

# for pe in ent_objs:
#     pe.pid = (pid_arr[pe.stripped_index]).replace("(","").replace(")","")
    
# [e.show() for e in ent_objs]

# ----------------------------------------------------

# s = "Ha Ha for two yards John Denver over the hill."
# l = [(0, 5), (20, 31)]
# s0 = ""
# last_index = 0

# for start, end in l:
#     s0 += s[last_index:start]
#     substring = s[start:end].replace(" ","")
#     s0 += substring
#     last_index = end

# s0 += s[last_index:]

# for start, end in l:
#     print(s0[start:end])

s = "Anders Carlson kicks off 66 yards, returned by Velus Jones for 32 yards (tackle by Kristian Welch and Patrick Taylor)"
sp = "CarlAn00 kicks off 66 yards, returned by JoneVe00 for 32 yards (tackle by WelcKr00 and TaylPa01)"
# Kristian Welch:83:97|Patrick Taylor:102:116,,,,,,,Anders Carlson:0:14,,Velus Jones:47:58
l = [(0, 14), (47, 58), (83, 97), (102, 116)]
s0 = s
indices = []

removed_count = 0

for start, end in l:
    substring = s[start:end]
    num_spaces = substring.count(" ")
    sr, er = start-removed_count, end-removed_count
    ent_key = (substring.replace(" ","") + str(start))
    s0 = s0[:sr] + ent_key + s0[er:]
    # print(f"Removed Start: {sr}")
    # print(f"Removed End: {er}")
    # show_start_end(s0, ent_key)
    indices.append([i for i, val in enumerate(s0.split(" ")) if ent_key in val][0])
    # Ha Ha - 1 space, len(start) = 1, start and end indices do not change, len(HaHa0) == len(Ha Ha)
    removed_count += num_spaces - len(str(start))
    
print(s0)
print(indices)

sp_arr = sp.split(" ")
[print(sp_arr[i]) for i in indices]
