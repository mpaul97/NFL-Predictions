import pandas as pd
import numpy as np
import os
import tkinter as tk
import json
import random
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append("../")

from pbp_names.custom_names import get_names_custom, get_name_indices

class Train:
    def __init__(self, _dir: str, K: int = None, lines: list[str] = None, key: str = None, target_sentence: str = None, keyword: str = None):
        self._dir = _dir
        self.data_dir = self._dir + "data/"
        self.pbp_dir = self._dir + "../playByPlay_v2/data/"
        if K: # random sample
            self.file = open((self.pbp_dir + "allDetails.txt"), "r")
            self.lines = (self.file.read()).split("\n")
            self.lines = random.sample(self.lines, k=K)
        if lines: # provided lines
            self.lines = lines
        if key: # all lines for game_key
            df = pd.read_csv("%s.csv" % (self.pbp_dir + "allTables"))
            self.lines = list(df.loc[df['key']==key, 'detail'].values)
        if target_sentence: # lines similar to target sentence
            file = open((self.pbp_dir + "allDetails.txt"), "r")
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            lines = (file.read()).split("\n")
            lines = random.sample(lines, k=5000)
            target = model.encode(target_sentence)
            vectors = model.encode(lines)
            sims = cosine_similarity([target], vectors)[0]
            ranked_lines = [(line, score) for line, score in zip(lines, sims)]
            ranked_lines.sort(key=lambda x: x[1], reverse=True)
            self.lines = [ranked_lines[i][0] for i in range(20)]
        if keyword: # lines with keyword
            file = open((self.pbp_dir + "allDetails.txt"), "r")
            self.lines = (file.read()).split("\n")
            self.lines = [l for l in self.lines if keyword in l]
            self.lines = random.sample(self.lines, k=20)
        self.ent_options = [
            'PASSER', 'RECEIVER', 'RUSHER', 'TACKLER',
            'DEFENDER', 'PENALIZER', 'FUMBLER', 
            'FUMBLE_RECOVERER', 'FUMBLE_FORCER',
            'INTERCEPTOR', 'KICKER', 'PUNTER', 
            'RETURNER', 'SACKER', 'LATERALER',
            'OTHER', 'TEAM_NAME', 'BLOCKER'
        ]
        self.data = self.get_data()
        self.all_data = []
        # tkinter - colors
        self.bg_color = '#1D1D1D'
        self.primary_color = '#29779e'
        # tkinter
        self.root = tk.Tk()
        self.root.configure(background=self.bg_color)
        w, h = 900, 750
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.data_index = 0
        self.submit_button = tk.Button(
            self.root,
            text="Submit",
            command=self.submit,
            background=self.primary_color,
            foreground='white',
            font=('Inter', 16, 'bold')
        )
        self.submit_button.pack(padx=30, pady=30, side=tk.BOTTOM, expand=True)
        self.selected_values = []
        self.display()
        self.root.mainloop()
        return
    def display(self):
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Frame) or isinstance(widget, tk.Label):
                widget.destroy()
        line = self.lines[self.data_index]
        num_menus = len(self.data[self.data_index])
        selected_values = [tk.StringVar() for _ in range(num_menus)]
        [s.set("PASSER") for s in selected_values]
        sentence = tk.Label(
            self.root,
            text=line,
            background=self.bg_color,
            foreground='white',
            font=('Inter', 16, 'bold'),
            wraplength=700,
            justify='center'
        )
        sentence.pack(padx=30, pady=30, side=tk.TOP, expand=True)
        main_frame = tk.Frame(self.root, background=self.bg_color)
        main_frame.pack()
        self.col_height = 5
        row = 0
        for index, (name, indicies) in enumerate(self.data[self.data_index]):
            frame = tk.Frame(main_frame, background=self.bg_color)
            col = int(index / self.col_height)
            frame.grid(column=col, row=row)
            row += 1
            if (index + 1) == self.col_height:
                row = 0
            name_label = tk.Label(
                frame,
                text=f'{name} ({indicies[0]}, {indicies[1]})',
                background=self.bg_color,
                foreground='white',
                font=('Inter', 12, 'bold')
            )
            name_label.grid(row=0, column=0)
            dropdown = tk.OptionMenu(
                frame,
                selected_values[index],
                *self.ent_options
            )
            dropdown.configure(
                background=self.primary_color,
                foreground='white',
                font=('Inter', 12, 'bold'),
                padx=10,
                pady=10
            )
            dropdown.grid(row=0, column=1, padx=20, pady=10)
        self.selected_values = selected_values
        return
    def submit(self):
        all_ents = []
        for i in range(len(self.selected_values)):
            ent_name = self.selected_values[i].get()
            _, indicies = self.data[self.data_index][i]
            all_ents.append((indicies[0], indicies[1], ent_name))
        self.all_data.append({
            "line": self.lines[self.data_index],
            "entities": all_ents
        })
        self.data_index += 1
        if self.data_index >= len(self.lines):
            self.root.destroy()
            # combine with train.json if exists
            if 'train.json' in os.listdir(self.data_dir):
                data = json.load(open((self.data_dir + "train.json"), "r"))
                self.all_data = data + self.all_data
            with open((self.data_dir + "train.json"), "w") as file:
                json.dump(self.all_data, file)
        else:
            self.display()
        return
    def get_data(self):
        data = {}
        for i, line in enumerate(self.lines):
            names = get_names_custom(line)
            if names != '':
                names = names.split("|")
                info = []
                for n in names:
                    groups = get_name_indices(line, n)
                    [info.append((n, g)) for g in groups]
                info.sort(key=lambda x: x[1][0])
                data[i] = info
            else: # no names <=> no entities
                data[i] = []
        return data
    
# END / Train

###########################

# !!! ONSIDE RECOVERER IS RETURNER !!!

# t = Train(
#     _dir="./", 
#     K=20
# )

lines = [
    "Dan Bailey yard field goal no good blocked by Joshua Kalu, recovered by Tye Smith. Penalty on Joshua Kalu: Defensive Offside, 5 yards (no play)",
    ", recovered by Jalen Hurts at PHI-20 Jalen Hurts for no gain. Penalty on Deatrich Wise: Defensive Offside, 5 yards (accepted) (no play)",
    "Dak Prescott pass incomplete intended for Rico Dowdle. Penalty on Preston Smith: Defensive Offside, 5 yards (accepted) (no play)",
    "De'Von Achane left guard for no gain (tackle by Ed Oliver). Penalty on Shaq Lawson: Defensive Offside, 5 yards (accepted) (no play)",
    "Gardner Minshew pass incomplete deep right intended for Michael Pittman. Penalty on Janarius Robinson: Defensive Offside, 5 yards (declined) . Penalty on Jack Jones: Defensive Pass Interference, 26 yards (accepted) (no play)",
    "Trevor Siemian pass complete deep left to Garrett Wilson for 22 yards. Penalty on Myles Garrett: Defensive Offside, 5 yards (declined)",
    "C.J. Beathard pass incomplete short left intended for Jamal Agnew. Penalty on Vita Vea: Defensive Offside (accepted) (no play)",
    "Jake Browning pass complete short left to Drew Sample for 16 yards (tackle by Eric Rowe and Mykal Walker). Penalty on Alex Highsmith: Defensive Offside, 5 yards (declined)",
    "Will Levis pass complete deep left to Nick Westbrook-Ikhine for 33 yards (tackle by Steven Nelson). Penalty on Sheldon Rankins: Defensive Offside, 5 yards (declined)",
    "Ryan Wright punts 52 yards, returned by DeAndre Carter for 4 yards (tackle by Johnny Mundt and Brian Asamoah). Penalty on Janarius Robinson: Defensive Offside, 5 yards (declined)",
]

t = Train(
    _dir="./", 
    lines=lines
)

# t = Train(
#     _dir="./",
#     key="202009100kan"
# )

# target_sentence = "Drew Brees pass complete short right to Darren Sproles for 3 yards, lateral to Robert Meachem for 1 yard (tackle by C.J. Spillman)"

# t = Train(
#     _dir="./",
#     target_sentence=target_sentence
# )

# t = Train(
#     _dir="./",
#     keyword='Dorian Thompson-Robinson'
# )