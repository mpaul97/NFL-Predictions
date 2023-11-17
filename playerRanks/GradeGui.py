import pandas as pd
import numpy as np
import os
import tkinter as tk

class GradeGui:
    def __init__(self, df: pd.DataFrame, merge_cols: list[str]):
        self.df = df
        self.merge_cols = merge_cols
        self.b_color = '#393e41'
        self.o_color = '#e94f37'
        self.w_color = '#f6f7eb'
        self.row_index = 0
        self.radio_values = []
        self.root = tk.Tk()
        self.root.configure(background=self.b_color)
        self.pid = tk.Label(self.root)
        self.pid.pack(pady=30)
        # Create a Tkinter label to display the row
        self.label = tk.Label(self.root)
        self.label.pack(padx=50, pady=(0, 30))
        # Create radio buttons
        self.radio_var = tk.IntVar()
        for i in range(1, 6):
            radio_button = tk.Radiobutton(
                self.root, 
                text=str(i), 
                variable=self.radio_var, value=i,
                foreground=self.o_color,
                background=self.w_color,
                activeforeground=self.b_color,
                activebackground=self.o_color,
                selectcolor=self.o_color,
                width=5,
                height=5,
                indicatoron=0,
                font=('Helvetica', 10)
            )
            radio_button.pack(anchor=tk.W, side=tk.LEFT)
        # Create a submit button
        self.submit_button = tk.Button(
            self.root, 
            text="Submit", 
            command=self.submit,
            width=10,
            height=3,
            foreground=self.o_color,
            font=('Helvetica', 16)
        )
        self.submit_button.pack(anchor=tk.W, side=tk.RIGHT)
        # Display the first row
        self.display_row()
        self.root.mainloop()
        return
    def display_row(self):
        row: pd.Series = self.df.iloc[self.row_index]
        pid = row['p_id']
        row = row[len(self.merge_cols):]
        row_str = '\n'.join([f"{col}: {value}" for col, value in row.items()])
        # self.pid.configure(text=pid, foreground=self.o_color, background=self.b_color, font=('Helvetica', 25))
        self.label.configure(text=row_str, background=self.b_color, foreground=self.w_color, font=('Helvetica', 20))
        return
    def submit(self):
        selected_value = self.radio_var.get()
        self.radio_values.append(selected_value)
        self.row_index += 1
        if self.row_index >= len(self.df):
            self.root.destroy()
        else:
            self.display_row()
        return