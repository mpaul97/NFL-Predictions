import tkinter as tk

class LabelSelector:
    def __init__(self, root, title: str, lines):
        self.root = root
        self.root.title(title)
        self.root.configure(background="black")
        # Initialize an empty list to store selected values
        self.selected_values = []
        # Create a label to display the strings from the list
        self.lines = lines
        self.current_index = 0
        self.label = tk.Label(
            root, 
            text=self.lines[self.current_index], 
            font=("Arial", 20), 
            background="black", 
            foreground="red",
            wraplength=400
        )
        self.label.pack(padx=40, pady=40)
        # Create buttons for 0 and 1
        self.button_0 = tk.Button(root, text="NO", width=20, command=lambda: self.add_value(0), font=("Arial", 18))
        self.button_0.pack(side=tk.LEFT, padx=20, pady=40)
        self.button_1 = tk.Button(root, text="YES", width=20, command=lambda: self.add_value(1), font=("Arial", 18))
        self.button_1.pack(side=tk.RIGHT, padx=20, pady=40)
        return
    def add_value(self, value):
        # Add the selected value to the list
        self.selected_values.append(value)
        # Move to the next string in the list
        self.current_index += 1
        if self.current_index < len(self.lines):
            self.label.config(text=self.lines[self.current_index])
        else:
            self.root.destroy()
        return

class PlayTypesGui:
    def __init__(self, lines: list[str]):
        self.root = tk.Tk()
        self.root.title("Label Selector")
        self.root.configure(background="black")
        # Initialize an empty list to store selected values
        self.play_types = [
            'pass', 'run', 'sack', 'penalty',
            'punt', 'kickoff', 'field_goal', 'extra_point',
            'coin_toss', 'timeout', 'challenge', 'kneel'
        ]
        self.selected_values = { pt: [] for pt in self.play_types }
        # Create a label to display the strings from the list
        self.lines = lines
        self.current_index = 0
        self.label = tk.Label(
            self.root, 
            text=self.lines[self.current_index], 
            font=("Arial", 20), 
            background="black", 
            foreground="red",
            wraplength=400
        )
        self.label.pack(padx=40, pady=40)
        # buttons
        self.frame = tk.Frame(self.root, background='black')
        self.frame.pack(padx=10, pady=10)
        column_length = 4
        row, column = 1, 0
        for i, pt in enumerate(self.play_types):
            self.add_button(pt, row=row, column=column)
            column += 1
            if column == column_length:
                column = 0
                row += 1
        self.root.mainloop()
        return
    def add_button(self, label: str, row: int, column: int):
        self.button = tk.Button(
            self.frame, 
            text=label, 
            width=10, 
            command=lambda: self.add_value(label), 
            font=("Arial", 18)
        )
        self.button.grid(row=row, column=column, padx=10, pady=10)
        return
    def add_value(self, label: str):
        # Add the selected value to the list
        self.selected_values[label].append(1)
        for key in self.selected_values:
            if key != label:
                self.selected_values[key].append(0)
        # Move to the next string in the list
        self.current_index += 1
        if self.current_index < len(self.lines):
            self.label.config(text=self.lines[self.current_index])
        else:
            self.root.destroy()
        return

# if __name__ == "__main__":
#     # root = tk.Tk()
#     # # label selector
#     # my_lines = ["String 1", "String 2", "String 3", "String 4"]
#     # app = LabelSelector(root, my_lines)
#     # root.mainloop()
#     # print(app.selected_values)
#     # ----------------------------
#     # play types
#     root = tk.Tk()
#     my_lines = ["String 1", "String 2", "String 3", "String 4"]
#     app = PlayTypes(root, my_lines)
#     root.mainloop()
#     print(app.selected_values)

