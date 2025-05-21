import tkinter as tk
from gui import TBRGS_GUI

if __name__ == "__main__":

    # Running the GUI
    root = tk.Tk()
    app = TBRGS_GUI(root)
    root.mainloop()