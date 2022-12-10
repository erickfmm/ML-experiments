import tkinter as tk

import matplotlib

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..')))

import preprocessing.psychoacoustics.bark as bark
import preprocessing.psychoacoustics.erb as erb
import preprocessing.psychoacoustics.mel as mel
import preprocessing.psychoacoustics.sone as sone

matplotlib.use("TkAgg")


class PsychoAcPlotsWin:
    def __init__(self, master, min_value=0, max_value=20000, step=10, max_sone=625):
        self.frame = tk.Frame(master)
        # scrollbar = tk.Scrollbar(self.frame, width=16)
        # scrollbar.pack(side=tk.RIGHT, fill=tk.Y)#, expand=False)
        figure = Figure()
        plot = figure.add_subplot(2, 2, 1)
        x = [i for i in range(min_value, max_value, step)]
        # Bark
        y = [bark.bark(i) for i in range(min_value, max_value, step)]
        plot.plot(x, y, color="red")
        y = [bark.bark2(i) for i in range(min_value, max_value, step)]
        plot.plot(x, y, color="blue")
        y = [bark.bark_1990_traunmuller(i) for i in range(min_value, max_value, step)]
        plot.plot(x, y, color="green")
        y = [bark.bark_1992_wang(i) for i in range(min_value, max_value, step)]
        plot.plot(x, y, color="black")
        plot.set_title("Bark plots vs frequency")
        plot.legend(["7000", "1990Traunmuller", "1992Wang", "7500"])
        # ERB
        plot = figure.add_subplot(2, 2, 2)
        y = [erb.erb_linear(i) for i in range(min_value, max_value, step)]
        plot.plot(x, y, color="red")
        y = [erb.erb_2ndorder_poly(i) for i in range(min_value, max_value, step)]
        plot.plot(x, y, color="black")
        y = [erb.erb_matlab_voicebox(i) for i in range(min_value, max_value, step)]
        plot.plot(x, y, color="orange")
        plot.set_title("ERB plots vs frequency")
        plot.legend(["linear", "poly 2nd", "matlab"])
        # MEL
        plot = figure.add_subplot(2, 2, 3)
        y = [mel.mel_700(i) for i in range(min_value, max_value, step)]
        plot.plot(x, y, color="red")
        y = [mel.mel_1000(i) for i in range(min_value, max_value, step)]
        plot.plot(x, y, color="blue")
        y = [mel.mel_625(i) for i in range(min_value, max_value, step)]
        plot.plot(x, y, color="green")
        plot.set_title("MEL plots vs frequency")
        plot.legend(["700", "1000", "625"])
        # SONE
        plot = figure.add_subplot(2, 2, 4)
        x_sone = [i for i in range(1, max_sone, step)]
        y = [sone.sone_aproximation(i) for i in range(1, max_sone, step)]
        plot.plot(x_sone, y, color="blue")
        y = [sone.sone(i) for i in range(1, max_sone, step)]
        plot.plot(x_sone, y, color="red")
        plot.set_title("Sone vs phons")
        plot.legend(["sone", "sone aproximation"])
        self.canvas = FigureCanvasTkAgg(figure, self.frame)
        self.canvas.get_tk_widget().pack(side="top", fill='both', expand=True)
        # canvas.pack(side="top",fill='both',expand=True)
        # main loop
        self.frame.pack(side="top", fill='both', expand=True)


if __name__ == '__main__':
    root = tk.Tk()
    app = PsychoAcPlotsWin(root)
    root.mainloop()
