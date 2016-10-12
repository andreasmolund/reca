from Tkinter import *
import numpy as np
import scipy as sp


COLOR_OFF = '#FFFFFF'
COLOR_ON = '#000000'
HEIGHT = 900


class CAWindow(Frame):

    def __init__(self, width, height):
        Frame.__init__(self)

        self.grid()
        self.master.title("CA")

        self.grid_cells = []
        self.width = width
        self.height = height
        self.time = -1
        self.init_grid()
        self.update_time(sp.zeros([width, height], dtype=np.dtype(int)))

    def init_grid(self):
        background = Frame(self,
                           bg='#FFFFFF',
                           width=self.width,
                           height=self.height)
        background.grid()

        for i in xrange(self.height):
            grid_time = []

            for j in xrange(self.width):
                cell = Frame(background,
                             bg=COLOR_OFF,
                             width=(HEIGHT/self.height)*self.width,
                             height=HEIGHT/self.height)
                cell.grid(row=i, column=j)
                t = Label(master=cell, text="", bg=COLOR_OFF, justify=CENTER, width=4, height=2)
                t.grid()
                grid_time.append(t)

            self.grid_cells.append(grid_time)

    def update_time(self, config):
        state = config[self.time]
        for i in xrange(self.width):
            try:
                self.grid_cells[self.time][i].configure(text="", bg=COLOR_ON if state[i] == 0b1 else COLOR_OFF)
            except IndexError as e:
                print "I/O error({0}): {1}".format(e.errno, e.strerror)
        self.time += 1
        self.update()

