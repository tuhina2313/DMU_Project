import tkinter as tk

from tkinter import Frame , Canvas , Tk , Button
import sys
import gridworld

master = tk.Tk()

if __name__ == '__main__':
    gw = gridworld.Gridworld(10, 0.3, 0.9)

    x = gw.size
    y = gw.size

    actions = gw.actions
    width = 100
    board = Canvas(master, width = x*width, height = y*width)


