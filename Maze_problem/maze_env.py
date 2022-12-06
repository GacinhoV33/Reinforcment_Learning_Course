#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import time
import tkinter as tk

UNIT = 40
MAZE_H = 8
MAZE_W = 8


class Maze:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Maze with Q-learning")
        self.window.geometry(f"{MAZE_W * UNIT}x{MAZE_H * UNIT}")
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_action = len(self.action_space)
        self.build_maze()

    def build_maze(self):
        self.canvas = tk.Canvas(self.window, width=MAZE_W * UNIT, height=MAZE_H * UNIT, bg='white')
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_W * UNIT
            self.canvas.create_line(x0, y0, x1, y1)

        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_H * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        origin = np.array([20, 20])
        hell1_center = origin + np.array([UNIT * 3, UNIT * 2])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15, fill='black'
        )
        hell2_center = origin + np.array([UNIT * 3, UNIT * 5])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15, fill='black'
        )

        oval_center = origin + UNIT * 6
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15, fill='yellow'
        )

        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15, fill='red'
        )
        self.canvas.pack()

    def render(self):
        time.sleep(0.01)
        self.window.update()

    def reset(self):
        self.window.update()
        time.sleep(0.25)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15, fill='red'
        )
        return self.canvas.coords(self.rect)

    def get_state_reward(self, action: int): # actions is 0, 1, 2, 3 -> up down right left
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:
            if s[1] < (MAZE_H -1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:
            if s[0] < (MAZE_W -1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:
            if s[0] > UNIT:
                base_action[0] -= UNIT
        self.canvas.move(self.rect, base_action[0], base_action[1])
        s_ = self.canvas.coords(self.rect)

        if s_ == self.canvas.coords(self.hell1) or s_ == self.canvas.coords(self.hell2):
            reward = -1
            s_ = 'terminal'
            done = True
            # done = True
        elif s_ == self.canvas.coords(self.oval):
            reward = 1
            s_ = 'terminal'
            done = True
        else:
            reward = 0
            done = False

        return s_, reward, done


if __name__ == '__main__':
    maze = Maze()
    maze.window.mainloop()