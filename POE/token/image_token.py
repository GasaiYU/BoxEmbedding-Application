import json
from math import ceil, sqrt
import random
import os
import argparse
import re

import matplotlib.pyplot as plt

import numpy as np

MAX_TOKEN_LEN = 27


SEED = 20010628
COLOR_ARR = ['RED', "BLUE", 'GREEN', 'PURPLE', 'BLACK']
SHAPE_ARR = ['Circle', 'Rectangle', 'Triangle']
PAD = 32

class Circle(object):
    """
    A circle object. To demonstrate a circle, we need to specify its radius
    and its center coordinate.
    We divide the x, y (-10 < x, y < 10) and radius (0 < radius < 10) into 20 bins
    to convert the float num to int.
    """
    def __init__(self, x, y, radius, color):
 
        assert x > -10 and x <= 10
        assert y > -10 and y <= 10
        assert radius > 0 and radius <= 10
        
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color

    def convert_to_image(self, action, index):
        fig = plt.figure(figsize=(5, 5))

        circle = plt.Circle((self.x, self.y), self.radius, color=self.color, fill=True)
        plt.gcf().gca().add_artist(circle)

        plt.axis('equal')
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.savefig(f'../image_dataset/action/data/circle_{index}.png')
        plt.close()
        
    def __str__(self):
        x = ceil(self.x) + 9
        y = ceil(self.y) + 9
        radius = ceil(self.radius*2) - 1
        color = self.color
        return f"<BEGIN> Circle({x}, {y}, {radius}); {color} <END>\n"
    
    
class Line(object):
    """
    A circle object. To demonstrate a line, we need to specify its start point
    and its end point.
    We divide the x, y (-10 < x, y < 10) into 20 bins to convert the float num to int.
    """
    def __init__(self, x0, y0, x1, y1):
        assert x0 > -10 and x0 <= 10
        assert y0 > -10 and y0 <= 10
        assert x1 > -10 and x1 <= 10
        assert y1 > -10 and y1 <= 10
        
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1  

    
    def __str__(self):
        x0 = ceil(self.x0) + 9
        y0 = ceil(self.y0) + 9
        x1 = ceil(self.x1) + 9
        y1 = ceil(self.y1) + 9
        
        return f"Line({x0}, {y0}, {x1}, {y1})\n"


class Rectangle(object):
    """
    A Rectangle object. To demonstrate a rectangle, we need to specify its four lines.
    We divide the x, y (-10 < x, y < 10) into 20 bins to convert the float num to int.
    line1: The above line.
    line2: The right line.
    line3: The below line.
    line4: The left line.
    """
    def __init__(self, lines, color):
        Rectangle.check_rectangle(lines)
        self.line1, self.line2, self.line3, self.line4 = lines 
        self.color = color
    
    
    @classmethod
    def check_rectangle(cls, lines):
        line1, line2, line3, line4 = lines
        assert line1.y0 == line1.y1
        assert line2.x0 == line2.x1
        assert line3.y0 == line3.y1
        assert line4.x0 == line4.x1
        line1_len = abs(line1.x1 - line1.x0)
        line2_len = abs(line2.y1 - line2.y0)
        line3_len = abs(line3.x1 - line3.x0)
        line4_len = abs(line4.y1 - line4.y0)
        
        assert line1_len == line3_len and line2_len == line4_len
    
    def __str__(self):
        return f"<BEGIN> {str(self.line1)[:-1]}; {str(self.line2)[:-1]}; "+\
                f"{str(self.line3)[:-1]}; {str(self.line4)[:-1]}; {self.color} <END>\n"
                
    def convert_to_image(self, action, index):
        fig = plt.figure(figsize=(5, 5))
        x = self.line3.x0
        y = self.line3.y0
        rect = plt.Rectangle((x, y), h, w, color=color)
        plt.gcf().gca().add_artist(rect)

        plt.axis('equal')
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.savefig(f'../dataset/data/rectangle/{index}.png')
        self.rectangle_labels[f'../dataset/data/rectangle/{index}.png'] = {'x':x, 'y':y, 'h':h, 'w':w, 'color':color}
        plt.close()