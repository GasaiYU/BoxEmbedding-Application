import numpy as np
import json
from math import ceil, floor, abs, sqrt

BEGIN = 0
END = 1
SEMICOLON = 2

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
    
    def to_token(self):
        x = ceil(self.x) + 9
        y = ceil(self.y) + 9
        radius = ceil(self.radius*2) - 1
        color = self.color
        return [x, y, radius, color, SEMICOLON]
    
    
class Line(object):
    """
    A circle object. To demonstrate a line, we need to specify its start point
    and its end point.
    We divide the x, y (-10 < x, y < 10) into 20 bins to convert the float num to int.
    """
    def __init__(self, x0, y0, x1, y1, color):
        assert x0 > -10 and x0 <= 10
        assert y0 > -10 and y0 <= 10
        assert x1 > -10 and x1 <= 10
        assert y1 > -10 and y1 <= 10
        
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1 
        self.color = color
    
    def to_token(self):
        x0 = ceil(self.x0) + 9
        y0 = ceil(self.y0) + 9
        x1 = ceil(self.x1) + 9
        y1 = ceil(self.y1) + 9
        color = self.color
        
        return [x0, y0, x1, y1, color, SEMICOLON]
    
    
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
        Rectangle.check_rectangle(lines, color)
        self.line1, self.line2, self.line3, self.line4 = lines
        self.color = color
    
    def to_token(self):
        res = []
        res.extend(self.line1.to_token()[:-1])
        res.extend(self.line2.to_token()[:-1])
        res.extend(self.line3.to_token()[:-1])
        res.extend(self.line4.to_token()[:-1])
        res.append(SEMICOLON)
        return res
    
    @classmethod
    def check_rectangle(lines):
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
 
 
class Triangle(object):
    """
    A Triangle object. To demonstrate a triangle, we need to specify its three lines.
    We divide the x, y (-10 < x, y < 10) into 20 bins to convert the float num to int.
    """
    def __init__(self, lines, color):
        Triangle.check_triangle(lines, color)
        self.line1, self.line2, self.line3 = lines
        self.color = color

    def to_token(self):
        res = []
        res.extend(self.line1.to_token()[:-1])
        res.extend(self.line2.to_token()[:-1])
        res.extend(self.line3.to_token()[:-1])
        res.append(SEMICOLON)
        
        return res

    @classmethod
    def check_triangle(lines):
        line1, line2, line3 = lines
        assert line1.x1 == line2.x0 and line1.y1 == line2.y0
        assert line2.x1 == line3.x0 and line2.y1 == line3.y0
        assert line3.x1 == line3.x0 and line3.y1 == line1.y0
        line1_len = sqrt((line1.x1 - line1.x0)**2 + (line1.y1 - line1.y0)**2)
        line2_len = sqrt((line2.x1 - line2.x0)**2 + (line2.y1 - line2.y0)**2)
        line3_len = sqrt((line3.x1 - line3.x0)**2 + (line3.y1 - line3.y0)**2)
        len_arr = sorted([line1_len, line2_len, line3_len])
        assert len_arr[0] + len_arr[1] > len_arr[2]
        pass