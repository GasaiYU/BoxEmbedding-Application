import json
from math import ceil, sqrt
import random
import os
import argparse
import re

import matplotlib.pyplot as plt

import numpy as np

MAX_TOKEN_LEN = 9


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

    def __str__(self):
        x = ceil(self.x) + 9
        y = ceil(self.y) + 9
        radius = ceil(self.radius*2) - 1
        color = self.color
        return f"Circle {color}\n"
    
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
        
        return f"Line\n"
    
    
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
        return f"{str(self.line1)[:-1]} ; {str(self.line2)[:-1]} ; "+\
                f"{str(self.line3)[:-1]} ; {str(self.line4)[:-1]} ; {self.color}\n"
 
class Triangle(object):
    """
    A Triangle object. To demonstrate a triangle, we need to specify its three lines.
    We divide the x, y (-10 < x, y < 10) into 20 bins to convert the float num to int.
    """
    def __init__(self, lines, color):
        Triangle.check_triangle(lines)
        self.line1, self.line2, self.line3 = lines
        self.color = color


    @classmethod
    def check_triangle(cls, lines):
        line1, line2, line3 = lines
        assert line1.x1 == line2.x0 and line1.y1 == line2.y0
        assert line2.x1 == line3.x0 and line2.y1 == line3.y0
        assert line3.x1 == line1.x0 and line3.y1 == line1.y0
        line1_len = sqrt((line1.x1 - line1.x0)**2 + (line1.y1 - line1.y0)**2)
        line2_len = sqrt((line2.x1 - line2.x0)**2 + (line2.y1 - line2.y0)**2)
        line3_len = sqrt((line3.x1 - line3.x0)**2 + (line3.y1 - line3.y0)**2)
        len_arr = sorted([line1_len, line2_len, line3_len])
        assert len_arr[0] + len_arr[1] > len_arr[2]
        pass
    
    def __str__(self):
        return f"{str(self.line1)[:-1]} ; {str(self.line2)[:-1]} ; " \
                + f"{str(self.line3)[:-1]} ; {self.color}\n"
    
    
class TokenGenerator(object):
    def __init__(self, num_circles, num_rectangles, num_triangles, config_dir):
        self.num_circles = num_circles
        self.num_rectangles = num_rectangles
        self.num_triangles = num_triangles
        self.config_dir = config_dir
        pass
    
    def __str__(self):
        return self.gen_circle_str() + self.gen_rectangle_str() + self.gen_triangle_str()
    
    def gen_circle_str(self):
        str_res = ""
        with open(os.path.join(self.config_dir, 'circle.txt'), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_arr = line.split(',')
                x, y, radius, color = line_arr
                x = float(x)
                y = float(y)
                radius = float(radius)
                circle = Circle(x, y, radius, color)
                str_res += str(circle)

        return str_res
        
    
    def gen_rectangle_str(self):
        str_res = ""
        with open(os.path.join(self.config_dir, 'rectangle.txt'), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_arr = line.split(',')
                x0, y0 = float(line_arr[0]), float(line_arr[1])
                x2, y2 = float(line_arr[2]), float(line_arr[3])
                color = line_arr[-1]
                line0 = Line(x0, y0, x2, y0)
                line1 = Line(x2, y0, x2, y2)
                line2 = Line(x2, y2, x0, y2)
                line3 = Line(x0, y2, x0, y0)
                rect = Rectangle([line0, line1, line2, line3], color)
                str_res += str(rect)

        return str_res
        
    def gen_triangle_str(self):
        str_res = ""
        with open(os.path.join(self.config_dir, 'triangle.txt'), 'r') as f:
            for line in f.readlines():
                line = line.replace('\n', '')
                line_arr = line.split(',')
                x0, y0 = float(line_arr[0]), float(line_arr[1])
                x1, y1 = float(line_arr[2]), float(line_arr[3])
                x2, y2 = float(line_arr[4]), float(line_arr[5])
                color = line_arr[-1]
                line1 = Line(x0, y0, x1, y1)
                line2 = Line(x1, y1, x2, y2)
                line3 = Line(x2, y2, x0, y0)
                tria = Triangle([line1, line2, line3], color)
                str_res += str(tria)

        return str_res

        

def judge_tri(p1, p2, p3):
    l1 = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    l2 = sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
    l3 = sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
    l = sorted([l1, l2, l3])
    
    if p1[0] == p2[0] and p2[0] == p3[0]:
        return False
    elif p1[1] == p2[1] and p2[1] == p3[1]:
        return False
    elif l[0] + l[1] <= l[2]:
        return False
    else:
        return True

def gen_config(num_circles, num_rectangles, num_triangles, config_dir):
    random.seed(SEED)
    with open(os.path.join(config_dir, 'circle.txt'), 'w') as f:
        for i in range(num_circles):
            x = random.uniform(-9.9, 10)
            y = random.uniform(-9.9, 10)
            radius = random.uniform(0.01, 10)
            color = COLOR_ARR[random.randint(0, len(COLOR_ARR)-1)]
            f.write(f'{x},{y},{radius},{color}\n')
            
    with open(os.path.join(config_dir, 'rectangle.txt'), 'w') as f:
        for i in range(num_rectangles):
            x0 = random.uniform(-9.9, 0)
            y0 = random.uniform(0.1, 10)
            x2 = x0 + random.uniform(0.1, 9.9)
            y2 = y0 - random.uniform(0.1, 9.9)
 
            color = COLOR_ARR[random.randint(0, len(COLOR_ARR)-1)]
            f.write(f'{x0},{y0},{x2},{y2},{color}\n')

    with open(os.path.join(config_dir, 'triangle.txt'), 'w') as f:
        for i in range(num_triangles):
            x0 = random.uniform(-9.9, 10)
            y0 = random.uniform(-9.9, 10)
            x1 = random.uniform(-9.9, 10)
            y1 = random.uniform(-9.9, 10)
            x2 = random.uniform(-9.9, 10)
            y2 = random.uniform(-9.9, 10)
            while not judge_tri([x0, y0], [x1, y1], [x2, y2]):
                x0 = random.uniform(-9.9, 10)
                y0 = random.uniform(-9.9, 10)
                x1 = random.uniform(-9.9, 10)
                y1 = random.uniform(-9.9, 10)
                x2 = random.uniform(-9.9, 10)
                y2 = random.uniform(-9.9, 10)
            color = COLOR_ARR[random.randint(0, len(COLOR_ARR)-1)]
            f.write(f'{x0},{y0},{x1},{y1},{x2},{y2},{color}\n')

def visualize(config_dir, save_dir):
    with open(os.path.join(config_dir, 'circle.txt')) as f:
        for i, line in enumerate(f.readlines()):
            line_arr = line.split(',')
            x = float(line_arr[0])
            y = float(line_arr[1])
            r = float(line_arr[2])
            circle = plt.Circle((x, y), r)
            plt.gcf().gca().add_artist(circle)
            plt.axis('equal')
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.savefig(os.path.join(save_dir, f'circle/{i}.png'))
            plt.close()
            
    with open(os.path.join(config_dir, 'rectangle.txt')) as f:
        for i, line in enumerate(f.readlines()):
            line_arr = line.split(',')
            x0 = float(line_arr[0])
            y0 = float(line_arr[1])
            x2 = float(line_arr[2]) 
            y2 = float(line_arr[3])
            h = y0 - y2
            w = x2 - x0
            rect = plt.Rectangle((x0, y2), h, w)
            plt.gcf().gca().add_artist(rect)

            plt.axis('equal')
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.savefig(f'../dataset/data/rectangle/{i}.png')
            plt.close()
    
    with open(os.path.join(config_dir, 'triangle.txt')) as f:
        for i, line in enumerate(f.readlines()):
            line_arr = line.split(',')
            p1 = [float(line_arr[0]), float(line_arr[1])]
            p2 = [float(line_arr[2]), float(line_arr[3])]
            p3 = [float(line_arr[4]), float(line_arr[5])]
            tria = plt.Polygon([p1, p2, p3])
            plt.gcf().gca().add_artist(tria)

            plt.axis('equal')
            plt.xlim(-10, 10)
            plt.ylim(-10, 10)
            plt.savefig(f'../dataset/data/triangle/{i}.png')
            plt.close()        
            
def gen_token_dict(num_count):
    token_dict = {}
    for i in range(num_count):
        token_dict[str(i)] = i
    token_dict['<BEGIN>'] = 21
    token_dict['<END>'] = 22
    token_dict[','] = 23
    token_dict[';'] = 24
    token_dict['Circle']  = 25
    token_dict['Line'] = 26
    for i, color in enumerate(COLOR_ARR):
        token_dict[color] = 27 + i
    return token_dict 
    
def str_to_token(str_path, token_dict_path, save_path):
    random.seed(SEED)
    with open(token_dict_path, 'r') as f:
        token_dict = json.load(f)
      
    token_res = []  
    token_info = []
    label_flag = []
    with open(str_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.replace('(', " ")
            line = line.replace(')', ' ')
            line = line.replace(',', '')
            line = line.replace('\n', '')
            line_arr = line.split(' ')
            line_res = []
            line_info = []
            line_count = 0
            if i % 3 == 1:
                for e in line_arr:
                    try:
                        p = int(e)
                    except ValueError:
                        line_res.append(token_dict[e])
                    if e == 'Line':
                        line_count += 1
                    if e == 'Circle' or e in COLOR_ARR:
                        line_info.append(e)
                    if line_count == 3 and len(line_info) == 0:
                        line_info.append('Triangle')
                    if line_count == 4:
                        line_info[0] = 'Rectangle'
                flag = True
            else:
                for e in line_arr:
                    try:
                        p = int(e)
                    except ValueError:
                        if e not in token_res:
                            line_res.append(token_dict[e])
                        else:
                            continue
                    if e == 'Circle':
                        if random.random() > 0.5:
                            line_info.append('Triangle')
                        else:
                            line_info.append('Rectangle')
                    if e in COLOR_ARR:
                        e_idx = COLOR_ARR.index(e)
                        color_idx = random.randint(0, len(COLOR_ARR)-1)
                        while color_idx == e_idx:
                            color_idx = random.randint(0, len(COLOR_ARR)-1)
                        line_info.append(COLOR_ARR[color_idx])
                    if e == 'Line':
                        line_count += 1
                    if line_count == 3 and len(line_info) == 0:
                        if random.random() > 0.5:
                            line_info.append('Circle')
                        else:
                            line_info.append('Rectangle')
                    if line_count == 4:
                        if random.random() > 0.5:
                            line_info.append('Circle')
                        else:
                            line_info.append('Triangle')
                flag = False
                    
            for _ in range(MAX_TOKEN_LEN-len(line_res)):
                line_res.append(PAD)
            # if line_info not in token_info:
            token_res.append(line_res)
            token_info.append(line_info)
            label_flag.append(flag)

    np.savetxt('../config/token_config/token.txt', token_res, fmt='%i')
    with open('../config/token_config/token_info.txt', 'w') as f:
        for i, info in enumerate(token_info):
            f.write(f'{info[0]} {info[1]} {label_flag[i]}\n')
    # with open('../config/token_config/token.txt', 'wb') as f:
    #     np.save(f, np.asarray(token_res))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='The path we save our config', \
                        default='/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/config/token_config')
    parser.add_argument('--num_circles', type=int, help='The number of the circles', default=2000)
    parser.add_argument('--num_rectangles', type=int, help='The number of the rectangles', default=2000)
    parser.add_argument('--num_triangles', type=int, help='The number of the triangles', default=2000)
    parser.add_argument('--gen_cfg', type=bool, help='Whether we generate the config', default=False)
    parser.add_argument('--vis_save_dir', type=str, help='Where to save our visualization result.', default='../dataset/data')
    parser.add_argument('--num_count', type=int, help='The range of the bin count from 0', default=20)
    args = parser.parse_args()
    
    if args.gen_cfg:    
        gen_config(args.num_circles, args.num_rectangles, args.num_triangles, args.config_path)
        # visualize(args.config_path, args.vis_save_dir)
    
        token_gen = TokenGenerator(args.num_circles, args.num_rectangles, args.num_triangles, args.config_path)
        gen_str = str(token_gen)
        with open(os.path.join(args.config_path, 'str.txt'), 'w') as f:
            f.write(gen_str)
        
        token_dict = gen_token_dict(args.num_count)
        with open(os.path.join(args.config_path, 'token_dict.json'), 'w') as f:
                json.dump(token_dict, f)

    str_to_token(os.path.join(args.config_path, 'str.txt'), os.path.join(args.config_path, 'token_dict.json'),
            os.path.join(args.config_path, 'token.txt'))