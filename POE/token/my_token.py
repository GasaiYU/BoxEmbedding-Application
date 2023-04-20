import json
from math import ceil, sqrt
import random
import os
import argparse
import re

import matplotlib.pyplot as plt

import numpy as np

MAX_TOKEN_LEN = 35


SEED = 20010628
BOLD = ['BOLD_NULL' ,'THIN', 'THICK']
LINE_COLOR_ARR = ['NULL' ,'RED', "BLUE", 'GREEN', 'PURPLE', 'BLACK']
COLOR_ARR = ['RED', "BLUE", 'GREEN', 'PURPLE', 'BLACK']
SHAPE_ARR = ['Circle', 'Rectangle', 'Triangle']
color_map_dict = {'RED':'r', 'BLUE':'b', 'GREEN':'g', 'PURPLE':'m', 'BLACK':'k'}
PAD = 32

class Circle(object):
    """
    A circle object. To demonstrate a circle, we need to specify its radius
    and its center coordinate.
    We divide the x, y (-10 < x, y < 10) and radius (0 < radius < 10) into 20 bins
    to convert the float num to int.
    """
    def __init__(self, x, y, radius, color, line_color=None, bold=None):
 
        assert x > -10 and x <= 10
        assert y > -10 and y <= 10
        assert radius > 0 and radius <= 10
        
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        
        self.line_color = line_color
        self.bold = bold

    def __str__(self):
        x = ceil(self.x) + 9
        y = ceil(self.y) + 9
        radius = ceil(self.radius*2) - 1
        color = self.color
        line_color = self.line_color
        bold = self.bold
        return f"<BEGIN> Circle({x}, {y}, {radius}); CIRCLE_LINE_{line_color} CIRCLE_LINE_{bold} {color} <END>\n"
    
class Line(object):
    """
    A circle object. To demonstrate a line, we need to specify its start point
    and its end point.
    We divide the x, y (-10 < x, y < 10) into 20 bins to convert the float num to int.
    """
    def __init__(self, x0, y0, x1, y1, color=None, bold=None, idx=1):
        assert x0 > -10 and x0 <= 10
        assert y0 > -10 and y0 <= 10
        assert x1 > -10 and x1 <= 10
        assert y1 > -10 and y1 <= 10
        
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1  

        self.color = color
        self.bold = bold
        self.idx = idx
    
    def __str__(self):
        x0 = ceil(self.x0) + 9
        y0 = ceil(self.y0) + 9
        x1 = ceil(self.x1) + 9
        y1 = ceil(self.y1) + 9
        
        color = self.color
        bold = self.bold
        idx = self.idx
        
        return f"Line({x0}, {y0}, {x1}, {y1}) LINE{idx}_{color} LINE{idx}_{bold} \n"
    
    
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
        return f"<BEGIN> {str(self.line1)[:-1]}; {str(self.line2)[:-1]}; " \
                + f"{str(self.line3)[:-1]}; {self.color} <END>\n"
    
    
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
                x, y, radius, color, line_color, bold = line_arr
                x = float(x)
                y = float(y)
                radius = float(radius)
                circle = Circle(x, y, radius, color, line_color, bold)
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
                color = line_arr[-9]
                line_color1, line_color2, line_color3, line_color4 = line_arr[-8:-4]
                line_bold1, line_bold2, line_bold3, line_bold4 = line_arr[-4:]
                line0 = Line(x0, y0, x2, y0, line_color1, line_bold1, 1)
                line1 = Line(x2, y0, x2, y2, line_color2, line_bold2, 2)
                line2 = Line(x2, y2, x0, y2, line_color3, line_bold3, 3)
                line3 = Line(x0, y2, x0, y0, line_color4, line_bold4, 4)
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
                color = line_arr[-7]
                line_color1, line_color2, line_color3 = line_arr[-6:-3]
                line_bold1, line_bold2, line_bold3 = line_arr[-3:]
                line1 = Line(x0, y0, x1, y1, line_color1, line_bold1, 1)
                line2 = Line(x1, y1, x2, y2, line_color2, line_bold2, 2)
                line3 = Line(x2, y2, x0, y0, line_color3, line_bold3, 3)
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
            bold = BOLD[random.randint(1, len(BOLD)-1)]
            line_color = LINE_COLOR_ARR[random.randint(1, len(LINE_COLOR_ARR)-1)]
            f.write(f'{x},{y},{radius},{color},{line_color},{bold}\n')
            
    with open(os.path.join(config_dir, 'rectangle.txt'), 'w') as f:
        for i in range(num_rectangles):
            x0 = random.uniform(-9.9, 0)
            y0 = random.uniform(0.1, 10)
            x2 = x0 + random.uniform(0.1, 9.9)
            y2 = y0 - random.uniform(0.1, 9.9)
 
            color = COLOR_ARR[random.randint(0, len(COLOR_ARR)-1)]
            line_colors = []
            line_bolds = []
            for i in range(4):
                line_colors.append(LINE_COLOR_ARR[random.randint(1, len(LINE_COLOR_ARR)-1)])
                line_bolds.append(BOLD[random.randint(1, len(BOLD)-1)])
            
            f.write(f'{x0},{y0},{x2},{y2},{color},{line_colors[0]},{line_colors[1]},{line_colors[2]},{line_colors[3]},' + 
                    f'{line_bolds[0]},{line_bolds[1]},{line_bolds[2]},{line_bolds[3]}\n')

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
            line_colors = []
            line_bolds = []
            for i in range(4):
                line_colors.append(LINE_COLOR_ARR[random.randint(1, len(LINE_COLOR_ARR)-1)])
                line_bolds.append(BOLD[random.randint(1, len(BOLD)-1)])
            f.write(f'{x0},{y0},{x1},{y1},{x2},{y2},{color},{line_colors[0]},{line_colors[1]},{line_colors[2]},'
                    f'{line_bolds[0]},{line_bolds[1]},{line_bolds[2]}\n')

def visualize(config_dir, save_dir):
    with open(os.path.join(config_dir, 'circle.txt')) as f:
        for i, line in enumerate(f.readlines()):
            line_arr = line.split(',')
            x = float(line_arr[0])
            y = float(line_arr[1])
            r = float(line_arr[2])
            c = str(line_arr[-1][:-1])
            circle = plt.Circle((x, y), r, color=color_map_dict[c], fill=True)
            plt.gcf().gca().add_artist(circle)
            plt.axis('equal')
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
            plt.savefig(os.path.join(save_dir, f'circle/{i}.png'))
            plt.close()
            
    with open(os.path.join(config_dir, 'rectangle.txt')) as f:
        for i, line in enumerate(f.readlines()):
            line_arr = line.split(',')
            x0 = float(line_arr[0])
            y0 = float(line_arr[1])
            x2 = float(line_arr[2]) 
            y2 = float(line_arr[3])
            c = str(line_arr[-1][:-1])
            h = y0 - y2
            w = x2 - x0
            rect = plt.Rectangle((x0, y2), h, w, color=color_map_dict[c], fill=True)
            plt.gcf().gca().add_artist(rect)

            plt.axis('equal')
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
            plt.savefig(f'../dataset/data/rectangle/{i}.png')
            plt.close()
    
    with open(os.path.join(config_dir, 'triangle.txt')) as f:
        for i, line in enumerate(f.readlines()):
            line_arr = line.split(',')
            p1 = [float(line_arr[0]), float(line_arr[1])]
            p2 = [float(line_arr[2]), float(line_arr[3])]
            p3 = [float(line_arr[4]), float(line_arr[5])]
            c = str(line_arr[-1][:-1])
            tria = plt.Polygon([p1, p2, p3], color=color_map_dict[c], fill=True)
            plt.gcf().gca().add_artist(tria)

            plt.axis('equal')
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)
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
    token_dict['PAD'] = 32
    for i, color in enumerate(LINE_COLOR_ARR):
        token_dict[f'CIRCLE_LINE_{color}'] = 33 + i

    for i, bold in enumerate(BOLD):
        token_dict[f'CIRCLE_LINE_{bold}'] = 39 + i
        
    for i in range(4):
        for j, color in enumerate(LINE_COLOR_ARR):
            token_dict[f'LINE{i+1}_{color}'] = 42 + i * 6 + j
    
    for i in range(4):
        for j, bold in enumerate(BOLD):
            token_dict[f'LINE{i+1}_{bold}'] = 66 + i * 3 + j
        
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
            flag = False
            tria_rect_flag = False
            if True:
                for e in line_arr:
                    if e == '':
                        continue
                    line_res.append(token_dict[e])
                    if e == 'Line':
                        line_count += 1
                    if e == 'Circle' or e in COLOR_ARR:
                        if 'Triangle' in line_info:
                            line_info.append('LINE4_NULL')
                            line_info.append('LINE4_BOLD_NULL')
                        line_info.append(e)
                    if line_count == 3 and not tria_rect_flag:
                        line_info.insert(0, "Triangle")
                        tria_rect_flag = True   
                    if line_count == 4:
                        line_info[0] = 'Rectangle'
                    if e.startswith('LINE1'):
                        if not flag:
                            line_info.append('CIRCLE_LINE_NULL')
                            line_info.append('CIRCLE_LINE_BOLD_NULL')
                            flag = True
                        line_info.append(e)
                    if e.startswith('LINE2') or e.startswith('LINE3') or e.startswith('LINE4'):
                        line_info.append(e)
                    if e.startswith('CIRCLE_LINE_THIN') or e.startswith('CIRCLE_LINE_THICK') or e.startswith('CIRCLE_LINE_BOLD_NULL'):
                        line_info.append(e)
                        if not flag:
                            line_info.append('LINE1_NULL')
                            line_info.append('LINE1_BOLD_NULL')
                            line_info.append('LINE2_NULL')
                            line_info.append('LINE2_BOLD_NULL')
                            line_info.append('LINE3_NULL')
                            line_info.append('LINE3_BOLD_NULL')
                            line_info.append('LINE4_NULL')
                            line_info.append('LINE4_BOLD_NULL')
                            flag = True
                    elif e.startswith('CIRCLE'):
                        line_info.append(e)
                label_flag.append(True)

            else:
                for e in line_arr:
                    line_res.append(token_dict[e])
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
                            line_info[0] = 'Circle'
                        else:
                            line_info[0] = 'Triangle'
                label_flag.append(False)
                    
            for _ in range(MAX_TOKEN_LEN-len(line_res)):
                line_res.append(PAD)
            token_res.append(line_res)
            token_info.append(line_info)

    np.savetxt('../config/token_config/token.txt', token_res, fmt='%i')
    with open('../config/token_config/token_info.txt', 'w') as f:
        for i, info in enumerate(token_info):
            res_str = ''
            for e in info:
                res_str = res_str + e + ' '
            f.write(f'{res_str}{label_flag[i]}\n')
    # with open('../config/token_config/token.txt', 'wb') as f:
    #     np.save(f, np.asarray(token_res))
    
def gen_op_dict(save_path):

    count = 0
    op_dict = {}
    for i in range(len(SHAPE_ARR)):
        for j in range(len(SHAPE_ARR)):
            if True:
                op_dict[f"{SHAPE_ARR[i]} {SHAPE_ARR[j]}"] = count
                count += 1
    
    for j in range(len(LINE_COLOR_ARR)):
        for k in range(len(LINE_COLOR_ARR)):
            if True:
                op_dict[f"CIRCLE_LINE_{LINE_COLOR_ARR[j]} CIRCLE_LINE_{LINE_COLOR_ARR[k]}"] = count
                count += 1
                
    for j in range(len(BOLD)):
        for k in range(len(BOLD)):
            if True:
                op_dict[f"CIRCLE_LINE_{BOLD[j]} CIRCLE_LINE_{BOLD[k]}"] = count
                count += 1
    
                
    for i in range(4):
        for j in range(len(LINE_COLOR_ARR)):
            for k in range(len(LINE_COLOR_ARR)):
                if True:
                    op_dict[f"LINE{i+1}_{LINE_COLOR_ARR[j]} LINE{i+1}_{LINE_COLOR_ARR[k]}"] = count
                    count += 1
                        
        for j in range(len(BOLD)):        
            for k in range(len(BOLD)):
                if True:
                    op_dict[f"LINE{i+1}_{BOLD[j]} LINE{i+1}_{BOLD[k]}"] = count
                    count += 1
                    
    for i in range(len(COLOR_ARR)):
        for j in range(len(COLOR_ARR)):
            if True:
                op_dict[f"{COLOR_ARR[i]} {COLOR_ARR[j]}"] = count
                count += 1
            
                
    with open(save_path, 'w') as f:
        json.dump(op_dict, f)

def save_no_repeat(info_path, save_path):
    res = []
    res_no_idx = []
    with open(info_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.replace('\n', '')
            line_arr = line.split(' ')
            e = ''
            e_no_idx = ''
            for attr in line_arr[:-1]:
                e = e + attr + ' '
                e_no_idx = e + attr + ' '
            e_no_idx = e_no_idx[:-1]
            e = e + str(i) + '\n'
            if e_no_idx not in res_no_idx:
                res.append(e)
                res_no_idx.append(e_no_idx)
                
    with open(save_path, 'w') as f:
        for e in res:
            f.write(e)
    
def file_line_content(file_path, line_num):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == line_num:
                line = line.replace('\n', '')
                return line
            
def generate_operator_txt(dict_path, info_path, token_path, save_path):
    with open(dict_path, 'r') as f:
        op_dict = json.load(f)
        
    f_info = open(info_path, 'r')
    
    info_res = []
    with open(info_path, 'r') as f_info:
        for line in f_info.readlines():
            line = line.replace('\n', '')
            info_res.append(line)

    with open(save_path, 'w') as f:
        for i in range(len(info_res)):
            count = 0
            while count < 3:
                j = random.randint(0, len(info_res) - 1)
                if i != j:
                    flag = False
                    line1 = info_res[i]
                    line1_arr = line1.split(' ')
                    line2 = info_res[j]
                    line2_arr = line2.split(' ')
                    token1 = file_line_content(token_path, int(line1_arr[-1]))
                    token2 = file_line_content(token_path, int(line2_arr[-1]))
                    # if line1_arr[0] == line2_arr[0] or line1_arr[1] == line2_arr[1]:
                    #     continue
                    ops = []
                    for k in range(12):
                        if line1_arr[k] == line2_arr[k]:
                            flag = True
                            break
                        ops.append(op_dict[f'{line1_arr[k]} {line2_arr[k]}'])
                    if flag:
                        break
                    res = f'{token1},'
                    for op in ops:
                        res = res + str(op) + ','
                    res = res + str(token2) + '\n'
                    count += 1
                    
                    f.write(res)

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='The path we save our config', \
                        default='/lustre/S/gaomj/bachelor/BoxEmbedding-Application/POE/config/token_config')
    parser.add_argument('--num_circles', type=int, help='The number of the circles', default=1000)
    parser.add_argument('--num_rectangles', type=int, help='The number of the rectangles', default=1000)
    parser.add_argument('--num_triangles', type=int, help='The number of the triangles', default=1000)
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
        
    gen_op_dict(os.path.join(args.config_path, 'multi_op_dict.json'))
    save_no_repeat(os.path.join(args.config_path, 'token_info.txt'), os.path.join(args.config_path, 'multi_no_repeat_info.txt'))
    # gen_operator_index(os.path.join(args.config_path, 'operator_index.json'))
    generate_operator_txt(os.path.join(args.config_path, 'multi_op_dict.json'),
                         os.path.join(args.config_path, 'multi_no_repeat_info.txt'),
                         os.path.join(args.config_path, 'token.txt'),
                         os.path.join(args.config_path, 'multi_op.txt'))
    