import json
import random
import matplotlib.pyplot as plt
import math

class SketchDataset(object):
    def __init__(self, num_imgs):
        self.num_imgs = num_imgs
        self.num_circles, self.num_tria, self.num_rect = self.num_imgs
        self.circle_labels = {}
        self.rectangle_labels = {}
        self.triangle_labels = {}
        
    def get_tri(self):
        def judge_tri(p1, p2, p3):
            l1 = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            l2 = math.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
            l3 = math.sqrt((p2[0] - p3[0])**2 + (p2[1] - p3[1])**2)
            l = sorted([l1, l2, l3])
            
            if p1[0] == p2[0] and p2[0] == p3[0]:
                return False
            elif p1[1] == p2[1] and p2[1] == p3[1]:
                return False
            elif l[0] + l[1] <= l[2]:
                return False
            else:
                return True
        
        p1 = [random.uniform(1, 19), random.uniform(1, 19)]
        p2 = [random.uniform(1, 19), random.uniform(1, 19)]
        p3 = [random.uniform(1, 19), random.uniform(1, 19)]
        
        while not judge_tri(p1, p2, p3):
            p1 = [random.uniform(1, 19), random.uniform(1, 19)]
            p2 = [random.uniform(1, 19), random.uniform(1, 19)]
            p3 = [random.uniform(1, 19), random.uniform(1, 19)]
            
        return p1, p2, p3
        
        
    def get_imgs(self):
        all_color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        
        for i in range(self.num_tria):
            color = all_color[random.randint(0, 6)]
            self.draw_triangle(color, i)
            
        for i in range(self.num_circles):
            x = random.randint(5, 15)
            y = random.randint(5, 15)
            r = random.uniform(3, 7)
            color = all_color[random.randint(0, 6)]
            self.draw_circle(x, y, r, color, i)
        
        for i in range(self.num_rect):
            x = random.randint(0, 10)
            y = random.randint(0, 10)
            h = random.uniform(2, 12)
            w = random.uniform(2, 12)
            color = all_color[random.randint(0, 6)]
            self.draw_rectangle(x, y, h, w, color, i)
        
        self.dump_labels()
    
    def draw_circle(self, x, y, r, color, index):

        fig = plt.figure(figsize=(5, 5))

        circle = plt.Circle((x, y), r, color=color, fill=True)
        plt.gcf().gca().add_artist(circle)

        plt.axis('equal')
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.savefig(f'../dataset/data/circle/{index}.png')
        self.circle_labels[f'../dataset/data/circle/{index}.png'] = {'x': x, 'y': y, 'r': r, 'color': color}
        plt.close()
        
    def draw_rectangle(self, x, y, h, w, color, index):
        fig = plt.figure(figsize=(5, 5))
        
        rect = plt.Rectangle((x, y), h, w, color=color)
        plt.gcf().gca().add_artist(rect)

        plt.axis('equal')
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.savefig(f'../dataset/data/rectangle/{index}.png')
        self.rectangle_labels[f'../dataset/data/rectangle/{index}.png'] = {'x':x, 'y':y, 'h':h, 'w':w, 'color':color}
        plt.close()
        
    def draw_triangle(self, color, index):
        fig = plt.figure(figsize=(5, 5))
        p1, p2, p3 = self.get_tri()
        tria = plt.Polygon([p1, p2, p3], color=color)
        plt.gcf().gca().add_artist(tria)

        plt.axis('equal')
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.savefig(f'../dataset/data/triangle/{index}.png')
        self.triangle_labels[f'../dataset/data/triangle/{index}.png'] = {'p1': p1, 'p2':p2, 'p3':p3, 'color': color}
        plt.close()
        
    def dump_labels(self):
        with open(f'../dataset/data/labels/circle_label.json', 'w') as f:
            json.dump(self.circle_labels, f)
            
        with open(f'../dataset/data/labels/rectangle_label.json', 'w') as f:
            json.dump(self.rectangle_labels, f)
            
        with open(f'../dataset/data/labels/triangle_label.json', 'w') as f:
            json.dump(self.triangle_labels, f)

if __name__ == "__main__":
    num_imgs = [50,50, 50]
    dataset_drawer = SketchDataset(num_imgs)
    dataset_drawer.get_imgs()