import matplotlib.pyplot as plt
import numpy as np
import torch
import re
import os

def get_loss(file_path):
    epoch_num = []
    loss = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.replace('\n', '')
            line_arr = line.split(' ')
            epoch_num.append(int(line_arr[2]))
            loss.append(float(line_arr[-1]))
    return epoch_num, loss

def plot_loss(train_file_path, test_file_path):
    train_epoch_num, train_loss = get_loss(train_file_path)
    test_epoch_num, test_loss = get_loss(test_file_path)

    plt.plot(train_epoch_num, train_loss, color='red')
    plt.plot(test_epoch_num, test_loss, color='green')

    plt.legend(['Train loss', 'Test loss'], loc='upper left')
    plt.savefig('1.png')
    plt.show()
    
    
if __name__ == "__main__":
    plot_loss('../experiment/log_image_train.txt', '../experiment/log_image_test.txt')