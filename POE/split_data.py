import random

with open('data.txt', 'r') as f:
    line_num = len(f.readlines())


train_data_num = int(line_num * 0.8)
shuffle_line = list(range(0, train_data_num))
random.shuffle(shuffle_line)
train_line = shuffle_line[0: train_data_num]
test_line = shuffle_line[train_data_num:]

with open('data.txt', 'r') as f, open('train.txt', 'w') as f1, open('test.txt', 'w') as f2:
    for i, line in enumerate(f.readlines()):
        if i in train_line:
            f1.write(line)
        else:
            f2.write(line)
