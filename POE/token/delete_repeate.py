import os

def delete_repeat(file_path, save_path):
    res = []
    res_no_idx = []
    with open(file_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            line = line.replace('\n', '')
            line_arr = line.split(' ')
            e = f'{line_arr[0]} {line_arr[1]} {i}\n'
            e_no_idx = f'{line_arr[0]} {line_arr[1]}'
            if e_no_idx not in res_no_idx:
                res.append(e)
                res_no_idx.append(e_no_idx)
                
    with open(save_path, 'w') as f:
        for e in res:
            f.write(e)
            
if __name__ == "__main__":
    delete_repeat('../config/token_config/token_info.txt', '../config/token_config/token_info_no_repeat.txt')
    