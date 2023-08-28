from pathlib import Path
import numpy as np
exp_dir = Path('example')

files = exp_dir.glob('*.txt')
avgs = []
for file in files:
    with open(file,'r') as f:
        content = f.read()
    rows = content.split('\n')
    # print(rows)
    line_num = int(rows[0].split()[0])
    dims = int(rows[0].split()[1])
    xs = []
    y = []
    for row_index in range(line_num):
        row = rows[row_index+1]
        digits = [int(d) for d in row.split()]
        # print(digits)
        y.append(digits[-1])
        xs.append(digits[:dims])
    xs = np.array(xs)
    y = np.array(y)
    # print(xs,y)
    avgs.append(np.average(xs))
    print(np.average(xs))
print(f'final_avg:{np.average(avgs)}')
