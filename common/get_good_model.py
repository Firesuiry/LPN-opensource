import os
from pathlib import Path
import re

DIR_PATTERN = re.compile(r'RLNBOA-(.*)-')
FILE_PATTERN = re.compile(r'-(.*)\t')


def get_good_model(path=r'/data/good_model.txt'):
    path = Path(path)
    models = []
    with open(path, 'r') as f:
        content = f.read()
    contents = content.split('\n')
    for line in contents:
        if not line:
            continue
        model_file = line.split()[0]
        dir_path = DIR_PATTERN.findall(model_file)[0]
        file_path = model_file.split('-')[-1]
        # print(dir_path, file_path, model_file)
        target = r'rl/{}/{}.h5'.format(dir_path,file_path)
        # print(target,os.path.exists(target))
        models.append(target)
    return models


if __name__ == '__main__':
    ms = get_good_model('/data/good_model_t30.txt')
    print('\r\n'.join(ms))
    pass
    pass