from pathlib import Path
import json

json_dir = Path('../evaluate/data')
outputdir = Path('./res')

def data2img(data,file_name):
    success_rate = 0
    record_len = len(data)
    for i in range(record_len):
        record = data[i]
        best = record['gbest']
        target = record['best']
        success_rate += 1 * (best == target)
    success_rate /= record_len
    print(f'file:{file_name} success:{success_rate}')


if __name__ == '__main__':
    for file in json_dir.glob('*.json'):
        file_name = file.name
        with open(file,'r') as f:
            data = json.loads(f.read())
        data2img(data,file_name)
