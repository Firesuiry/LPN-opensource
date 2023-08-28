import logging
import sys
import time

localtime = time.strftime("%Y-%m-%d-%Hx%Mx%S", time.localtime())

# 创建一个日志器logger并设置其日志级别为DEBUG
logger = logging.getLogger('')
logger.setLevel(logging.DEBUG)

# 创建一个流处理器handler并设置其日志级别为DEBUG
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)

# 创建一个格式器formatter并将其添加到处理器handler
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

# 创建文件处理器
fh = logging.FileHandler(f'logs/{localtime}.log', encoding='utf-8')
# fh = logging.FileHandler(f'logs/logs.log',encoding='utf-8')
fh.setFormatter(formatter)
logger.addHandler(fh)


# 为日志器logger添加上面创建的处理器handler
logger.addHandler(handler)

if __name__ == '__main__':
    logger.info('test log')
