from matplotlib import pyplot as plt

data = '''
LOG2ROW	FWHT	GNS
14	0	0
15	0	0
16	0	0
17	0	0
18	0	2.34375
19	0	41.40625
20	0	83.59375
21	0	99.21875
22	5.46875	100
23	39.0625	100
24	100	100
25	100	100
26	100	100
'''
xs = []
y1s = []
y2s = []
y3s = []
for line in data.split('\n'):
    if 'FWHT' in line:
        continue
    if line == '':
        continue
    xs.append(int(line.split()[0]))
    noq = 2 ** int(line.split()[0])
    y1s.append(float(line.split()[1]))
    y2s.append(float(line.split()[2]))
    y3s.append(noq / (1 / 0.501 ** 28))
for i in range(27, 30):
    xs.append(i)
    y1s.append(100)
    y2s.append(100)
    noq = 2 ** i
    t = noq / (1 / 0.501 ** 28) * 100
    y3s.append(min(t, 100))
    print(f'{i} {t} {min(t, 100)}')
# plot
plt.plot(xs, y1s, label='FWHT')
plt.plot(xs, y2s, label='GNS')
plt.plot(xs, y3s, label='Gauss')
# x-axis Log2 of Number of query
plt.xlabel('Log2 of Number of query')
# y-axis Success rate
plt.ylabel('Success rate')
# add grid lines
plt.grid()
plt.legend()
# save to file
plt.savefig('query_plot.png', dpi=200)
