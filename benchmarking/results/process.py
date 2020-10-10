#!/usr/bin/env python3

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors as mcolor
import pathlib


def get_results(name):
    results = []
    devices = []
    for n, i in enumerate(open(name)):
        if n > 12:
            # It takes 12 lines to get to the actual benchmarking. Above that is information
            i = i.strip().split()
            results.append((i[0][:-1], float(i[1])))
        elif n < 9:
            if i.startswith('\t'):
                for x in i.strip().split(','):
                    x = x.strip().split(':')
                    key = x[0].strip()
                    value = x[1].strip()
                    devices[-1][key] = value
            else:
                devices.append(dict(name=i.strip().split(':')[1].strip()))
    results = sorted(results, key=lambda x: x[1])
    return (devices, results)


running = pathlib.Path('.').glob("*.bench")
devices = []
results = []
for r in running:
    dev, res = get_results(r)
    devices.append(dev)
    results.append(res)

mapper = mpl.cm.ScalarMappable(
    mcolor.PowerNorm(1, 0, max(map(lambda x: x[-1][1], results))), 'plasma')


def print_results(device, results, mapper):
    x = []
    y = []
    for name, time in results:
        x.append(name)
        y.append(time)

    fig = plt.figure()
    ax = plt.axes()

    for x, y in zip(x, y):
        ax.barh(x, y, color=mapper.to_rgba(y))
    ax.set_title("%s: %s @ %s (%s)" %
                 (device[1]['name'], device[1]['Compute units'],
                  device[1]['Clock speed'], device[1]['Compute capability']))
    ax.set_xlabel('time')
    ax.set_ylabel('benchmark')
    fig.savefig(ax.title.get_text() + '.png')
    fig.show()


for device, result in zip(devices, results):
    print_results(device, result, mapper)
input()
