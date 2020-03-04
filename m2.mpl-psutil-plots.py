import time
from itertools import count

import matplotlib as mpl
import matplotlib.pyplot as plt
import psutil
from matplotlib.animation import FuncAnimation

# plt.rcParams['animation.html'] = 'jshtml'

# plt.style.context('seaborn-darkgrid')
i , x, y = 0, [], []
index = count()

plt.style.context('dark_background')
# mpl.rc('lines',linewidth=4, color='k', marker='o',\
#     markerfacecolor='c', markeredgecolor='g',\
#     markeredgewidth=1, markersize=5,\
#     antialiased=True)

def animate(i):
    x.append(next(index))
    y.append(psutil.cpu_percent())

    plt.cla()
    plt.plot(x, y, color='b')
    plt.xlim(index.__reduce__()[1][0]-80, index.__reduce__()[1][0]+20)
    # plt.set_xlim(left=max(0,i-50), right=i+50)
    i+=1

ani = FuncAnimation(plt.gcf(), animate, interval=500)

plt.tight_layout()
plt.show()


"""
# fig = plt.figure() ## an empty figure with no axes
# fig.suptitle('live plotting cpu-usage')
fig, ax = plt.subplots(1, 1)
fig.show()


while(True):
    x.append(i)
    y.append(psutil.cpu_percent())
    ax.plot(x, y, color='r')
    
    # fig.canvas.draw()
    
    ax.set_xlim(left=max(0,i-50), right=i+50)
    time.sleep(0.5)
    i+=1
"""