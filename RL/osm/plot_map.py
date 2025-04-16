import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import collections  as mc

###############
idx, x, y = np.loadtxt('./txt/node.txt', delimiter=' ', \
                usecols=(0, 1, 2), unpack=True, \
                dtype={'names': ('id', 'lat', 'lon'),
                        'formats': ('i4', 'f8', 'f8')})
n1, n2, length = np.loadtxt('./txt/edge.txt', delimiter=' ', \
                usecols=(0, 1, 2), unpack=True, \
                dtype={'names': ('n1', 'n2', 'length'),
                        'formats': ('i8', 'i8', 'f8')})



x_dict = dict(zip(idx, x))
y_dict = dict(zip(idx, y))


####### try1 #######
x_coord_list = []
y_coord_list = []

idx = -1

for node1, node2 in zip(n1, n2):
    x_coord_list.append([x_dict[node1], x_dict[node2]])
    y_coord_list.append([y_dict[node1], y_dict[node2]])
   
fig, ax = plt.subplots(1,1)
ax.scatter(x, y, s=1)
# plt.plot(np.array(x_coord_list), np.array(y_coord_list), 'r-')
for node1, node2 in zip(n1[:idx], n2[:idx]):
    ax.plot([x_dict[node1], x_dict[node2]], [y_dict[node1], y_dict[node2]], 'r-')
ax.axis('equal')
plt.savefig('whole_graph.jpg')


# ######## try2 #######
# data = []
# for node1, node2 in zip(data_edge['u'], data_edge['v']):
#     data.append((x_dict[node1], x_dict[node2]))
#     data.append((y_dict[node1], y_dict[node2]))
#     data.append('r')
    
# plt.figure()
# plt.scatter(data_node['x'], data_node['y'], s=1)
# plt.plot(*data)
# plt.show()


# ######## try3 #######
# lines = []
# for node1, node2 in zip(data_edge['u'], data_edge['v']):
#     lines.append([(x_dict[node1], x_dict[node2]), (y_dict[node1], y_dict[node2])])
# c = np.array([(1, 0, 0, 1)] * len(lines))
# lc = mc.LineCollection(lines, colors=c, linewidths=2)


# fig, ax = plt.subplots()
# ax.scatter(data_node['x'], data_node['y'], s=1)
# ax.add_collection(lc)
# plt.show()