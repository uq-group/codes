import networkx as nx
import numpy as np
import time

rootpath = '.'

G = nx.read_edgelist('{}/txt/edge_raw.txt'.format(rootpath), nodetype=int, data=(('weight',float),))
print("BEFORE: directed: {}".format(nx.is_directed(G)), "connected: {}".format(nx.is_connected(G)))
print("check node id in order: {}".format(sorted(list(G.nodes())) == [i for i in range(len(G.nodes()))]))

node_remove_list = []

for component in list(nx.connected_components(G)):
    # print(len(component))
    # time.sleep(0.1)
    if len(component)<100:
        node_remove_list += component
        G.remove_nodes_from(component)
print("remove num: ", len(node_remove_list), node_remove_list[:10])
print("AFTER: directed: {}".format(nx.is_directed(G)), "connected: {}".format(nx.is_connected(G)))
assert(nx.is_connected(G))

idx, x, y = np.loadtxt('./txt/node_raw.txt', delimiter=' ', \
                usecols=(0, 1, 2), unpack=True, \
                dtype={'names': ('id', 'lat', 'lon'),
                        'formats': ('i4', 'f8', 'f8')})
n1, n2, length = np.loadtxt('./txt/edge_raw.txt', delimiter=' ', \
                usecols=(0, 1, 2), unpack=True, \
                dtype={'names': ('n1', 'n2', 'length'),
                        'formats': ('i8', 'i8', 'f8')})

print("before", idx[node_remove_list[0]])
idx = np.delete(idx, node_remove_list, 0)
x = np.delete(x, node_remove_list, 0)
y = np.delete(y, node_remove_list, 0)
print("after", idx[node_remove_list[0]])

n_node = len(idx)
node_ID = np.array([i for i in range(n_node)])
print("check total num in graph", len(idx), n_node)

# new node txt
node_map = dict(zip(idx, node_ID))
new_x = np.array(x)
new_y = np.array(y)

new_node_list = np.concatenate((node_ID.reshape(-1, 1), new_x.reshape(-1, 1), \
    new_y.reshape(-1, 1)), axis=1)

new_edge_list = []

# new edge txt
for (f_node, t_node, l) in zip(n1, n2, length):
    if f_node in idx and t_node in idx:
        new_edge_list.append([node_map[f_node], node_map[t_node], l])

np.savetxt('./txt/node_connect.txt', new_node_list, fmt='%i %4.8f %4.8f', delimiter=',')
np.savetxt('./txt/edge_connect.txt', np.vstack(new_edge_list), fmt="%i %i %4.8f", delimiter=',')

