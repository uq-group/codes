import argparse
import networkx as nx
import numpy as np
import time
import copy
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='refine mesh')
parser.add_argument("--percentile", type=int, help="percentile", default=10)
args = parser.parse_args()

rootpath = '.'

G = nx.read_edgelist('{}/txt/edge_connect.txt'.format(rootpath), nodetype=int, data=(('weight',float),))
print("BEFORE: directed: {}".format(nx.is_directed(G)), "connected: {}".format(nx.is_connected(G)))
print("check node id in order: {}".format(sorted(list(G.nodes())) == [i for i in range(len(G.nodes()))]))

# add node attribute --to do in the future, interpolate the lat and lon of new node
idx, lat, lon = np.loadtxt('./txt/node_connect.txt', delimiter=' ', \
                usecols=(0, 1, 2), unpack=True, \
                dtype={'names': ('id', 'lat', 'lon'),
                        'formats': ('i4', 'f8', 'f8')})

lat_dict = dict(zip(idx, lat))
lon_dict = dict(zip(idx, lon))
nx.set_node_attributes(G, lat_dict, "lat")
nx.set_node_attributes(G, lon_dict, "lon")
# print(G.nodes[0]["lat"], G.nodes[0]["lon"], G.edges[0, 1]["weight"])


# check the edge length
_, _, length = np.loadtxt("{}/txt/edge_connect.txt".format(rootpath), dtype={'names': ('n1', 'n2', 'length'),
                     'formats': ('i4', 'i4', 'f8')}, delimiter=' ', unpack=True)
length_5, length_20, length_25, length_50, length_75 = np.percentile(length, 5), np.percentile(length, 20), \
    np.percentile(length, 25), np.percentile(length, 50), np.percentile(length, 75)
print(length_5, length_25, length_50, length_75)
# length_factor = np.maximum(length / length_5, np.ones_like(length)).astype(int)
# print(np.max(length_factor))

curr_node_ID = len(G.nodes()) - 1
G_refine = copy.deepcopy(G)
unit_length = np.percentile(length, args.percentile) ### change percentile ## best choice 15 for now
print("total num: {}, maximum node id: {}".format(curr_node_ID+1, curr_node_ID))
for src, dst in G.edges():
    # print("original edge: ", G.edges[src, dst]["weight"])
    n_new_edge = np.maximum(G.edges[src, dst]["weight"]/unit_length + 0.9, 1).astype(int)  # need to change--done
    n_new_node = n_new_edge - 1
    if n_new_node > 0:
        new_node_ID_list = [i for i in range(curr_node_ID+1, curr_node_ID+n_new_edge)]
        print("new node ID list: {}, # new node: {}".format(new_node_ID_list, n_new_node))
        
        # remove old edge
        G_refine.remove_edge(src, dst)
        print("old edge removed: ({}, {})".format(src, dst))
        
        # add new node
        G_refine.add_nodes_from(new_node_ID_list)

        # add new edge
        newpath = [src, *new_node_ID_list, dst]
        node1, node2 = newpath[:-1], newpath[1:]
        new_edge_list = [(i,j) for i, j in zip(node1, node2)]
        G_refine.add_edges_from(new_edge_list, weight=G.edges[src, dst]["weight"]/n_new_edge)
        print("--------------")
        print(newpath)
        print(new_edge_list)

        # add node feature
        src_lat, src_lon = G_refine.nodes[src]['lat'], G_refine.nodes[src]['lon']
        dst_lat, dst_lon = G_refine.nodes[dst]['lat'], G_refine.nodes[dst]['lon']
        # print(src_lat, src_lon, dst_lat, dst_lon)
        newlat = np.linspace(src_lat, dst_lat, len(newpath))[1:-1]
        newlon = np.linspace(src_lon, dst_lon, len(newpath))[1:-1]
        new_lat_dict = dict(zip(newpath[1:-1], newlat))
        new_lon_dict = dict(zip(newpath[1:-1], newlon))
        nx.set_node_attributes(G_refine, new_lat_dict, "lat")
        nx.set_node_attributes(G_refine, new_lon_dict, "lon")
        # print(G_refine.nodes[0]['lat'], G_refine.nodes[4526]['lat'], G_refine.nodes[4527]['lat'], G_refine.nodes[4528]['lat'], G_refine.nodes[1049]['lat'])
        # print(G_refine.nodes[0]['lon'], G_refine.nodes[4526]['lon'], G_refine.nodes[4527]['lon'], G_refine.nodes[4528]['lon'], G_refine.nodes[1049]['lon'])
        # ALsadf
        curr_node_ID += n_new_node
        print("current_node_ID: {}, #node in Gr: {}, new edge length: {}".format(curr_node_ID, len(G_refine.nodes()), G.edges[src, dst]["weight"]/n_new_edge))
        
print("new graph -- #node: {} max node ID: {} #edge: {}".format(len(G_refine.nodes()), max(G_refine.nodes()), len(G_refine.edges())))
print("AFTER: directed: {}".format(nx.is_directed(G_refine)), "connected: {}".format(nx.is_connected(G_refine)))

# check the histogram of the edge length
edge_e = nx.get_edge_attributes(G_refine, 'weight').values()
plt.figure()
plt.hist(edge_e, bins=50)
plt.savefig('edge_length_hist.jpg')


# create new node.txt
n_node = len(G_refine.nodes())
final_lat_dict = nx.get_node_attributes(G_refine, 'lat')
final_lon_dict = nx.get_node_attributes(G_refine, 'lon')
node_ID = np.array([i for i in range(n_node)])
new_x = np.array([final_lat_dict[i] for i in node_ID])
new_y = np.array([final_lon_dict[i] for i in node_ID])

final_node_list = np.concatenate((node_ID.reshape(-1, 1), new_x.reshape(-1, 1), \
    new_y.reshape(-1, 1)), axis=1)

# create new edge.txt
temp = nx.get_edge_attributes(G_refine, 'weight')
final_edge_list = []
for k, l in temp.items():
    final_edge_list.append([k[0], k[1], l])

np.savetxt('./txt/node.txt', final_node_list, fmt='%i %4.8f %4.8f', delimiter=',')
np.savetxt('./txt/edge.txt', np.vstack(final_edge_list), fmt="%i %i %4.8f", delimiter=',')

_, _, length = np.loadtxt("{}/txt/edge.txt".format(rootpath), dtype={'names': ('n1', 'n2', 'length'),
                     'formats': ('i4', 'i4', 'f8')}, delimiter=' ', unpack=True)
length_1, length_5, length_25, length_50, length_75 = np.percentile(length, 1), np.percentile(length, 5), \
    np.percentile(length, 25), np.percentile(length, 50), np.percentile(length, 75)
print("AFTER: dist", length_5, length_25, length_50, length_75)
