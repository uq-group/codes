import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import networkx as nx

def select(fromNA = True):
    if fromNA:
        osmid_raw, x_raw, y_raw = np.loadtxt('./nodes.csv', 
                                skiprows=1,
                                delimiter=' ',
                                unpack=True, 
                                dtype={'names': ('osmid', 'x', 'y'), 'formats': ('i8', 'f8', 'f8')})

        n1, n2, length = np.loadtxt('./edges.csv', delimiter=' ', \
                        usecols=(1, 2, 3), unpack=True, \
                        dtype={'names': ('n1', 'n2', 'length'),
                                'formats': ('i8', 'i8', 'f8')})

        x_dict = dict(zip(osmid_raw, x_raw))
        y_dict = dict(zip(osmid_raw, y_raw))


        ########## map the normalized coord into geo coord
        x_range = (0, 10000)
        lat_range = (-129.9, -63.6)
        y_range = (0, 10000)
        lon_range = (14.0, 55.2)

        x = x_raw/(x_range[1]-x_range[0])*(lat_range[1]-lat_range[0]) + lat_range[0]
        y = y_raw/(y_range[1]-y_range[0])*(lon_range[1]-lon_range[0]) + lon_range[0]


        # ########## locate the region
        upper_left = (40.1, -80.1)
        bottom_right = (35.5, -70.1)

        mask1 = (x > upper_left[1]) & (x < bottom_right[1])
        mask2 = (y > bottom_right[0]) & (y < upper_left[0])
        mask = np.logical_and(mask1, mask2)


        node_id = osmid_raw[mask]
        x, y = x[mask], y[mask]

        print(node_id.shape)

        # fig, ax = plt.subplots(1,1)
        # ax.scatter(x, y, s=1)
        # ax.axis('equal')
        # plt.savefig('org.jpg')

        ########## extract the selected region
        new_edge_list = []
        for temp1, temp2, l in zip(n1, n2, length):
            if (temp1 in node_id) and (temp2 in node_id):
                new_edge_list.append([temp1, temp2, l])

        new_node_list = np.concatenate((node_id.reshape(-1, 1), x.reshape(-1, 1), \
            y.reshape(-1, 1)), axis=1)
        new_edge_list = np.vstack(new_edge_list)

        np.savetxt('txt/node_incomplete.txt', new_node_list, fmt='%i %4.8f %4.8f', delimiter=' ')
        np.savetxt('txt/edge_incomplete.txt', new_edge_list, fmt="%i %i %4.4f", delimiter=' ')
    else:
        osmid, x, y = np.loadtxt('./node_list.csv', 
                          skiprows=1,
                          delimiter=',',
                          unpack=True, 
                          dtype={'names': ('osmid', 'x', 'y'), 'formats': ('i8', 'f8', 'f8')})

        u, v, oneway, highway, length = np.loadtxt('./edge_list.csv',
                                                    delimiter=',',
                                                    unpack=True, 
                                                    dtype={'names': ('u', 'v', 'oneway', 'highway', 'length'), \
                                                    'formats': ('i8', 'i8', 'i4', 'S20', 'f8')})



        n_node = len(list(set(osmid)))
        node_ID = np.array([i for i in range(n_node)])

        node_map = dict(zip(osmid, node_ID))
        new_u = np.array([node_map[i] for i in u])
        new_v = np.array([node_map[i] for i in v])

        # save txt
        new_node_list = np.concatenate((node_ID.reshape(-1, 1), x.reshape(-1, 1), \
            y.reshape(-1, 1)), axis=1)
        new_edge_list = np.concatenate((new_u.reshape(-1, 1), new_v.reshape(-1, 1), \
            length.reshape(-1, 1)), axis=1)

        np.savetxt('txt/node_incomplete.txt', new_node_list, fmt='%i %4.8f %4.8f', delimiter=' ')
        np.savetxt('txt/edge_incomplete.txt', new_edge_list, fmt="%i %i %4.4f", delimiter=' ')

def refine(firstpass=True):
    if firstpass:
        in_name1, out_name1 = 'txt/node_incomplete.txt', 'txt/node_incomplete2.txt'
        in_name2, out_name2 = 'txt/edge_incomplete.txt', 'txt/edge_incomplete2.txt'
    else:
        in_name1, out_name1 = 'txt/node_incomplete3.txt', 'txt/node_raw.txt'
        in_name2, out_name2 = 'txt/edge_incomplete3.txt', 'txt/edge_raw.txt'
    
    osmid_raw, x_raw, y_raw = np.loadtxt(in_name1,
                            delimiter=' ',
                            unpack=True, 
                            dtype={'names': ('osmid', 'x', 'y'), 'formats': ('i8', 'f8', 'f8')})

    u, v, length = np.loadtxt(in_name2,
                            delimiter=' ',
                            unpack=True, 
                            dtype={'names': ('u', 'v', 'length'), \
                                                'formats': ('i8', 'i8', 'f8')})

    if len(x_raw.tolist()) == len(set(u.tolist()) | set(v.tolist())):
        print("node num == node edge num")
        osmid, x, y = osmid_raw, x_raw, y_raw
        osmid, x, y = np.array(osmid), np.array(x), np.array(y)
    elif len(x_raw.tolist()) > len(set(u.tolist()) | set(v.tolist())):
        print("node num > node edge num")
        osmid, x, y = [], [], []
        all_node_set = set(u.tolist()) | set(v.tolist())
        for temp1, temp2, temp3 in zip(osmid_raw, x_raw, y_raw):
            if temp1 in all_node_set:
                osmid.append(temp1)
                x.append(temp2)
                y.append(temp3)
        osmid, x, y = np.array(osmid), np.array(x), np.array(y)
    elif len(x_raw.tolist()) < len(set(u.tolist()) | set(v.tolist())):
        print("node num < node edge num")
        u_new, v_new, length_new = [], [], []
        for temp1, temp2, temp3 in zip(u, v, length):
            if temp1 in osmid_raw and temp2 in osmid_raw:
                u_new.append(temp1)
                v_new.append(temp2)
                length_new.append(temp3)
        u, v, length = np.array(u_new), np.array(v_new), np.array(length_new)
        osmid, x, y = np.array(osmid_raw), np.array(x_raw), np.array(y_raw)


    print("node number {}".format(len(x.tolist())))
    print("node number in edge {}".format(len(set(u.tolist()) | set(v.tolist()))))

    n_node = len(list(set(osmid)))
    node_ID = np.array([i for i in range(n_node)])

    node_map = dict(zip(osmid, node_ID))
    new_u = np.array([node_map[i] for i in u])
    new_v = np.array([node_map[i] for i in v])

    # save txt
    new_node_list = np.concatenate((node_ID.reshape(-1, 1), x.reshape(-1, 1), \
        y.reshape(-1, 1)), axis=1)
    new_edge_list = np.concatenate((new_u.reshape(-1, 1), new_v.reshape(-1, 1), \
        length.reshape(-1, 1)), axis=1)

    np.savetxt(out_name1, new_node_list, fmt='%i %4.8f %4.8f', delimiter=',')
    np.savetxt(out_name2, new_edge_list, fmt="%i %i %4.4f", delimiter=',')
    
def mergenode():
    G = nx.read_edgelist('txt/edge_incomplete2.txt', nodetype=int, data=(('weight',float),))
    print("BEFORE: directed: {}".format(nx.is_directed(G)), "connected: {}".format(nx.is_connected(G)))
    print("check node id in order: {}".format(sorted(list(G.nodes())) == [i for i in range(len(G.nodes()))]))

    while len(G.nodes()) > 1000:
        nodes_to_remove = [n for n in G.nodes if len(list(G.neighbors(n))) == 2]
        # print("num of node can be removed: {}".format(len(nodes_to_remove)))
        if len(nodes_to_remove) < 500:
            break
        remove_id = np.random.choice(nodes_to_remove, size=500, replace=False)
        weight_dict = nx.get_edge_attributes(G, 'weight')
        # For each of those nodes
        for node in remove_id:
            try:
                # We add an edge between neighbors (len == 2 so it is correct)
                n1, n2 = G.neighbors(node)
                # print(node, n1, n2)
                try:
                    l1 = weight_dict[(n1, node)]
                except: 
                    l1 = weight_dict[(node, n1)]
                    
                try:
                    l2 = weight_dict[(n2, node)]
                except: 
                    l2 = weight_dict[(node, n2)]
                    
                new_length = l1 + l2
                # print(l1, l2, new_weight)
                G.add_edge(*G.neighbors(node), weight=new_length)
                # And delete the node
                G.remove_node(node)
            except:
                pass
    print('after num: {}'.format(len(G.nodes())))

    node_survive = G.nodes()
    weight_dict = nx.get_edge_attributes(G, 'weight')
    ## plot the graph afterward
    idx, x, y = np.loadtxt('./txt/node_incomplete2.txt', delimiter=' ', \
                usecols=(0, 1, 2), unpack=True, \
                dtype={'names': ('id', 'lat', 'lon'),'formats': ('i4', 'f8', 'f8')})
    # save the new node list and edge list
    new_edge_list = []
    new_node_list = np.hstack((np.array(node_survive).reshape(-1, 1), \
        x[node_survive].reshape(-1, 1), y[node_survive].reshape(-1, 1)))
    for n1, n2 in G.edges():
        try:
            l = weight_dict[(n1, n2)]
        except:
            l = weight_dict[(n2, n1)]
        new_edge_list.append([n1, n2, l])
    new_edge_list = np.vstack(new_edge_list)    
    print(new_node_list.shape)
    print(new_edge_list.shape)
    np.savetxt('./txt/node_incomplete3.txt', new_node_list, fmt='%i %4.8f %4.8f', delimiter=',')
    np.savetxt('./txt/edge_incomplete3.txt', new_edge_list, fmt="%i %i %4.4f", delimiter=',')
    
fromNA = False
if fromNA:
    # select region from whole NA map
    select(fromNA = True)
    print("finish select")

    # reorder the number
    refine(firstpass=True)
    print("finish refinement")

    # merge the node with degree of one
    mergenode()
    print("finish mergenode")

    # reorder the number
    refine(firstpass=False)
    print("finish refinement")
else:
    select(fromNA = False)
    print("finish select")
    
    # reorder the number
    refine(firstpass=True)
    print("finish refinement")

    # merge the node with degree of one
    mergenode()
    print("finish mergenode")

    # reorder the number
    refine(firstpass=False)
    print("finish refinement")
