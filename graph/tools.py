import numpy as np

def edge2mat(link, num_node):
    A = np.zeros((num_node, num_node))
    for i, j in link:
        A[j, i] = 1
    return A

def normalize_digraph(A):
    Dl = np.sum(A, 0)
    h, w = A.shape
    Dn = np.zeros((w, w))
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    AD = np.dot(A, Dn)
    return AD

def get_spatial_graph(num_node, hierarchy):
    A = []
    for i in range(len(hierarchy)):
        A.append(normalize_digraph(edge2mat(hierarchy[i], num_node)))

    A = np.stack(A)

    return A

def get_spatial_graph_original(num_node, self_link, inward, outward):
    I = edge2mat(self_link, num_node)
    In = normalize_digraph(edge2mat(inward, num_node))
    Out = normalize_digraph(edge2mat(outward, num_node))
    A = np.stack((I, In, Out))
    return A

def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)

def get_graph(num_node, edges):

    I = edge2mat(edges[0], num_node)
    Forward = normalize_digraph(edge2mat(edges[1], num_node))
    Reverse = normalize_digraph(edge2mat(edges[2], num_node))
    A = np.stack((I, Forward, Reverse))
    return A # 3, 25, 25

def get_hierarchical_graph(num_node, edges):
    A = []
    for edge in edges:
        A.append(get_graph(num_node, edge))
    A = np.stack(A)
    return A

def get_groups(dataset='NTU', CoM=21):
    groups  =[]
    
    if dataset == 'NTU':

        ### left hand occlusion
        if CoM == 24:
            groups.append([24])
            groups.append([10, 11, 12, 25])
            # groups.append([6, 7, 8, 22, 23])
            groups.append([3, 4, 5, 9, 21])
            groups.append([1, 2, 13, 17])
            groups.append([18, 19, 20])
            groups.append([14, 15, 16])

        ### right hand occlusion
        elif CoM == 22:
            groups.append([22])
            groups.append([6, 7, 8, 23])
            # groups.append([10, 11, 12, 24, 25])
            groups.append([3, 4, 5, 9, 21])
            groups.append([1, 2, 13, 17])
            groups.append([14, 15, 16])
            groups.append([18, 19, 20])
            

        ### right leg occlusion
        elif CoM == 16:
            groups.append([16])
            groups.append([14, 15])
            # groups.append([18, 19, 20])
            groups.append([1, 2, 13, 17])
            groups.append([3, 4, 5, 9, 21])
            groups.append([6, 7, 8, 22, 23])
            groups.append([10, 11, 12, 24, 25])


        ### left leg occlusion
        elif CoM == 20:
            groups.append([20])
            groups.append([18, 19])
            # groups.append([14, 15, 16])
            groups.append([1, 2, 13, 17])
            groups.append([3, 4, 5, 9, 21])
            groups.append([10, 11, 12, 24, 25])
            groups.append([6, 7, 8, 22, 23])

        ### body occlusion
        elif CoM == 11:
            groups.append([1])
            groups.append([13, 17, 3, 4])
            # groups.append([3, 4, 5, 9, 21])
            groups.append([18, 19, 20])
            groups.append([14, 15, 16])
            groups.append([6, 7, 8, 22, 23])
            groups.append([10, 11, 12, 24, 25])

        else:
            raise ValueError()
        
    return groups

def get_edgeset(dataset='NTU', CoM=21):
    groups = get_groups(dataset=dataset, CoM=CoM)
    
    for i, group in enumerate(groups):
        group = [i - 1 for i in group]
        groups[i] = group

    identity = []
    forward_hierarchy = []
    reverse_hierarchy = []

    for i in range(len(groups) - 1):
        self_link = groups[i] + groups[i + 1]
        self_link = [(i, i) for i in self_link]
        identity.append(self_link)
        forward_g = []
        for j in groups[i]:
            for k in groups[i + 1]:
                forward_g.append((j, k))
        forward_hierarchy.append(forward_g)
        
        reverse_g = []
        for j in groups[-1 - i]:
            for k in groups[-2 - i]:
                reverse_g.append((j, k))
        reverse_hierarchy.append(reverse_g)

    edges = []
    for i in range(len(groups) - 1):
        edges.append([identity[i], forward_hierarchy[i], reverse_hierarchy[-1 - i]])
    # print("\n edges[0] :", edges[0])
    # print("\n edges[1] :", edges[1])
    # print("\n edges[2] :", edges[2])
    # print("\n edges[3] :", edges[3])
    # print("\n edges[4] :", edges[4])
    # print("\n edges[5] :", edges[5])
    # exit()
    return edges