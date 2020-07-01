    #print(dist_to_node)
    subgraph_list = []

    tmp_list = list(range(len(ind)))
    for i in range(len(ind)):
      idx_list = list(range(K))

      tmp_list.remove(i)
      for tmp in tmp_list:
        idx_list.remove(tmp)
      tmp_list.append(i)

      subgraph_tmp = subgraph[idx_list, :][:, idx_list]
      subgraph_list.append(subgraph_tmp)

    dist_to_node = []
    for subgraph_i in subgraph_list:
      dist_to_k = ssp.csgraph.shortest_path(subgraph_i, directed=False, unweighted=True)
      dist_to_node.append(dist_to_k[1:, 0])
