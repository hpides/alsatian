from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshotNode, MultiModelSnapshotEdge


def mm_snapshot_dfs(mm_root: MultiModelSnapshotNode, order=None):
    if order is None:
        order = []

    stack = list(reversed(mm_root.edges))
    while stack:
        current_edge = stack.pop()
        order.append(current_edge)
        current_node = current_edge.child
        stack += list(reversed(current_node.edges))

    return order

if __name__ == '__main__':
    node1 = MultiModelSnapshotNode(layer_state="Node 1")
    node2 = MultiModelSnapshotNode(layer_state="Node 2")
    node3 = MultiModelSnapshotNode(layer_state="Node 3")
    node4 = MultiModelSnapshotNode(layer_state="Node 4")
    node5 = MultiModelSnapshotNode(layer_state="Node 5")
    node6 = MultiModelSnapshotNode(layer_state="Node 6")
    node7 = MultiModelSnapshotNode(layer_state="Node 7")
    node8 = MultiModelSnapshotNode(layer_state="Node 8")
    node9 = MultiModelSnapshotNode(layer_state="Node 9")

    # Create edges
    edge1 = MultiModelSnapshotEdge("Edge 1-2", node1, node2)
    edge2 = MultiModelSnapshotEdge("Edge 1-3", node1, node3)
    edge3 = MultiModelSnapshotEdge("Edge 2-4", node2, node4)
    edge4 = MultiModelSnapshotEdge("Edge 2-5", node2, node5)
    edge5 = MultiModelSnapshotEdge("Edge 3-6", node3, node6)
    edge6 = MultiModelSnapshotEdge("Edge 3-7", node3, node7)
    edge7 = MultiModelSnapshotEdge("Edge 4-8", node4, node8)
    edge8 = MultiModelSnapshotEdge("Edge 5-9", node5, node9)

    # Connect edges to nodes
    node1.edges.extend([edge1, edge2])
    node2.edges.extend([edge3, edge4])
    node3.edges.extend([edge5, edge6])
    node4.edges.append(edge7)
    node5.edges.append(edge8)

    # Run DFS
    order = mm_snapshot_dfs(node1)
    print([e.info for e in order])