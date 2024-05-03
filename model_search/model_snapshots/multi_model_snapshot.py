from model_search.model_snapshots.rich_snapshot import RichModelSnapshot, LayerState


class MultiModelSnapshotNode:

    def __init__(self, layer_state=None):
        self.layer_state: LayerState = layer_state
        self.edges: [MultiModelSnapshotEdge] = []
        self.model_ids = []


class MultiModelSnapshotEdge:

    def __init__(self, info, parent, child):
        self.info = ""
        self.parent: MultiModelSnapshotNode = parent
        self.child: MultiModelSnapshotNode = child


class MultiModelSnapshot:

    def __init__(self):
        self.root: MultiModelSnapshotNode = MultiModelSnapshotNode()

    def add_snapshot(self, snapshot: RichModelSnapshot):
        # the assumption is that both models have the same input and thus the same root node
        # add the new snapshot id to the root node
        self.root.model_ids.append(snapshot._id)

        # if there is no model added so far just add it as a child to the root node
        if len(self.root.edges) == 0:
            self._append_layers_to_node(self.root, snapshot.layer_states, snapshot._id)

        else:
            self._merge_layers_in_model(self.root, snapshot.layer_states, snapshot._id)

    def _append_layers_to_node(self, prev_node, layers, snapshot_id):
        prev = prev_node
        for current_layer in layers:
            # connect current layer to previous layer
            new_node = MultiModelSnapshotNode(current_layer)
            edge = MultiModelSnapshotEdge("", prev, new_node)
            prev.edges.append(edge)
            new_node.model_ids.append(snapshot_id)

            prev = new_node

    def _merge_layers_in_model(self, current_root: MultiModelSnapshotNode, layer_states: [LayerState],
                               snapshot_id: str):
        # compare if we find matching child
        multi_model_children = [edge.child for edge in current_root.edges]
        current_layer_state = layer_states[0]
        match = self._find_layer_match(multi_model_children, current_layer_state)

        if match is not None:
            # if we find matching child
            # mark that current node belongs to multiple models
            current_root.model_ids.append(snapshot_id)
            # deduplicate: forget about current layer state and continue merging
            self._merge_layers_in_model(match, layer_states[1:], snapshot_id)
        else:
            # if we do not find matching child create new edge and connect
            self._append_layers_to_node(current_root, layer_states, snapshot_id)

    def _find_layer_match(self, multi_model_children: [MultiModelSnapshotNode], current_layer_state: LayerState):
        for node in multi_model_children:
            if node.layer_state == current_layer_state:
                return node

        return None
