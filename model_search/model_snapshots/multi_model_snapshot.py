from model_search.model_snapshots.rich_snapshot import RichModelSnapshot, LayerState


# This class represents two things

# On the one hand it represents a set of model snapshots merged into one multi-model snapshot. In this case
# - a node/vertex represents a layer of a model snapshot (and can belong to multiple base models)
# - an edge represents the connection to other layers (while in a regular model one layer will often only point to one
# next layer, in this case a layer can point to multiple next layers of different model snapshots)

# On the other hand it represents an execution tree that is the base for our planning algorithm. In this case
# - a node/vertex represents a (persisted intermediate) OUTPUT of the corresponding layer in the multi-model snapshot
# - an edge that represent takling the (persisted) output of layer i (parent) to produce compute layer i+1 (child) and save its output

class MultiModelSnapshotNode:

    def __init__(self, layer_state=None):
        self.layer_state: LayerState = layer_state
        self.edges: [MultiModelSnapshotEdge] = []
        self.snapshot_ids = []

    def get_referencing_edge(self, snapshot_id):
        matching_edges = [e for e in self.edges if e.references_snapshot_ids(snapshot_id)]
        assert len(matching_edges) == 1
        return matching_edges[0]


class MultiModelSnapshotEdge:

    def __init__(self, info, parent, child):
        self.info = info
        self.parent: MultiModelSnapshotNode = parent
        self.child: MultiModelSnapshotNode = child

    def references_snapshot_ids(self, snapshot_ids):
        return snapshot_ids in self.child.snapshot_ids


class MultiModelSnapshot:

    def __init__(self):
        self.root: MultiModelSnapshotNode = MultiModelSnapshotNode()
        self.snapshot_ids = set()

    def add_snapshot(self, snapshot: RichModelSnapshot):
        # the assumption is that both models have the same input and thus the same root node
        self.snapshot_ids.add(snapshot.id)

        # if there is no model added so far just add it as a child to the root node
        if len(self.root.edges) == 0:
            self._append_layers_to_node(self.root, snapshot.layer_states, snapshot.id)

        else:
            self._merge_layers_in_model(self.root, snapshot.layer_states, snapshot.id)

    def prune_snapshot(self, snapshot_id, root=None):
        self.snapshot_ids.discard(snapshot_id)

        # logic: do a "guided" DFS, once we find first node that has only one id equivalent to the given id prune entire branch
        if root is None:
            root = self.root

        # if id not found, nothing to prune
        if snapshot_id in root.snapshot_ids:
            root.snapshot_ids.remove(snapshot_id)
            edge = root.get_referencing_edge(snapshot_id)
            new_root = edge.child
            if new_root.snapshot_ids == [snapshot_id]:
                # if this branch only represents the snapshot we want to prune -> prune
                root.edges.remove(edge)
            else:
                self.prune_snapshot(snapshot_id, new_root)

    def prune_snapshots(self, snapshot_ids):
        for _id in snapshot_ids:
            self.prune_snapshot(_id)

    def _append_layers_to_node(self, prev_node, layers, snapshot_id):
        prev = prev_node
        prev.snapshot_ids.append(snapshot_id)
        for current_layer in layers:
            # connect current layer to previous layer
            new_node = MultiModelSnapshotNode(current_layer)
            edge = MultiModelSnapshotEdge("", prev, new_node)
            prev.edges.append(edge)
            new_node.snapshot_ids.append(snapshot_id)

            prev = new_node

    def _merge_layers_in_model(self, current_root: MultiModelSnapshotNode, layer_states: [LayerState],
                               snapshot_id: str):
        if len(layer_states) > 0:
            # compare if we find matching child
            multi_model_children = [edge.child for edge in current_root.edges]
            current_layer_state = layer_states[0]
            match = self._find_layer_match(multi_model_children, current_layer_state)

            if match is not None:
                # if we find matching child
                # mark that current node belongs to multiple models
                current_root.snapshot_ids.append(snapshot_id)
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
