## TODO check sun of children vs current node
from experiments.snapshots.synthetic.generate import RetrainDistribution
from experiments.snapshots.synthetic.generate_sets.generate_set import get_architecture_models
from model_search.execution.planning.execution_tree import execution_tree_from_mm_snapshot
from model_search.model_snapshots.multi_model_snapshot import MultiModelSnapshot

# def generate_execution_plan(self, mm_snapshot: MultiModelSnapshot, train_dataset_range: [int], len_test_data: int,
#                             first_iteration=False, strategy="DFS", model_input_size=3 * 224 * 224) -> ExecutionPlan:
#     execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)

# # TODO move this code to separate experiment file
# # TODO investigate where difference comes from
# possible_traversals = execution_tree.generate_all_traversals_nodes(early_parent_release=False)
# cheapest_max_budget = execution_tree.min_intermediate_cost_for_traversal(early_parent_release=False)
# execution_tree.annotate_intermediates_with_max_path_costs()
# dfs_max_budget = execution_tree.root.accumulated_path_costs
# cost_sequence, node_sequence, edge_sequence = execution_tree.dfs_traversal(cheapest_path_first=True,
#                                                                            return_costs=True)
# # TODO think about how to deal with discrepancy between release of leaf nodes VS not
# # DFS traversal does not release leaf nodes (which somewhat makes sense, but maybe we should add that for
# # consistency OR change all change cost calculation for all traversal to never release leaf nodes)
#
# print("cheapest_max_budget", cheapest_max_budget)
# print("dfs_max_budget", dfs_max_budget)
#
# assert False

if __name__ == '__main__':
    architecture = 'resnet152'
    model_snapshots, model_store = get_architecture_models(
        '/mount-fs/snapshot-sets', RetrainDistribution.TOP_LAYERS, 50, [architecture])
    model_store.add_output_sizes_to_rich_snapshots('../model_resource_info/outputs/layer_output_infos.json')

    mm_snapshot = MultiModelSnapshot()
    for snapshot in model_snapshots:
        # get snapshot from model store to have access to the rich model snapshot
        mm_snapshot.add_snapshot(model_store.get_snapshot(snapshot.id))

    execution_tree = execution_tree_from_mm_snapshot(mm_snapshot, 0)
    out = execution_tree.find_cheap_children([execution_tree.root])
    print("current node:", out[0])
    print("sum children:", out[1])
    print("children:", out[2])
