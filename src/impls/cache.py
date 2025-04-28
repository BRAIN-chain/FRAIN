import numpy as np
import random


class ItemCache:
    def __init__(self, min_counter=0, max_counter=50):
        self.cache = []
        self.min_counter = min_counter
        self.max_counter = max_counter
        self.next_id = 0  # For item identification
        self.staleness = {}

    def add_item_with_random_counter(self, item):
        counter = random.randint(self.min_counter, self.max_counter)
        id_ = self.next_id
        self.cache.append((id_, item, counter))
        self.staleness[id_] = counter
        self.next_id += 1
        return id_

    def add_item_with_specific_counter(self, item, counter):
        id_ = self.next_id
        self.cache.append((id_, item, counter))
        self.staleness[id_] = counter
        self.next_id += 1
        return id_

    def update_counters(self):
        removed_items = [item for _, item,
                         counter in self.cache if counter-1 <= 0]
        self.cache = [(id_, item, counter-1)
                      for id_, item, counter in self.cache if counter-1 > 0]
        return removed_items

    def update_counters_with_ids(self):
        removed_items = [item for _, item,
                         counter in self.cache if counter-1 <= 0]
        removed_item_ids = [id_ for id_, _,
                            counter in self.cache if counter-1 <= 0]
        self.cache = [(id_, item, counter-1)
                      for id_, item, counter in self.cache if counter-1 > 0]
        return removed_items, removed_item_ids


# Example usage
if __name__ == "__main__":
    # import seaborn as sns
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import networkx as nx

    iterations = 10
    max_staleness = 4
    clients_per_round = 2

    # Create a directed graph
    G = nx.DiGraph()

    # Define colors for gids nodes and others
    gid_colors = 'black'  # Color for nodes in gids

    cmap = plt.get_cmap('Blues')
    default_colors = [cmap(i) for i in np.linspace(0.3, 1.0, max_staleness+1)]
    # default_colors = sns.color_palette("Blues", max_staleness+1)

    cache = ItemCache(min_counter=1, max_counter=max_staleness+1)
    gid_ = 0
    gids = [f'G{gid_}']
    G.add_node(gid_)

    for i in range(iterations):
        prevGid = f'G{gid_}'

        for _ in range(clients_per_round):
            # specific_item = random.randint(1, max_staleness+1)
            choices = np.arange(1, max_staleness + 2)
            weights = np.arange(max_staleness + 1, 0, -1)
            specific_item = np.random.choice(
                choices, p=weights / weights.sum())

            id_ = cache.add_item_with_specific_counter(
                specific_item, specific_item)
            G.add_edge(f'G{gid_}', id_)

        removed_items, removed_item_ids = cache.update_counters_with_ids()
        print(f"Iteration {i+1}: Removed {len(removed_items)} items")

        gid_ += 1
        gids.append(f'G{gid_}')
        G.add_edge(prevGid, f'G{gid_}')

        if len(removed_items) != 0:
            G.add_edges_from(
                zip(removed_item_ids, [f'G{gid_}' for _ in range(len(removed_item_ids))]))
        # else:
            # G.add_edge(prevGid, f'G{gid_}')

    # # topologically sorted
    # for layer, nodes in enumerate(nx.topological_generations(G)):
    #     # `multipartite_layout` expects the layer as a node attribute, so add the
    #     # numeric layer value as a node attribute
    #     for node in nodes:
    #         G.nodes[node]["layer"] = layer
    # # Compute the multipartite_layout using the "layer" node attribute
    # pos = nx.multipartite_layout(G, subset_key="layer")

    # Use topological sort to determine the order and layering of nodes
    topo_sort = list(nx.topological_sort(G))
    layer_map = {}  # Map each node to its layer based on topological sorting

    # Initial layer assignment based on topological sorting
    for node in topo_sort:
        if node in gids:
            # Directly map gids to their index in gids list for central alignment
            layer_map[node] = gids.index(node)
        else:
            # Assign other nodes to a layer based on the highest layer of their predecessors
            # Assign other nodes to a layer based on the highest layer of their predecessors + 1
            preds = list(G.predecessors(node))
            if preds:
                # layer_map[node] = max(layer_map[pred] for pred in preds)
                layer_map[node] = max(layer_map[pred] for pred in preds) + 1
            else:
                layer_map[node] = 0

    # Adjust positions for central alignment and to minimize edge overlap
    # Keep track of the number of nodes in each layer to spread them out evenly
    layer_counts = {}
    pos = {}
    for node, layer in layer_map.items():
        if layer not in layer_counts:
            layer_counts[layer] = 0
        pos[node] = (layer, layer_counts[layer])
        layer_counts[layer] += 1

    # Adjust gids to be centered
    max_count = max(layer_counts.values())
    for gid in gids:
        layer = layer_map[gid]
        # Center gids by adjusting their x position
        pos[gid] = (layer, max_count)

    # Create a color map for nodes
    color_map = [
        gid_colors if node in gids else default_colors[cache.staleness[node]-1] for node in G.nodes()]

    fig, ax = plt.subplots(figsize=(12, 2))

    # nx.draw_networkx(G,
    #                  pos=pos,
    #                  ax=ax,
    #                  node_color=color_map,
    #                  connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=color_map)
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_color='white', font_size=8, font_weight="bold")

    # Separating edges into two groups: straight edges and arc edges
    straight_edges = [(u, v) for u, v in G.edges() if u in gids and v in gids]
    # Assuming gids are where you want dashed lines

    arc_edges = [(u, v) for u, v in G.edges() if not (u in gids and v in gids)]

    dashed_edges = [(u, v)
                    for u, v in G.edges() if (u in gids and v not in gids)]
    normal_edges = [(u, v)
                    for u, v in G.edges() if (u not in gids and v in gids)]

    # Draw straight edges with default connection style (straight lines)
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges,
                           arrows=True, arrowstyle='->', arrowsize=10, width=2, edge_color=gid_colors)

    # Draw arc edges with arc connection style
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=dashed_edges,
                           arrows=True, arrowstyle='->', arrowsize=10, connectionstyle='arc3,rad=0.1',
                           width=0.5, edge_color='gray', style='dashed')
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=normal_edges,
                           arrows=True, arrowstyle='->', arrowsize=10, connectionstyle='arc3,rad=0.1')

    # ax.set_title("Model DAG")
    plt.axis('off')  # Hide axes for better visualization

    # Create patches for the legend
    patches = [mpatches.Patch(
        color=default_colors[i], label=rf'${i}$') for i in range(len(default_colors))]
    # Add the legend to the plot
    plt.legend(handles=patches,
               bbox_to_anchor=(1.02, 0.5),
               loc='center right',
               title="Staleness")

    fig.tight_layout()
    plt.savefig("./save/DAG.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Find nodes with more than 2 incoming edges
    nodes_with_more_than_two_incoming = [
        node for node, degree in G.in_degree() if degree > 2]

    print(
        f"Nodes with more than 2 incoming edges: {nodes_with_more_than_two_incoming}")
