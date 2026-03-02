import os
import sys
import random

sys.path.append('models/')
import environment
import agent
import dijkstra

import matplotlib.pyplot as plt
import sumolib


def sumo_configuration():
    os.environ["SUMO_HOME"] = "C:/Program Files (x86)/Eclipse/Sumo/"
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)


def nearest_node(net, nodes, x, y):
    best = None
    best_dist = float("inf")
    for n in nodes:
        nx, ny = n.getCoord()
        d = (nx - x) ** 2 + (ny - y) ** 2
        if d < best_dist:
            best_dist = d
            best = n.getID()
    return best


def draw_network_gray(ax, edges):
    for edge in edges:
        shape = edge.getShape()
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        ax.plot(xs, ys, linewidth=0.5, color="lightgray", alpha=0.8, zorder=1)


def draw_route(ax, net, edge_ids, color, width, zorder=4):
    for edge_id in edge_ids:
        e = net.getEdge(edge_id)
        shape = e.getShape()
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        ax.plot(xs, ys, linewidth=width, color=color, zorder=zorder)


if __name__ == '__main__':

    sumo_configuration()
    random.seed(42)

    network_file = './network_files/kelambakkam.net.xml'

    # Load network
    net = sumolib.net.readNet(network_file)
    nodes = list(net.getNodes())
    edges = list(net.getEdges())

    # Create environment
    env = environment.traffic_env(
        network_file=network_file,
        evaluation="energy",
        congestion_level="medium"
    )

    # ==============================
    # STEP 1: Interactive Selection with Validation
    # ==============================

    while True:

        fig, ax = plt.subplots(figsize=(10, 8))
        draw_network_gray(ax, edges)

        ax.set_title("Click START then DESTINATION")
        ax.axis("equal")
        ax.axis("off")

        print("\nClick START then DESTINATION on map...")

        clicks = plt.ginput(2, timeout=-1)

        if len(clicks) < 2:
            print("❌ Selection cancelled. Please select two nodes.")
            plt.close(fig)
            continue

        (sx, sy), (ex, ey) = clicks

        start_node = nearest_node(net, nodes, sx, sy)
        end_node = nearest_node(net, nodes, ex, ey)

        print("\nSelected Nodes:")
        print("Start:", start_node)
        print("End:", end_node)

        # -------- VALIDATION --------

        if start_node == end_node:
            print("❌ Rejected: Start and Destination are the same node.")
            plt.close(fig)
            continue

        if not env.decode_node_to_edges(start_node, 'outgoing'):
            print("❌ Rejected: Start node has no outgoing roads (dead end).")
            plt.close(fig)
            continue

        if not env.decode_node_to_edges(end_node, 'incoming'):
            print("❌ Rejected: Destination node has no incoming roads.")
            plt.close(fig)
            continue

        test_node_path, test_edge_path, _, _, _ = \
            dijkstra.Dijkstra(env, start_node, end_node).search()

        if not test_node_path:
            print("❌ Rejected: No reachable path exists (one-way restriction or disconnected network).")
            plt.close(fig)
            continue

        print("✅ Valid selection. Running algorithms...\n")

        # Mark start/end
        sxc, syc = net.getNode(start_node).getCoord()
        exc, eyc = net.getNode(end_node).getCoord()

        ax.scatter([sxc], [syc], s=120, color="red", zorder=6)
        ax.scatter([exc], [eyc], s=120, color="red", zorder=6)
        ax.text(sxc, syc, " START", fontsize=10, fontweight="bold", color="red")
        ax.text(exc, eyc, " END", fontsize=10, fontweight="bold", color="red")

        break

    # ==============================
    # STEP 2: Run Dijkstra + A*
    # ==============================

    print("\nRunning Distance-Based Algorithms...\n")

    d_node, d_edge, d_dist, d_time, d_energy = \
        dijkstra.Dijkstra(env, start_node, end_node).search()

    a_node, a_edge, a_dist, a_time, a_energy = \
        dijkstra.AStar(env, start_node, end_node).search()

    # ==============================
    # STEP 3: Run RL
    # ==============================

    print("\nTraining Reinforcement Learning Agent...\n")

    rl_agent = agent.Q_Learning(env, start_node, end_node)
    rl_node, rl_edge, _, _ = rl_agent.train(5000, 10)

    rl_dist = env.get_edge_distance(rl_edge)
    rl_time = env.get_edge_time(rl_edge)
    rl_energy = env.get_edge_energy(rl_edge)

    # ==============================
    # STEP 4: Print Comparison
    # ==============================

    print("\n=========== ALGORITHM COMPARISON ===========")

    print("\nRL (Energy-Aware)")
    print("Distance:", round(rl_dist, 2), "m")
    print("Time:", round(rl_time, 2), "min")
    print("Energy:", round(rl_energy, 4), "KWh")

    print("\nDijkstra (Shortest Distance)")
    print("Distance:", round(d_dist, 2), "m")
    print("Time:", round(d_time, 2), "min")
    print("Energy:", round(d_energy, 4), "KWh")

    print("\nA* (Shortest Distance)")
    print("Distance:", round(a_dist, 2), "m")
    print("Time:", round(a_time, 2), "min")
    print("Energy:", round(a_energy, 4), "KWh")

    if d_energy > 0:
        savings = ((d_energy - rl_energy) / d_energy) * 100
        print("\nEnergy Savings vs Dijkstra:", round(savings, 2), "%")

    # ==============================
    # STEP 5: Draw RL Route
    # ==============================

    draw_route(ax, net, rl_edge, color="green", width=4, zorder=5)

    ax.set_title("Optimized Energy Route (RL)")
    plt.show()