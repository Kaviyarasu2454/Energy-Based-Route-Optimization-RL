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
        ax.plot(xs, ys, linewidth=1.2, color="black", alpha=0.9, zorder=1)


def draw_route(ax, net, edge_ids, color, width):
    for edge_id in edge_ids:
        e = net.getEdge(edge_id)
        shape = e.getShape()
        xs = [p[0] for p in shape]
        ys = [p[1] for p in shape]
        # 🔥 Improved clarity (ONLY CHANGE)
        ax.plot(xs, ys, linewidth=width, color=color, zorder=5, alpha=0.9)


if __name__ == '__main__':

    sumo_configuration()
    random.seed(42)

    network_file = './network_files/kelambakkam.net.xml'

    net = sumolib.net.readNet(network_file)
    nodes = list(net.getNodes())
    edges = list(net.getEdges())

    env = environment.traffic_env(
        network_file=network_file,
        evaluation="energy",
        congestion_level="medium"
    )

    # 🔥 Ask Battery from User
    initial_battery = float(input("Enter Initial Battery (%) [Example: 100]: "))

    while True:

        fig, ax = plt.subplots(figsize=(14, 12))

        draw_network_gray(ax, edges)

        ax.set_title("Click START then DESTINATION")
        ax.axis("equal")
        ax.axis("off")

        print("\nClick START then DESTINATION on map...")

        clicks = plt.ginput(2, timeout=-1)

        if len(clicks) < 2:
            print("❌ Selection cancelled.")
            plt.close(fig)
            continue

        (sx, sy), (ex, ey) = clicks

        start_node = nearest_node(net, nodes, sx, sy)
        end_node = nearest_node(net, nodes, ex, ey)

        print("\nSelected Nodes:")
        print("Start:", start_node)
        print("End:", end_node)

        test_node_path, test_edge_path, _, _, _ = \
            dijkstra.Dijkstra(env, start_node, end_node).search()

        if not test_node_path:
            print("❌ No path found.")
            plt.close(fig)
            continue

        # START & END MARKERS
        sxc, syc = net.getNode(start_node).getCoord()
        exc, eyc = net.getNode(end_node).getCoord()

        ax.scatter([sxc], [syc], s=150, color="red", zorder=10)
        ax.text(sxc, syc, " START", fontsize=10, fontweight="bold", color="red")

        ax.scatter([exc], [eyc], s=150, color="green", zorder=10)
        ax.text(exc, eyc, " END", fontsize=10, fontweight="bold", color="green")

        break

    # --------------------------
    # Algorithms
    # --------------------------

    d_node, d_edge, d_dist, d_time, d_energy = \
        dijkstra.Dijkstra(env, start_node, end_node).search()

    a_node, a_edge, a_dist, a_time, a_energy = \
        dijkstra.AStar(env, start_node, end_node).search()

    rl_agent = agent.Q_Learning(env, start_node, end_node)
    rl_node, rl_edge, _, info = rl_agent.train(5000, 10)

    rl_dist = env.get_edge_distance(rl_edge)
    rl_time = env.get_edge_time(rl_edge)
    rl_energy = env.get_edge_energy(rl_edge)

    # 🔥 Battery Calculation
    final_battery = initial_battery - (rl_energy * 20)
    final_battery = max(0, final_battery)

    # --------------------------
    # OUTPUT
    # --------------------------

    print("\n=========== FINAL COMPARISON ===========")

    print(f"{'Algorithm':<10} | {'Distance':<10} | {'Time':<10} | {'Energy':<10}")
    print("-" * 50)

    print(f"{'RL':<10} | {round(rl_dist,2):<10} | {round(rl_time,2):<10} | {round(rl_energy,4):<10}")
    print(f"{'Dijkstra':<10} | {round(d_dist,2):<10} | {round(d_time,2):<10} | {round(d_energy,4):<10}")
    print(f"{'A*':<10} | {round(a_dist,2):<10} | {round(a_time,2):<10} | {round(a_energy,4):<10}")

    savings = ((d_energy - rl_energy) / d_energy) * 100
    savings = max(0, savings)

    print("\nEnergy Saving (RL vs Dijkstra):", round(savings, 2), "%")

    print("\nInitial Battery:", initial_battery, "%")
    print("Final Battery after RL:", round(final_battery, 2), "%")

    # --------------------------
    # Visualization
    # --------------------------

    # 🔥 RL thicker for clarity
    draw_route(ax, net, rl_edge, color="blue", width=6)
    draw_route(ax, net, d_edge, color="green", width=3)

    # 🔥 Legend (ONLY ADDITION)
    import matplotlib.lines as mlines
    rl_line = mlines.Line2D([], [], color='blue', linewidth=3, label='RL Route')
    dj_line = mlines.Line2D([], [], color='green', linewidth=3, label='Dijkstra Route')
    plt.legend(handles=[rl_line, dj_line])

    plt.title("Energy-Based Route Optimization (RL vs Dijkstra)")
    plt.show()