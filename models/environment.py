import sys
import sumolib
import math
import networkx as nx
import matplotlib.pyplot as plt
import random


class traffic_env:

    def __init__(self, network_file, congested=[], traffic_light=[],
                 evaluation="energy", congestion_level="low", travel_speed=80):

        self.network_file = network_file
        self.net = sumolib.net.readNet(network_file)

        self.nodes = [node.getID() for node in self.net.getNodes()]
        self.edges = [edge.getID() for edge in self.net.getEdges()]

        self.action_space = [0, 1, 2, 3]
        self.state_space = self.nodes
        self.edge_label = self.decode_edges_to_label()

        # --------------------------------------------------
        # Congestion Setup
        # --------------------------------------------------

        if congested:
            self.congested_edges = [item[0] for item in congested]
            self.congestion_duration = [item[1] for item in congested]
        else:
            traffic_level = 0.05 if congestion_level == "low" else \
                            0.1 if congestion_level == "medium" else 0.2

            self.congested_edges = random.sample(
                self.edges,
                round(len(self.edges) * traffic_level)
            )

            self.congestion_duration = [
                random.randint(15, 40)   # stronger congestion delay
                for _ in range(len(self.congested_edges))
            ]

        self.tl_nodes = [item[0] for item in traffic_light]
        self.tl_duration = [item[1] for item in traffic_light]

        self.evaluation = evaluation.lower()
        self.travel_speed = travel_speed

    # --------------------------------------------------
    # Node / Edge Utilities
    # --------------------------------------------------

    def set_start_end(self, start_node, end_node):
        self.start_node = start_node
        self.end_node = end_node

    def decode_node_to_edges(self, node, direction=None):
        net_node = self.net.getNode(node)

        if direction == 'incoming':
            return [e.getID() for e in net_node.getIncoming()]
        elif direction == 'outgoing':
            return [e.getID() for e in net_node.getOutgoing()]
        else:
            return [e.getID() for e in net_node.getIncoming() + net_node.getOutgoing()]

    def decode_edges_to_label(self):
        edge_labelled = {edge: None for edge in self.edges}

        for node in self.nodes:
            outgoing = self.decode_node_to_edges(node, 'outgoing')
            if outgoing:
                for i, edge in enumerate(outgoing):
                    edge_labelled[edge] = i
        return edge_labelled

    def decode_edges_to_actions(self, edges):
        return [self.edge_label[e] for e in edges if self.edge_label[e] is not None]

    def decode_edges_action_to_edge(self, edges, action):
        for e in edges:
            if self.edge_label[e] == action:
                return e
        return None

    def decode_edge_to_node(self, edge, direction='end'):
        e = self.net.getEdge(edge)
        return e.getFromNode().getID() if direction == 'start' else e.getToNode().getID()

    # --------------------------------------------------
    # Cost Functions
    # --------------------------------------------------

    def get_edge_distance(self, travel_edges):
        if isinstance(travel_edges, str):
            travel_edges = [travel_edges]

        return sum(self.net.getEdge(e).getLength() for e in travel_edges)

    def get_edge_time(self, travel_edges):
        if isinstance(travel_edges, str):
            travel_edges = [travel_edges]

        total_time = ((self.get_edge_distance(travel_edges) / 1000)
                      / self.travel_speed) * 60

        for edge in travel_edges:
            if edge in self.congested_edges:
                delay = self.congestion_duration[
                    self.congested_edges.index(edge)
                ]
                total_time += delay

        return total_time

    # 🔥 MODIFIED ENERGY MODEL
    def get_edge_energy(self, travel_edges):

        if isinstance(travel_edges, str):
            travel_edges = [travel_edges]

        total_energy = 0

        base_consumption = 0.2  # kWh per km

        for edge in travel_edges:

            # Base energy from distance
            distance_km = self.net.getEdge(edge).getLength() / 1000
            energy = distance_km * base_consumption

            # 🔥 Strong congestion penalty
            if edge in self.congested_edges:
                delay = self.congestion_duration[
                    self.congested_edges.index(edge)
                ]

                # Higher multiplier so congestion hurts more
                energy += delay * 0.02

            total_energy += energy

        return total_energy

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    def visualize_plot(self, travel_edges):

        nodes_dict = {
            node: self.net.getNode(node).getCoord()
            for node in self.nodes
        }

        edges_dict = {
            e: (
                self.net.getEdge(e).getFromNode().getID(),
                self.net.getEdge(e).getToNode().getID()
            )
            for e in self.edges
        }

        G = nx.DiGraph()

        for edge in edges_dict:
            G.add_edge(edges_dict[edge][0],
                       edges_dict[edge][1])

        plt.figure(figsize=(12, 10))

        nx.draw_networkx_edges(G, nodes_dict,
                               edge_color='lightgray',
                               width=0.5)

        route_edges = [edges_dict[e] for e in travel_edges]

        nx.draw_networkx_edges(G, nodes_dict,
                               edgelist=route_edges,
                               edge_color='green',
                               width=3)

        plt.title("RL Energy Efficient Route")
        plt.axis("off")
        plt.show()