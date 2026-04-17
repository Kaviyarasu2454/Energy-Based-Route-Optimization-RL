import heapq
import datetime
import math


class Dijkstra:

    def __init__(self, env, start_node, end_node):
        self.env = env
        self.env.set_start_end(start_node, end_node)

    def search(self):

        cost = {node: float('inf') for node in self.env.nodes}
        cost[self.env.start_node] = 0
        predecessor = {}
        pq = [(0, self.env.start_node)]

        while pq:
            current_cost, node = heapq.heappop(pq)

            if node == self.env.end_node:
                break

            for edge in self.env.decode_node_to_edges(node, 'outgoing'):
                next_node = self.env.decode_edge_to_node(edge)
                new_cost = current_cost + self.env.get_edge_distance(edge)

                if new_cost < cost[next_node]:
                    cost[next_node] = new_cost
                    predecessor[next_node] = node
                    heapq.heappush(pq, (new_cost, next_node))

        node_path = []
        edge_path = []
        current = self.env.end_node

        if current not in predecessor and current != self.env.start_node:
            return [], [], 0, 0, 0

        while current != self.env.start_node:

            if current not in predecessor:
                return [], [], 0, 0, 0

            node_path.append(current)
            prev = predecessor[current]

            edges = set(self.env.decode_node_to_edges(prev, 'outgoing')) & \
                    set(self.env.decode_node_to_edges(current, 'incoming'))

            if not edges:
                return [], [], 0, 0, 0

            edge_path.append(next(iter(edges)))
            current = prev

        node_path.append(self.env.start_node)
        node_path.reverse()
        edge_path.reverse()

        dist = self.env.get_edge_distance(edge_path)
        time = self.env.get_edge_time(edge_path)
        energy = self.env.get_edge_energy(edge_path)

        return node_path, edge_path, dist, time, energy


class AStar:

    def __init__(self, env, start_node, end_node):
        self.env = env
        self.env.set_start_end(start_node, end_node)

    def heuristic(self, node):
        x1, y1 = self.env.net.getNode(node).getCoord()
        x2, y2 = self.env.net.getNode(self.env.end_node).getCoord()
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def search(self):

        open_set = [(0, self.env.start_node)]
        g_cost = {node: float('inf') for node in self.env.nodes}
        g_cost[self.env.start_node] = 0
        predecessor = {}

        while open_set:
            _, node = heapq.heappop(open_set)

            if node == self.env.end_node:
                break

            for edge in self.env.decode_node_to_edges(node, 'outgoing'):
                next_node = self.env.decode_edge_to_node(edge)

                tentative = g_cost[node] + self.env.get_edge_distance(edge)

                if tentative < g_cost[next_node]:
                    g_cost[next_node] = tentative
                    predecessor[next_node] = node

                    f_cost = tentative + self.heuristic(next_node)
                    heapq.heappush(open_set, (f_cost, next_node))

        node_path = []
        edge_path = []
        current = self.env.end_node

        if current not in predecessor and current != self.env.start_node:
            return [], [], 0, 0, 0

        while current != self.env.start_node:

            if current not in predecessor:
                return [], [], 0, 0, 0

            node_path.append(current)
            prev = predecessor[current]

            edges = set(self.env.decode_node_to_edges(prev, 'outgoing')) & \
                    set(self.env.decode_node_to_edges(current, 'incoming'))

            if not edges:
                return [], [], 0, 0, 0

            edge_path.append(next(iter(edges)))
            current = prev

        node_path.append(self.env.start_node)
        node_path.reverse()
        edge_path.reverse()

        dist = self.env.get_edge_distance(edge_path)
        time = self.env.get_edge_time(edge_path)
        energy = self.env.get_edge_energy(edge_path)

        return node_path, edge_path, dist, time, energy