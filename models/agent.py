import numpy as np
import datetime


class Q_Learning:

    def __init__(self, env, start_node, end_node,
                 learning_rate=0.3,
                 discount_factor=0.9,
                 epsilon=0.3):

        self.env = env
        self.env.set_start_end(start_node, end_node)

        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

        self.q_table = np.zeros((len(env.state_space),
                                 len(env.action_space)))

    def choose_action(self, state):

        state_idx = self.env.state_space.index(state)

        outgoing = self.env.decode_node_to_edges(state, 'outgoing')
        valid_actions = self.env.decode_edges_to_actions(outgoing)

        if not valid_actions:
            return np.random.choice(len(self.env.action_space))

        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)

        q_values = self.q_table[state_idx]

        valid_q = {a: q_values[a] for a in valid_actions}
        return max(valid_q, key=valid_q.get)

    def train(self, episodes, converge_threshold):

        start_time = datetime.datetime.now()

        best_energy = float('inf')
        best_path = []
        best_states = []
        final_battery = 100

        for episode in range(episodes):

            state = self.env.start_node
            state_path = [state]
            edge_path = []

            battery = 100

            max_steps = 200
            steps = 0

            visited_states = set()

            while state != self.env.end_node and steps < max_steps:

                action = self.choose_action(state)
                outgoing = self.env.decode_node_to_edges(state, 'outgoing')

                if action not in self.env.decode_edges_to_actions(outgoing):
                    reward = -100
                    next_state = state
                    next_edge = None

                else:
                    next_edge = self.env.decode_edges_action_to_edge(outgoing, action)
                    next_state = self.env.decode_edge_to_node(next_edge)

                    energy_cost = self.env.get_edge_energy(next_edge)

                    battery -= energy_cost * 20
                    battery = max(0, battery)

                    # 🔥 MAIN FIX: ENERGY-FOCUSED REWARD
                    reward = -(energy_cost * 8)

                    if next_state in visited_states:
                        reward -= 20   # loop penalty

                    if next_state == self.env.end_node:
                        reward += 500   # stronger goal reward

                s_idx = self.env.state_space.index(state)
                ns_idx = self.env.state_space.index(next_state)

                self.q_table[s_idx][action] += self.lr * (
                    reward +
                    self.gamma * np.max(self.q_table[ns_idx]) -
                    self.q_table[s_idx][action]
                )

                if next_edge:
                    edge_path.append(next_edge)
                    state_path.append(next_state)

                visited_states.add(state)
                state = next_state
                steps += 1

            if state == self.env.end_node:

                total_energy = self.env.get_edge_energy(edge_path)

                if total_energy < best_energy:
                    best_energy = total_energy
                    best_path = edge_path.copy()
                    best_states = state_path.copy()
                    final_battery = battery

            if episode % 500 == 0:
                print(f"Episode {episode}, Best Energy: {best_energy:.4f}")

            # 🔥 FIX: slower epsilon decay (prevents early bad learning)
            self.epsilon = max(0.05, self.epsilon * 0.998)

        # 🔥 CLEAN PATH (remove loops)
        clean_path = []
        visited_edges = set()

        for edge in best_path:
            if edge not in visited_edges:
                clean_path.append(edge)
                visited_edges.add(edge)

        end_time = datetime.datetime.now()
        print(f"RL Training Time: {(end_time - start_time).total_seconds()} seconds")

        return best_states, clean_path, episodes, {"battery": final_battery}