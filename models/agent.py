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

    # -------------------------------------------------
    # Choose Action (Epsilon-Greedy)
    # -------------------------------------------------
    def choose_action(self, state):

        state_idx = self.env.state_space.index(state)

        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.env.action_space))

        return np.argmax(self.q_table[state_idx])

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
    def train(self, episodes, converge_threshold):

        start_time = datetime.datetime.now()

        best_energy = float('inf')
        best_path = []
        best_states = []

        for episode in range(episodes):

            state = self.env.start_node
            state_path = [state]
            edge_path = []

            max_steps = 200   # 🔥 Safety limit
            steps = 0

            visited_states = set()

            while state != self.env.end_node and steps < max_steps:

                action = self.choose_action(state)
                outgoing = self.env.decode_node_to_edges(state, 'outgoing')

                # -------------------------------------------------
                # Invalid Action
                # -------------------------------------------------
                if action not in self.env.decode_edges_to_actions(outgoing):
                    reward = -100
                    next_state = state
                    next_edge = None

                else:
                    next_edge = self.env.decode_edges_action_to_edge(outgoing, action)
                    next_state = self.env.decode_edge_to_node(next_edge)

                    # Energy-based penalty
                    energy_cost = self.env.get_edge_energy(next_edge)
                    reward = -energy_cost

                    # Loop penalty
                    if next_state in visited_states:
                        reward -= 5

                    # Completion reward
                    if next_state == self.env.end_node:
                        reward += 1000

                # -------------------------------------------------
                # Q-Update
                # -------------------------------------------------
                s_idx = self.env.state_space.index(state)
                ns_idx = self.env.state_space.index(next_state)

                self.q_table[s_idx][action] += self.lr * (
                    reward +
                    self.gamma * np.max(self.q_table[ns_idx]) -
                    self.q_table[s_idx][action]
                )

                # -------------------------------------------------
                # Update state
                # -------------------------------------------------
                if next_edge:
                    edge_path.append(next_edge)
                    state_path.append(next_state)

                visited_states.add(state)
                state = next_state
                steps += 1

            # -------------------------------------------------
            # Track Best Path
            # -------------------------------------------------
            if state == self.env.end_node:

                total_energy = self.env.get_edge_energy(edge_path)

                if total_energy < best_energy:
                    best_energy = total_energy
                    best_path = edge_path.copy()
                    best_states = state_path.copy()

            # Slowly reduce exploration
            self.epsilon = max(0.01, self.epsilon * 0.997)

        end_time = datetime.datetime.now()
        print(f"RL Training Time: {(end_time - start_time).total_seconds()} seconds")

        return best_states, best_path, episodes, {}