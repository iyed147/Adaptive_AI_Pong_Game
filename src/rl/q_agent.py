import random
import pickle
import os


class QAgent:

    def __init__(
        self,
        actions,
        alpha=0.2,
        gamma=0.95,
        epsilon=1.0
    ):

        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = {}

        # Chemin ABSOLU du fichier
        base_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.save_path = os.path.join(
            base_dir,
            "qtable.pkl"
        )

    def get_state_key(self, state):

        return tuple(state)

    def choose_action(self, state):

        state_key = self.get_state_key(state)

        if (
            random.random() < self.epsilon
            or state_key not in self.q_table
        ):

            return random.choice(
                self.actions
            )

        max_q = max(
            self.q_table[state_key].values()
        )

        best_actions = [

            action

            for action, value
            in self.q_table[state_key].items()

            if value == max_q
        ]

        return random.choice(
            best_actions
        )

    def update(
        self,
        state,
        action,
        reward,
        next_state
    ):

        s = self.get_state_key(state)
        s2 = self.get_state_key(next_state)

        if s not in self.q_table:

            self.q_table[s] = {

                action: 0
                for action in self.actions
            }

        if s2 not in self.q_table:

            self.q_table[s2] = {

                action: 0
                for action in self.actions
            }

        old_q = self.q_table[s][action]

        next_max = max(
            self.q_table[s2].values()
        )

        new_q = old_q + self.alpha * (

            reward
            + self.gamma * next_max
            - old_q
        )

        self.q_table[s][action] = new_q

    # ---------------- SAVE ----------------

    def save(self):

        with open(
            self.save_path,
            "wb"
        ) as file:

            pickle.dump(
                self.q_table,
                file
            )

        print("\nQ-table sauvegardée :")
        print(self.save_path)

    # ---------------- LOAD ----------------

    def load(self):

        if os.path.exists(
            self.save_path
        ):

            with open(
                self.save_path,
                "rb"
            ) as file:

                self.q_table = pickle.load(
                    file
                )

            print("\nQ-table chargée :")
            print(self.save_path)

        else:

            print("\nAucune Q-table trouvée.")
            print("Nouvel apprentissage.")