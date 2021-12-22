from enum import IntEnum
from enum import Enum
import numpy as np
import random
import operator
from functools import total_ordering
import math

NUM_EPISODES = 1000
MAX_EPISODE_LENGTH = 500


class ExplorationMode(Enum):
    E_GREEDY = 0
    SOFTMAX = 1
    UCB_1 = 2


DEFAULT_DISCOUNT = 0.9
EPSILON = 0.05
LEARNINGRATE = 0.1
EXPLORATION_MODE = ExplorationMode.UCB_1


class QLearner:
    """
    Q-learning agent
    """
    def __init__(self, num_states, num_actions, nrow, ncol, discount=DEFAULT_DISCOUNT, learning_rate=LEARNINGRATE,
                 exploration_mode=EXPLORATION_MODE):
        self.name = "agent1"
        self.num_states = num_states
        self.num_actions = num_actions
        self.discount = discount
        self.learning_rate = learning_rate
        self.nrow = nrow
        self.ncol = ncol
        # print('size', ncol, nrow)
        self.size = nrow*ncol
        self.q_table = np.zeros((self.num_states, self.num_actions))
        self.exploration_mode = exploration_mode
        self.exploration_bonus = np.ones((self.num_states, self.num_actions))

        #for i in range(0, self.size):
        #    if i / ncol != 0:
        #        self.q_table[i, 3] = 0.5
        #    if i / ncol != nrow - 1:
        #        self.q_table[i, 1] = 0.5
        #    if i % ncol != 0:
        #        self.q_table[i, 0] = 0.5
        #    if i % ncol != ncol - 1:
        #        self.q_table[i, 2] = 0.5

    def x_coord(self, state):
        return int(state % self.ncol)

    def y_coord(self, state):
        return int(state / self.ncol)

    def action_space(self, state):
        x = self.x_coord(state)
        y = self.y_coord(state)
        # print("coord", x, y)
        can_go_left = x != 0
        can_go_up = y != 0
        can_go_right = x < self.ncol - 1
        can_go_down = y < self.nrow - 1

        actions = []
        if can_go_left:
            actions.append(Action.LEFT)
        if can_go_right:
            actions.append(Action.RIGHT)
        if can_go_up:
            actions.append(Action.UP)
        if can_go_down:
            actions.append(Action.DOWN)

        return actions

    def reset_episode(self):
        """
        Here you can update some of the statistics that could be helpful to maintain
        """
        pass

    def process_experience(self, state, action, next_state, reward, done): 
        """
        Update the Q-value based on the state, action, next state and reward.
        """
        # print(
        #     state, action, next_state, reward, done
        # )
        alpha = self.learning_rate
        q_old = self.q_table[state, action]
        r = reward
        gamma = self.discount

        #Register that the agent has been here another time
        self.exploration_bonus[state, action] += 1.0

        if done:
            self.q_table[state, action] = (1 - alpha) * q_old + alpha * r
        else:
            self.q_table[state, action] = (1 - alpha) * q_old + \
                                          alpha * (r + gamma * np.argmax(self.q_table[next_state, :]))

    def select_action(self, state):
        """
        Returns an action, selected based on the current state
        """
        # print("State: ", state)
        action_space = self.action_space(state)
        if self.exploration_mode == ExplorationMode.E_GREEDY:
            if np.random.random() > EPSILON:
                possible_actions = [Option(action, self.q_table[state, action]) for action in action_space]

                random.shuffle(possible_actions)
                # print(list(map(lambda x: str(x), possible_actions)))
                return max(possible_actions, key=operator.attrgetter("q")).action
                # random.shuffle(best_actions)
                # print('best actions', best_actions)
                # return np.random.choice(best_actions)
            return np.random.choice(action_space)
        elif self.exploration_mode == ExplorationMode.SOFTMAX:
            temperature = 100.0
            t = temperature
            p = np.array([self.q_table[(state, action)] / t for action in action_space])
            prob_actions = np.exp(p) / np.sum(np.exp(p))
            cumulative_probability = 0.0
            choice = random.uniform(0, 1)
            for a, pr in enumerate(prob_actions):
                cumulative_probability += pr
                if cumulative_probability > choice:
                    return a
        else:
            possible_actions = [Option(action, self.q_table[state, action] + self.bonus(state, action))
                                for action in action_space]
            random.shuffle(possible_actions)
            return max(possible_actions, key=operator.attrgetter("q")).action
        # choose action according to
        # the probability distribution
        # action = np.random.choice(np.arange(
        #     len(action_probabilities)),
        #     p=action_probabilities)
        #
        # return action

    def report(self):
        """
        Function to print useful information, printed during the main loop
        """
        print("---")

    def bonus(self, state, action):
        c = 0.3
        exploration_sum = np.sum(self.exploration_bonus[state, :])
        in_root: float = 2 * (math.log(exploration_sum) / self.exploration_bonus[state, action])
        return 100.0 * c * math.sqrt(in_root)


class Action(IntEnum):
    DOWN = 1
    RIGHT = 2
    LEFT = 0
    UP = 3


@total_ordering
class Option:
    def __init__(self, action, q):
        self.action = action
        self.q = q

    def _is_valid_operand(self, other):
        return hasattr(other, "q")

    def __eq__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.q == other.q

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.q <= other.q

    def __str__(self):
        return "(" + str(self.action) + ", " + str(self.q) + ")"
