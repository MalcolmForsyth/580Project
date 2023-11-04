'''
A class for Monte Carlo Tree Search (MCTS) in the feature selection game.
The state of the game is represented by two numpy arrays of booleans for the features and columns currently selected. ([1, 0, 1, 0, 1], [1, 1, 0]) means the 2nd and 4th row have been excluded and the last column has been excluded.

'''
import numpy as np
import random
from MCTS_Node import MCTS_Node
from copy import deepcopy
import pickle
#constants
START_PLAYER = 1
NUM_SPLITS = 100
TARGET_ROWS = 0.3
TARGET_COLS = 0.1
EXPLORE_EXPLOIT_PARAM = np.sqrt(2)
class MCTS:
    def __init__(self, data):
        self.X, self.y = data[0], data[1]
        self.num_rows, self.num_cols = self.X.shape
        self.root = MCTS_Node(
            state=[np.ones(self.num_rows, dtype=bool), np.ones(self.num_cols, dtype=bool)],
            player=1,
            num_splits = NUM_SPLITS,
            target_rows = int(TARGET_ROWS * self.num_rows),
            target_cols = int(TARGET_COLS * self.num_cols)
        )
        
        
    def rollout(self, node):
        curr = node
        while not curr.is_terminal_node():
            actions = curr.get_legal_actions()
            action = random.choice(actions)
            next_state = deepcopy(curr.state)
            next_state[curr.player][action] = False 
            next_node = MCTS_Node(
                next_state, 
                curr.player ^ 1, 
                curr.num_splits, 
                curr.target_rows, 
                curr.target_cols, 
                parent=curr, 
                parent_action = action
            )
            curr = next_node
        data = self.X[curr.state[0]][:, curr.state[1]]
        data_with_bias = np.hstack([np.ones((data.shape[0], 1)), data])
        targets = self.y[curr.state[0]]
        coefficients, residuals, rank, s = np.linalg.lstsq(data_with_bias, targets, rcond=None)
        predictions = data_with_bias.dot(coefficients)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2_score = 1 - ss_res / ss_tot
        print(r2_score)
        return r2_score
    
    
    #TODO: fix and demean the winrate component
    def select_child(self, node):
        n = node.number_of_visits
        payoff_visits = [(np.mean(c.result_history), c.number_of_visits) if c.number_of_visits != 0 else (np.inf, 0) for c in node.children]
        choices_weights = [(q / num) + EXPLORE_EXPLOIT_PARAM * np.sqrt((2 * np.log(n) / num)) if num != 0 else np.inf for q, num in payoff_visits]
        return node.children[np.argmax(choices_weights)]
    
    def expand_tree(self):
        curr = self.root
        while not curr.is_terminal_node():
            if not (len(curr.untried_actions) == 0):
                return curr.expand()
            else:
                curr = self.select_child(curr)
        return curr
    
    def build_tree(self, iterations=100):
        for _ in range(iterations):
            v = self.expand_tree()
            payoff = self.rollout(v)
            v.backpropogate(payoff)
            if _ % 10000 == 0:
                pickle.dump(self, open(f'MCTS_it{_}.pkl', 'wb'))