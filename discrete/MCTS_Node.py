import numpy as np
from copy import deepcopy
class MCTS_Node:
    def __init__(self, state, player, num_splits, target_rows, target_cols, parent=None, parent_action = None):
        self.state = state # two boolean arrays lengths (m , n) for m rows and n columns
        self.parent = parent
        self.parent_actions = parent_action
        self.children = [] #list of child nodes
        self.number_of_visits = 0 #number of times explored
        self.result_history = [] #list of r^2 values
        self.player = player #0 for row and 1 for column
        self.num_splits = num_splits
        self.target_rows = target_rows
        self.target_cols = target_cols
        self.untried_actions = self.get_legal_actions()
        
    def average_score(self):
        return sum(self.result_history) / len(self.result_history)
    
    def is_terminal_node(self):
        rows_remaining, cols_remaining = len(np.where(self.state[0])[0]), len(np.where(self.state[1])[0])
        return (rows_remaining <= self.target_rows) and (cols_remaining <= self.target_cols)
    
    def get_legal_actions(self):
        #if column, choose any remaining column
        if self.player == 1:
            action_array = self.state[self.player]
            return list(np.where(action_array)[0])
        
        #if row, generate a number of splits equal to num_splits
        else:
            legal_rows = np.where(self.state[self.player])[0]
            rows_to_select = (len(self.state[0]) - self.target_rows)/ (len(self.state[1]) - self.target_cols)
            actions = []
            offset = rows_to_select % 1
            for _ in range(self.num_splits):
                size = int(rows_to_select) + np.random.choice([0, 1], p=[1-offset, offset])
                actions.append(np.random.choice(legal_rows, size=size))
            return list(actions)
            
    
    def expand(self):
        action = self.untried_actions.pop()
        next_state = deepcopy(self.state)
        next_state[self.player][action] = False
        child = MCTS_Node(
            next_state, 
            self.player ^ 1, 
            self.num_splits, 
            self.target_rows, 
            self.target_cols, 
            parent=self, 
            parent_action=action
        )
        self.children.append(child)
        return child
        
    def backpropogate(self, result):
        self.number_of_visits += 1
        self.result_history.append(result)
        if self.parent:
            self.parent.backpropogate