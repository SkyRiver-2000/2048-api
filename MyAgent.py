import torch
import numpy as np
import random as rd
from cnn import Conv_Net, Conv_Net_com
from game2048.game import Game
from game2048.displays import Display
from game2048.agents import Agent, ExpectiMaxAgent

# A new single_run function for testing model in train_model.py
def single_run(size, score_to_win, AgentClass, model):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display = Display())
    agent.import_model(model)
    agent.play(max_iter = 5e3, verbose = False)
    return game.score

# MyOwnAgent class, with a model attribute for direction decision
class MyOwnAgent(Agent):
    def __init__(self, game, display = None, load_path = None):
        self.game = game
        self.display = display
        
        # Use trained CNN as model
        self.model = Conv_Net_com().cuda()
        if load_path is not None:
            self.model.load_state_dict(torch.load(load_path))
        self.model.eval()

    # The play function is exactly the same as in base class
    def play(self, max_iter = np.inf, verbose = False):
        n_iter = 0
        while (n_iter < max_iter) and (not self.game.end):
            direction = self.step()
            self.game.move(direction)
            n_iter += 1
            if verbose:
                print("Iter: {}".format(n_iter))
                print("======Direction: {}======".format(
                    ["left", "down", "right", "up"][direction]))
                if self.display is not None:
                    self.display.display(self.game)

    # Choose the best move on the board
    def step(self):
        direction = Get_direction(self.game, self.model)
        return direction
    
    # Another method for model loading
    def import_model(self, mdl):
        self.model = mdl

# Make decision for next move
def Get_direction(game, model):
    temp = torch.zeros(size = (1, 12, 4, 4))
    for i in range(4):
        for j in range(4):
            if game.board[i, j] != 0:
                temp[0, int(np.log2(game.board[i, j])), i, j] = 1
            else:
                temp[0, 0, i, j] = 1
    
    _, direction = torch.max(model(temp.cuda()), 1)
    return int(direction.cpu().numpy())