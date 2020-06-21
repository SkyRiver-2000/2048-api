import pandas as pd
import numpy as np
from tqdm import trange

from game2048.game import Game
from game2048.displays import Display
from game2048.agents import Agent, ExpectiMaxAgent

# Normalization with logarithm before storing the flattened board
def board_log2(board):
    for i in range(16):
        if board[i] != 0:
            board[i] = np.log2(board[i])
    return board

# Set the needed number of games
# Limit the max number of steps given by Expectimax each game
NUM_GAMES = 10000
MAX_ITER = 1e3

# This loop takes really a long time
# Use trange to monitor the execution status
for i in trange(NUM_GAMES):
    game = Game(size = 4, score_to_win = 2048)
    target = ExpectiMaxAgent(game)
    n_iter = 0
    data = np.zeros((0, 17), dtype = float)
    # To avoid frequent I/O operation,
    # data is written into .csv after an entire game rather than one step
    while n_iter < MAX_ITER and not target.game.end:
        dir_ = target.step()
        x = board_log2(np.reshape(target.game.board, newshape = (16, )))
        item = np.hstack((x, dir_))
        data = np.vstack([data, item])
        target.game.move(dir_)
        n_iter += 1
    df = pd.DataFrame(data, columns = None, index = None)
    df.to_csv('./Data_Compressed.csv', mode = 'a', index = False, header = False)
# print(df)