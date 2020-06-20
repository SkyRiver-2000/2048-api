from game2048.game import Game
from game2048.displays import Display

def single_run(size, score_to_win, AgentClass, **kwargs):
    game = Game(size, score_to_win)
    agent = AgentClass(game, display = Display(), **kwargs)
    agent.play(verbose = True, max_iter = 5e3)
    return game.score

if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 10

    '''====================
    My own agent here.'''
    from MyAgent import MyOwnAgent as TestAgent
    '''===================='''
    
    # game = Game(GAME_SIZE, SCORE_TO_WIN)
    # print(type(game.board))

    scores = []
    for _ in range(N_TESTS):
        score = single_run(GAME_SIZE, SCORE_TO_WIN,
                           AgentClass = TestAgent, load_path = "./mdl_final.pkl")
        scores.append(score)

    print("Average scores: @%s times" % N_TESTS, sum(scores) / len(scores))
