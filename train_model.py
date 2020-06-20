from game2048.game import Game
# from game2048.displays import Display
from game2048.agents import Agent, ExpectiMaxAgent

from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import random as rd
from copy import deepcopy

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as opt

from torch.utils.data import DataLoader

from data_process import Dataset_2048

from MyAgent import *
from cnn import Conv_Net, Conv_Net_v2, Conv_Net_com

# After each epoch, this function is called to test the performance of my CNN
# My own CNN makes every decision, and Expectimax gives a reference
# Return the decision accuracy in n_tests games
def testAgent(n_tests, game_size, score_to_win, model, max_iter = 1000):
    acc, total = 0, 0
    for i in trange(n_tests):
        game, n_iter = Game(game_size, score_to_win), 0
        target = ExpectiMaxAgent(game)
        while n_iter < max_iter and not target.game.end:
            dir_ = Greedy_Action(game, model)
            target_dir = target.step()
            n_iter += 1
            total += 1
            if dir_ == target_dir:
                acc += 1
            target.game.move(dir_)
    return acc / total

if __name__ == '__main__':
    GAME_SIZE = 4
    SCORE_TO_WIN = 2048
    N_TESTS = 30
    TRAIN_TESTS = 20
    NUM_EPOCHS = 100
    MAX_ITER = 2000
    BATCH_SIZE = 1024
    INIT_LR = 1e-5      # Use extremely small LR to avoid the problem of forgetting
    
    # NUM_EPOCHS = 0 means only testing without training
    if NUM_EPOCHS != 0:
        print('Loading data...', '\n')
        # Use the self-defined dataset to store game records
        # Use DataLoader provided in torchvision to load batches
        data = Dataset_2048(file_path = './Data_Compressed.csv')
        train_data_loader = DataLoader(dataset = data, batch_size = BATCH_SIZE, shuffle = False)
    
    # Loading pre-trained model if needed
    # Move the model and computation afterwards to GPU
    print('Loading model...', '\n')
    model = Conv_Net_com().cuda()
    # model.load_state_dict(torch.load("./mdl_final.pkl"))
    model.train()
    
    # Define optimizer and exponential LR decay
    # Use multi-class cross entropy loss
    optim = opt.Adam(model.parameters(), lr = INIT_LR)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optim, gamma = 0.96)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    # As we need print information each epoch,
    # we do not use tqdm on outside loop
    print('Training model...', '\n')
    for ep in range(NUM_EPOCHS):
        optim.zero_grad()
        train_correct, train_total = 0, len(data)
        # For inner loop, we use tqdm to monitor the training status
        for X, y in tqdm(train_data_loader):
            X_train, y_train = Variable(X).type(torch.FloatTensor).cuda(), Variable(y).type(torch.LongTensor).cuda()
            outputs = model(X_train)
            _, y_pred = torch.max(outputs, 1)
            train_correct += torch.sum(y_pred.data == y_train.data).item()
            loss = criterion(outputs, y_train)
            loss.backward()
            optim.step()
        if ep < 30:
            ExpLR.step()    # LR will decrease to about 3e-6 after 30 epoches
            
        # Print necessary information of current model performance on training set
        print("Epoch: {:d}, Loss: {:.3f}".format(ep + 1, loss.data.cpu().numpy()))
        print("Training Accuracy: %.2f%%" % (train_correct / train_total * 100))
        
        # Test model on new games to check if model can play well itself
        score_list, test_model = [ ], deepcopy(model).cuda()
        for idx in range(TRAIN_TESTS):
            score_list.append(single_run(GAME_SIZE, SCORE_TO_WIN, MyOwnAgent, test_model))
        print("Average scores: @%s times" % TRAIN_TESTS, np.mean(score_list))
    
    # Training completed, set model to evalutaion status
    print('Testing model...', '\n')
    model.eval()
    
    # This is to find the gap of performance between training set and testing set
    # If the decision accuracy of my own model here is close to the training accuracy,
    # I know that the problem of overfitting is generally avoid
    if NUM_EPOCHS != 0:
        print(testAgent(N_TESTS, GAME_SIZE, SCORE_TO_WIN, model))
    
    # Use the model in evaluation status to test the real performance
    # It is strange that the model obtained by deepcopy is obviously weaker than the model here
    score_list = [ ]
    for idx in trange(N_TESTS):
        score_list.append(single_run(GAME_SIZE, SCORE_TO_WIN, MyOwnAgent, model))
    print("Scores:", score_list)
    print("Average scores: @%s times" % N_TESTS, np.mean(score_list))
    
    # Let me decide if I want the newly-trained model by myself
    flag = input("Save Model? [Y/n] ")
    if flag == 'Y' or flag == 'y':
        torch.save(model.state_dict(), "./mdl_final_v2.pkl")
        print("Save model successfully!")
