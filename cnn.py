import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.autograd import Variable

# The number of units in the flattened convolution layer
HIDDEN_SIZE_1 = 2 * 4 * 128 + 9 * 128 + 4 * 128 + 1 * 128
HIDDEN_SIZE_2 = 2 * 12 * 128 + 2 * 9 * 128 + 2 * 8 * 128
HIDDEN_SIZE_3 = HIDDEN_SIZE_1 * 2 + HIDDEN_SIZE_2

# The CNN offered by TA
# This CNN seems to focus more on large-scale and global patterns
class Conv_Net(nn.Module):
    def __init__(self):
        super(Conv_Net, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 128, kernel_size = (4, 1), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 128, kernel_size = (1, 4), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 128, kernel_size = (2, 2), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 128, kernel_size = (3, 3), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_5 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 128, kernel_size = (4, 4), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.dropout_1 = nn.Dropout(p = 0.5)
        self.fc_1 = nn.Sequential(nn.Linear(in_features = HIDDEN_SIZE_1, out_features = 512),
                                  nn.ReLU()
        )
        self.dropout_2 = nn.Dropout(p = 0.5)
        self.fc_2 = nn.Sequential(nn.Linear(in_features = 512, out_features = 128),
                                  nn.ReLU()
        )
        self.dropout_3 = nn.Dropout(p = 0.5)
        self.fc_3 = nn.Linear(in_features = 128, out_features = 4)
        
    def forward(self, x):
        x1, x2 = self.conv_1(x), self.conv_2(x)
        x1, x2 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)
        
        x3, x4, x5 = self.conv_3(x), self.conv_4(x), self.conv_5(x)
        x3, x4, x5 = x3.view(x3.size(0), -1), x4.view(x4.size(0), -1), x5.view(x5.size(0), -1)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.dropout_1(x)
        x = self.fc_1(x)
        x = self.dropout_2(x)
        x = self.fc_2(x)
        x = self.dropout_3(x)
        x = self.fc_3(x)
        return x

# The CNN found on GitHub
# This CNN seems to focus more on small-scale and local patterns
class Conv_Net_v2(nn.Module):
    def __init__(self):
        super(Conv_Net_v2, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 128, kernel_size = (2, 1), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 128, kernel_size = (1, 2), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (2, 1), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (2, 1), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_5 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (1, 2), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_6 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (1, 2), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.dropout_1 = nn.Dropout(p = 0.5)
        self.fc_1 = nn.Sequential(nn.Linear(in_features = HIDDEN_SIZE_2, out_features = 1024),
                                  nn.ReLU()
        )
        self.dropout_2 = nn.Dropout(p = 0.5)
        self.fc_2 = nn.Sequential(nn.Linear(in_features = 1024, out_features = 256),
                                  nn.ReLU()
        )
        self.dropout_3 = nn.Dropout(p = 0.5)
        self.fc_3 = nn.Linear(in_features = 256, out_features = 4)
        
    def forward(self, x):
        x1, x2 = self.conv_1(x), self.conv_2(x)
        x3, x4, x5, x6 = self.conv_3(x1), self.conv_4(x2), self.conv_5(x1), self.conv_6(x2)
        
        x1, x2 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)
        x3, x4, x5, x6 = x3.view(x3.size(0), -1), x4.view(x4.size(0), -1), x5.view(x5.size(0), -1), x6.view(x6.size(0), -1)
        
        x = torch.cat((x1, x2, x3, x4, x5, x6), 1)
        x = self.dropout_1(x)
        x = self.fc_1(x)
        x = self.dropout_2(x)
        x = self.fc_2(x)
        x = self.dropout_3(x)
        x = self.fc_3(x)
        return x
    
# The CNN used as in MyOwnAgent
# This CNN is a combination of the above two CNNs,
# which gives out better and stable performances
class Conv_Net_com(nn.Module):
    def __init__(self):
        super(Conv_Net_com, self).__init__()
        self.conv_1 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 128, kernel_size = (2, 1), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_2 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 128, kernel_size = (1, 2), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_3 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (2, 1), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_4 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (2, 1), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_5 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (1, 2), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_6 = nn.Sequential(nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = (1, 2), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_7 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 256, kernel_size = (4, 1), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_8 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 256, kernel_size = (1, 4), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_9 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 256, kernel_size = (2, 2), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_10 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 256, kernel_size = (3, 3), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.conv_11 = nn.Sequential(nn.Conv2d(in_channels = 12, out_channels = 256, kernel_size = (4, 4), stride = 1, padding = 0),
                                    nn.ReLU()
        )
        self.dropout_1 = nn.Dropout(p = 0.5)
        self.fc_1 = nn.Sequential(nn.Linear(in_features = HIDDEN_SIZE_3, out_features = 2048),
                                  nn.ReLU()
        )
        self.dropout_2 = nn.Dropout(p = 0.5)
        self.fc_2 = nn.Sequential(nn.Linear(in_features = 2048, out_features = 512),
                                  nn.ReLU()
        )
        self.dropout_3 = nn.Dropout(p = 0.5)
        self.fc_3 = nn.Linear(in_features = 512, out_features = 4)
        
    def forward(self, x):
        # The convolution structure of Conv_Net_v2
        x1, x2 = self.conv_1(x), self.conv_2(x)
        x3, x4, x5, x6 = self.conv_3(x1), self.conv_4(x2), self.conv_5(x1), self.conv_6(x2)
        
        # Flatten
        x1, x2 = x1.view(x1.size(0), -1), x2.view(x2.size(0), -1)
        x3, x4, x5, x6 = x3.view(x3.size(0), -1), x4.view(x4.size(0), -1), x5.view(x5.size(0), -1), x6.view(x6.size(0), -1)
        
        # The convolution structure of Conv_Net
        x7, x8, x9, x10, x11 = self.conv_7(x), self.conv_8(x), self.conv_9(x), self.conv_10(x), self.conv_11(x)
        x7, x8, x9, x10, x11 = x7.view(x7.size(0), -1), x8.view(x8.size(0), -1) \
            , x9.view(x9.size(0), -1), x10.view(x10.size(0), -1), x11.view(x11.size(0), -1)
        
        # Combine two flattened layer and send it into fully connected layers
        x = torch.cat((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11), 1)
        x = self.dropout_1(x)
        x = self.fc_1(x)
        x = self.dropout_2(x)
        x = self.fc_2(x)
        x = self.dropout_3(x)
        x = self.fc_3(x)
        
        # As I find the softmax layer impacts performance in practice, I discard it here
        return x