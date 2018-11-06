## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

    
class Net(nn.Module):
    
        
    def __init__(self):
        super().__init__()
        
        # output shape: (None, 1024)
        self.init_conv()
        
        # output shape: (None, 1024)
        self.init_loc()
        
        # output layer
        self.fc_combine = nn.Linear(2048, 2048)
        self.bn_combine = nn.BatchNorm1d(2048)
        self.drop_combine = nn.Dropout(p=0.1)
        
        self.fc_out = nn.Linear(2048, 136)
        
        with torch.no_grad():
            I.xavier_uniform_(self.fc_combine.weight)
            I.xavier_uniform_(self.fc_out.weight)
    
    
    def init_loc(self):
        
        # out shape = (56x56) = 3,136
        self.downsample_4 = nn.MaxPool2d(4, stride=4)
        
        # out shape = (28x28) = 784
        self.downsample_8 = nn.MaxPool2d(8, stride=8)
        
        # out shape = (14x14) = 196
        self.downsample_16 = nn.MaxPool2d(16, stride=16)
        
        self.fc_loc = nn.Linear(4116, 1024)
        
        self.bn_loc = nn.BatchNorm1d(1024)
        
        with torch.no_grad():
            I.xavier_uniform_(self.fc_loc.weight)
        
        
    def init_conv(self):
        self.dropout_rate_init = 0.05
        self.dropout_step = 0.05
        
        dropout_rate = self.dropout_rate_init
        
        # in shape=(1, 224, 224), out shape=(16, 224, 224)
        self.conv1 = nn.Conv2d(1, 16, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.drop1 = nn.Dropout2d(p=dropout_rate)
        dropout_rate += self.dropout_step
        
        self.pool = nn.MaxPool2d(2, stride=2)
        
        # in shape = (16, 112, 112), out shape=(32, 112, 112)
        self.conv2 = nn.Conv2d(16, 32, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.drop2 = nn.Dropout2d(p=dropout_rate)
        dropout_rate += self.dropout_step
        
        # in shape = (32, 56, 56), out shape = (64, 56, 56)
        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.drop3 = nn.Dropout2d(p=dropout_rate)
        dropout_rate += self.dropout_step
        
        # in shape = (64, 28, 28), out shape = (128, 24, 24)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.bn4 = nn.BatchNorm2d(128)
        self.drop4 = nn.Dropout2d(p=dropout_rate)
        dropout_rate += self.dropout_step
        
        self.fc1 = nn.Linear(12*12*128, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        
        with torch.no_grad():
            I.xavier_uniform_(self.conv1.weight)
            I.xavier_uniform_(self.conv2.weight)
            I.xavier_uniform_(self.conv3.weight)
            I.xavier_uniform_(self.conv4.weight)
            I.xavier_uniform_(self.fc1.weight)
        
        
    def forward(self, x):
        activation = F.relu
        
        x_loc = self.forward_loc(x)
        x_conv = self.forward_conv(x)
        
        # concatenate conv layers and loc layer
        x = torch.cat((x_conv, x_loc), dim=1)
        
        # combine layer
        x = activation(self.fc_combine(x))
        x = self.bn_combine(x)
        x = self.drop_combine(x)
        
        # output layer
        x = self.fc_out(x)
        
        return x
    
    
    def forward_conv(self, x):
        activation = F.relu
        
        # layer 1
        x = activation(self.conv1(x))
        x = self.bn1(x)
        x = self.pool(x)
        x = self.drop1(x)
        
        # layer 2
        x = activation(self.conv2(x))
        x = self.bn2(x)
        x = self.pool(x)
        x = self.drop2(x)
        
        # layer 3
        x = activation(self.conv3(x))
        x = self.bn3(x)
        x = self.pool(x)
        x = self.drop3(x)
        
        # layer 4
        x = activation(self.conv4(x))
        x = self.bn4(x)
        x = self.pool(x)
        x = self.drop4(x)

#         print(x.size())
        x = x.view(x.size(0), -1)
#         print(x.size())
        
        x = activation(self.fc1(x))
        x = self.bn_fc1(x)
        
        return x
    
   
    def forward_loc(self, x):
        activation = F.relu
        
        x_org_4 = self.downsample_4(x)
        x_org_4 = x_org_4.view(x_org_4.size(0), -1)
        
        x_org_8 = self.downsample_8(x)
        x_org_8 = x_org_8.view(x_org_8.size(0), -1)
        
        x_org_16 = self.downsample_16(x)
        x_org_16 = x_org_16.view(x_org_16.size(0), -1)
        
        x_combine = torch.cat((x_org_4, x_org_8, x_org_16), dim=1)
        
        x_combine = activation(self.fc_loc(x_combine))
        x_combine = self.bn_loc(x_combine)
        
        return x_combine
    
    
    def __str__(self):
        s = super().__str__()
        
        n_parameters = sum(p.numel() for p in self.parameters())
        n_trainable_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        s += '\n'
        s += ('number of parameters          : {:,}\n'.format(n_parameters))
        s += ('number of trainable parameters: {:,}\n'.format(n_trainable_parameters))
        
        return s  
   