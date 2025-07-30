import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )
            
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, in_channels, ResidualBlock, img_channels, num_classes=10):
        super(ResNet, self).__init__()
        self.res_channel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.res_channel, 
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False),
            nn.BatchNorm2d(self.res_channel),
            nn.ReLU(inplace=True),
            
        )
        self.maxpool2d  =   nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 
    
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)     
        
        self.avgpool    =   nn.AdaptiveAvgPool2d((1,1))  
         
        # self.fc = nn.Linear(512, num_classes) # No need
        
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.res_channel, channels, stride))
            self.res_channel = channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool2d(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # out = self.avgpool(out)   # No need for ALOHA
        out = torch.flatten(out, 1)
        # out = self.fc(out)
        return out
    

def sin_2d_positional_encoding(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    
    source: https://github.com/wzlxjtu/PositionalEncoding2D/blob/master/positionalembedding2d.py
    """
    
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :]       = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :]       = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :]        = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :]    = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


def get_fixed_queries(k, dim):
    pe = torch.zeros(k, dim)
    position = torch.arange(k).unsqueeze(1)      # shape: (k,1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # shape: (k, dim)


class predictive(nn.Module):
    def __init__(self, batch_size, batchNorm=True):
        super(predictive,self).__init__()
        
        self.batch_size = batch_size
        
        self.simple_conv1 = nn.Conv2d(in_channels=4, out_channels=64, kernel_size=5, stride=2, padding=1).to("cuda:1")
        self.simple_conv2 = nn.Conv2d(64,  128, 3, 2, 2).to("cuda:1")
        self.simple_conv3 = nn.Conv2d(128, 256, 3, 2, 2).to("cuda:1")
        self.simple_conv4 = nn.Conv2d(256, 256, 3, 1, 1).to("cuda:1")
        self.simple_conv5 = nn.Conv2d(256, 256, 3, 1, 1).to("cuda:1")
        self.simple_conv6 = nn.Conv2d(256, 512, 3, 1, 1).to("cuda:1")
        
        # RNN
        self.rnn = nn.LSTM(
                    input_size=128*306,
                    hidden_size=1024, 
                    num_layers=4, 
                    dropout=0, 
                    batch_first=True,
                    bidirectional=True).to("cuda:0")
        self.rnn_drop_out = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(2048, 6).to("cuda:1")
        
        self.maxpool_1 = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=2)
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=2)
        self.maxpool_6 = nn.MaxPool2d(kernel_size=3, stride=2, padding=2)
        
        self.conv_bn1 = nn.BatchNorm2d(64).to("cuda:1")
        self.conv_bn2 = nn.BatchNorm2d(128).to("cuda:1")
        self.conv_bn3 = nn.BatchNorm2d(256).to("cuda:1")
        self.conv_bn4 = nn.BatchNorm2d(256).to("cuda:1")
        self.conv_bn5 = nn.BatchNorm2d(256).to("cuda:1")
        self.conv_bn6 = nn.BatchNorm2d(512).to("cuda:1")
        
        
    def forward(self, x):
        batch_size = x.size(0)
        rnn_size = x.size(1)
        
        x = x.view(batch_size * rnn_size, x.size(2), x.size(3), x.size(4))
        
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.maxpool(x)
        
        # CNN
        x = self.encode_image(x)
        
        x = x.view(batch_size, rnn_size, -1)
        
        x, hc = self.rnn(x.to("cuda:0"))

        x = self.rnn_drop_out(x)
        
        x = x.reshape(batch_size * rnn_size, -1)
        
        output = self.fc_part(x.to("cuda:1"))
        
        output = output.reshape(batch_size, rnn_size, -1)

        return output
    
    def encode_image(self, x):
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv1(x)
        x = self.conv_bn1(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv2(x)
        x = self.conv_bn2(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv3(x)
        x = self.conv_bn3(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv4(x)
        x = self.conv_bn4(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv5(x)
        x = self.conv_bn5(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode='circular').to("cuda:1")
        x = self.simple_conv6(x)
        x = self.conv_bn6(x)
        x = F.leaky_relu(x, 0.1)
        return x
    
    def fc_part(self, x):
        x = F.leaky_relu(x, 0.2)
        x = self.fc1(x)
        return x
    
    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class WeightedLoss(nn.Module):
    def __init__(self, learn_hyper_params=True, device="cpu"):
        super(WeightedLoss, self).__init__()
        self.w_rot = 100

    def forward(self, pred, target):
        L_t = F.mse_loss(pred[:,:,:3], target[:,:,:3])
        L_r = F.mse_loss(pred[:,:,3:], target[:,:,3:])
        loss = L_t + L_r * self.w_rot
        return loss
    
def RMSEError(pred, label):
    return torch.sqrt(torch.mean((pred-label)**2))
