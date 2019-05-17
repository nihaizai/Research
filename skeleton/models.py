import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

import math

class ConvNet(nn.Module):
    """LeNet++ as described in the Center Loss paper."""
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)  #Conv.1
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 64, 3, stride=1, padding=1) #Conv.2
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  #Conv.3
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)  #Conv.4
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1) #Conv.5
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1)  #Conv.6
        self.prelu3_2 = nn.PReLU()
        self.conv3_3 = nn.Conv2d(128,128,3,stride=1,padding=1)  #Conv.7
        self.prelu3_3 = nn.PReLU()
        
        self.conv4_1 = nn.Conv2d(128,128,3,stride=1,padding=1)  #Conv.8
        self.prelu4_1 = nn.PReLU()
        self.conv4_2 = nn.Conv2d(128,128,3,stride=1,padding=1)  #Conv.9
        self.prelu4_2 = nn.PReLU()
        
        self.conv5_1 = nn.Conv2d(128,128,3,stride=1,padding=1)  #Conv.10
        self.prelu5_1 = nn.PReLU()
        
        
        self.fc1 = nn.Linear(128*60*60, 512)  #FC.1
        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(512,2)  
        self.prelu_fc2 = nn.PReLU()
        self.fc3 = nn.Linear(2, num_classes)

    def forward(self, x):
        #print("x.shape:{}".format(x.shape))     #[batch_size,1,H(240),W(240)]        
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        out1 = F.max_pool2d(x, 2)        
        #print("out1.shape:{}".format(out1.shape))  #[batch_size,64,H/2,W/2]
        
        
        out2 = self.prelu2_1(self.conv2_1(out1))
        out2 = self.prelu2_2(self.conv2_2(out2))
        out2 = out2 + out1    
        #print("out2.shape:{}".format(out2.shape)) #[batch_size,64,H/2,W/2]
        #out2 = F.max_pool2d(out2, 2)
        
        out3 = self.prelu3_1(self.conv3_1(out2))
        out3 = F.max_pool2d(out3,2)
        #print("out3.shape:{}".format(out3.shape))  #[batch_size,128,H/4,W/4]
        
        out4 = self.prelu3_2(self.conv3_2(out3))
        out4 = self.prelu3_3(self.conv3_3(out4))
        out4 = out4 + out3 
        #print("out4.shape:{}".format(out4.shape))  #[batch_size,128,H/4,W/4]
        
        out5 = self.prelu4_1(self.conv4_1(out4))
        out5 = self.prelu4_2(self.conv4_2(out5))
        out5 = out5 + out4
        #print("out5.shape:{}".format(out5.shape))  #[batch_size,128,H/4,W/4]
        
        out6 = self.prelu5_1(self.conv5_1(out5))
        #print("out6.shape:{}".format(out6.shape))  #[batch_size,128,H/4,W/4]
        
        out6 = out6.view(-1, 128*60*60)
        out6 = self.prelu_fc1(self.fc1(out6))
        out7 = self.prelu_fc2(self.fc2(out6))
        y = self.fc3(out7)

        return out6,out7, y

__factory = {
    'cnn': ConvNet,
}

def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)

if __name__ == '__main__':
    test_net = create('cnn',10)
    test_x = Variable(torch.zeros(5,1,240,240))
    feature1,feature2,class_label = test_net(test_x)
    print('feature1.shape:{}'.format(feature1.shape))  #[batch_size,512]
    print('feature2.shape:{}'.format(feature2.shape))  #[batch_size,2]
#    pass
