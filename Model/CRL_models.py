"""
Cognitive Representation Learning in paper
based on "A Deep Cybersickness Predictor Based on Brain Signal Analysis for Virtual Reality Contents" paper
2022.01.30
"""
from numpy import concatenate
import torch
import torch.nn as nn

class CRLNet(nn.Module):
    def __init__(self, num_classes, input_ch, batch_norm=True, batch_norm_alpha=0.1):
        super(CRLNet, self).__init__()

        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = num_classes
        self.n_channel = input_ch
        
        self.temporalnet = nn.Sequential(
            nn.Conv2d(self.n_channel, (self.n_channel*2), kernel_size = (1, 28), stride = 1, bias = False),
            nn.BatchNorm2d((self.n_channel*2)),
            nn.LeakyReLU(),
            nn.Conv2d((self.n_channel*2), (self.n_channel*2), kernel_size = (1, 14), stride = 2, bias = False, padding=(31,0)), #(32,1)
            nn.BatchNorm2d((self.n_channel*2)),
            nn.LeakyReLU(),
            nn.Conv2d((self.n_channel*2), self.n_channel, kernel_size = (1, 7), stride = 2, bias = False, padding=(31,0)), #(32,0)
            nn.BatchNorm2d(self.n_channel),
            nn.LeakyReLU(),
            # nn.MaxPool2d(kernel_size=0)
        )

        self.spectralnet = nn.Sequential(
            nn.Conv2d(self.n_channel, (self.n_channel*2), kernel_size = (8, 1), stride = 1, bias = False),
            nn.BatchNorm2d((self.n_channel*2)),
            nn.LeakyReLU(),
            nn.Conv2d((self.n_channel*2), (self.n_channel*2), kernel_size = (4, 1), stride = 2, bias = False, padding=(1,27)), #(1,26)
            nn.BatchNorm2d((self.n_channel*2)),
            nn.LeakyReLU(),
            nn.Conv2d((self.n_channel*2), self.n_channel, kernel_size = (2, 1), stride = 2, bias = False, padding=(0,27)), #(1,26)
            nn.BatchNorm2d(self.n_channel),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=(13,1))
        )

        # self.clf = nn.Linear(3304, 128)
        # self.clf2 = nn.Linear(128, self.n_classes)
        self.clf = nn.Linear(3304, self.n_classes)

    def forward(self, x):
        output_temporal = self.temporalnet(x)
        output_temporal = output_temporal.view(output_temporal.size()[0], -1)
        output_spectral = self.spectralnet(x)
        output_spectral = output_spectral.view(output_spectral.size()[0], -1)
        # print(output_temporal.shape)
        # print(output_spectral.shape)
        
        output = torch.cat([output_temporal, output_spectral], dim=1)
        # output = output.view(output.size()[0], -1)
        output = self.clf(output)
        # output = self.clf2(output)

        return output

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='openbmi_gigadb')
    parser.add_argument('--data-root',
                        default='C:/Users/Starlab/Documents/onedrive/OneDrive - 고려대학교/untitled/convert/')
    parser.add_argument('--save-root', default='../data')
    parser.add_argument('--result-dir', default='/deep4net_origin_result_ch20_raw')
    parser.add_argument('--batch-size', type=int, default=400, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--test-batch-size', type=int, default=50, metavar='N',
                        help='input batch size for testing (default: 8)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current best Model')

    args = parser.parse_args()

    '''model = CRLNet(5, 8)
    
    from pytorch_model_summary import summary

    print(summary(model, torch.rand((1, 8, 64, 53)), show_input=True))
    print(summary(model, torch.rand((1, 8, 64, 53)), show_input=False))'''
    # model input = torch.rand((1,1,32,200))
    # batch size, channel, eeg electrodes, time window 
    model = CRLNet(3, 28)
    from pytorch_model_summary import summary
    print(summary(model, torch.rand((1, 28, 63, 55)), show_input=True))
    print(summary(model, torch.rand((1, 28, 63, 55)), show_input=False))