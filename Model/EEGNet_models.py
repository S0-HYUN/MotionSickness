"""
Simple EEGNet
based on "~" paper
"""
import torch
import torch.nn as nn

class EEGNet(nn.Module):
    def __init__(self, num_classes, input_ch, input_time, batch_norm=True, batch_norm_alpha=0.1):
        super(EEGNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = num_classes
        freq = input_time #################### frequency
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = (1, freq//2), stride = 1, bias = False, padding = (1 , freq//4)),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, kernel_size = (input_ch, 1), stride = 1, groups = 8),
            nn.BatchNorm2d(16),
            nn.ELU(),
            # nn.AdaptiveAvgPool2d(output_size = (1,265)),
            nn.AvgPool2d(kernel_size=(1,4)),
            nn.Dropout(p=0.25),
            nn.Conv2d(16, 16, kernel_size = (1,freq//4), padding = (0,freq//8), groups = 16),
            nn.Conv2d(16, 16, kernel_size = (1,1)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.2),
            )
    
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        # print("=======", out)
        self.n_outputs = out.size()[1] * out.size()[2] * out.size()[3]

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        # self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.clf(output) 
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

    
    model = EEGNet(3, 28, 750) # n_classes, n_channel, n_timewindow
    # pred = model(torch.zeros(50, 1, 20, 250))
    # print(model)
    
    from pytorch_model_summary import summary
    print(summary(model, torch.rand((1, 1, 28, 750)), show_input=False))
    # model input = torch.rand((1,1,32,200))
    # batch size, channel, eeg electrodes, time window 