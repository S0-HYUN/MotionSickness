from numpy.core.fromnumeric import reshape, transpose
import torch
import torch.nn as nn

class DeepConvNet(nn.Module):
    def __init__(self, n_classes, input_ch, input_time, batch_norm=True, batch_norm_alpha=0.1):
        super(DeepConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 25
        n_ch2 = 50
        n_ch3 = 100
        self.n_ch4 = 200

        if self.batch_norm:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1), # 10 -> 5
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch2,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch3,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(self.n_ch4,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                )
        else:
            self.convnet = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 10), stride=1,bias=False),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1),
                # nn.InstanceNorm2d(n_ch1),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch1, n_ch2, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch2),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch2, n_ch3, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(n_ch3),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
                nn.Dropout(p=0.5),
                nn.Conv2d(n_ch3, self.n_ch4, kernel_size=(1, 10), stride=1),
                # nn.InstanceNorm2d(self.n_ch4),
                nn.ELU(),
                nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            )
        self.convnet.eval()
        out = self.convnet(torch.zeros(1, 1, input_ch, input_time))
        # out = torch.zeros(1, 1, input_ch, input_time)
        #
        # for i, module in enumerate(self.convnet):
        #     print(module)
        #     out = module(out)
        #     print(out.size())
        #

        n_out_time = out.cpu().data.numpy().shape[3]
        self.final_conv_length = n_out_time

        self.n_outputs = out.size()[1]*out.size()[2]*out.size()[3]

        self.clf = nn.Sequential(nn.Linear(self.n_outputs, self.n_classes), nn.Dropout(p=0.2))  ####################### classifier 
        # DG usually doesn't have classifier
        # so, add at the end

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        # output = self.l2normalize(output)
        output=self.clf(output) 

        return output

    def get_embedding(self, x):
        return self.forward(x)

    def l2normalize(self, feature):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
        return torch.div(feature, norm)


class ShallowConvNet(nn.Module):
    def __init__(self, n_classes, input_ch, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 40

        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 13), stride=1, padding=(6,7)),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))
        
        self.fc = nn.Linear(51480, n_classes)
                                            
    def forward(self, x):
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, (1,35), (1,7))
        x = torch.log(x)
        x = x.flatten(1)
        x = torch.nn.functional.dropout(x)
        x = self.fc(x)
        return x

class ShallowConvNet_dk(nn.Module):
    def __init__(self,n_classes,input_ch,input_time):
        super(ShallowConvNet_dk, self).__init__()
        self.num_filters = 40
        self.n_classes = n_classes

        self.convnet = nn.Sequential(nn.Conv2d(1, self.num_filters, (1,25), stride=1),
                                     nn.Conv2d(self.num_filters, self.num_filters, (input_ch, 1), stride=1,bias=False),
                                     nn.BatchNorm2d(self.num_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                     square(),
                                     nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 1), padding=0),
                                     log(),
                                     nn.Dropout(p=0.5),
                                     nn.Conv2d(self.num_filters, 4, kernel_size=(1, 30),  stride=(1, 1), dilation=(1, 15)),
                                     nn.LogSoftmax(dim=1)
                                     )

        self.convnet.eval()
        out = self.convnet(torch.zeros((1, 1, input_ch, input_time)))
        self.out_size = out.size()[3]

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], self.n_classes,self.out_size)

        return output
    '''def __init__(self, n_classes, input_ch, batch_norm=True, batch_norm_alpha=0.1):
        super(ShallowConvNet_dk, self).__init__()
        self.batch_norm = batch_norm
        self.batch_norm_alpha = batch_norm_alpha
        self.n_classes = n_classes
        n_ch1 = 40
        print("input", input_ch)
        if self.batch_norm:
            self.layer1 = nn.Sequential(
                nn.Conv2d(1, n_ch1, kernel_size=(1, 25), stride=1),
                nn.Conv2d(n_ch1, n_ch1, kernel_size=(input_ch, 1), stride=1, bias=not self.batch_norm),
                nn.BatchNorm2d(n_ch1,
                               momentum=self.batch_norm_alpha,
                               affine=True,
                               eps=1e-5))

        self.fc = nn.Linear(2760, n_classes)
        self.fc = nn.Linear(1760, n_classes)
                                            
    def forward(self, x):
        x = self.layer1(x)
        x = torch.square(x)
        x = torch.nn.functional.avg_pool2d(x, kernel_size = (1,75), stride =  (1,15))
        x = torch.log(x)
        x = x.flatten(1)
        x = torch.nn.functional.dropout(x) # shape [1, 1760]
        x = self.fc(x)
        return x'''
# [8,1,22,1125] -> [8,4,1,592] -> [8,2368] -> [8,2368]

class square(nn.Module):
    def __init__(self):
        super(square, self).__init__()

    def forward(self, x):
        out = x*x
        return out

class log(nn.Module):
    def __init__(self):
        super(log, self).__init__()

    def forward(self, x):
        out = self._log(x)
        return out

    def _log(self, x, eps=1e-6):
        return torch.log(torch.clamp(x, min=eps))


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

    # args = parser.parse_args()
    
    # model = DeepNet_origin(2, 32, 200) # n_classes, n_channel, n_timewindow -> 원본은  3, 62, 750
    # # pred = model(torch.zeros(50, 1, 20, 250))
    # print(model)
    # from pytorch_model_summary import summary

    # print(summary(model, torch.rand((1, 1, 32, 200)), show_input=False))
    # # model input = torch.rand((1,1,32,200))
    # # batch size, channel, eeg electrodes, time window 

    args = parser.parse_args()
    # model = DeepConvNet(2,32,200)
    # model = ShallowConvNet_dk(1,30,725)
    model = ShallowConvNet_dk(1,22,1125)
    
    # model = FcClfNet(embedding_net)
    from pytorch_model_summary import summary

    # print(summary(model, torch.zeros((1, 1, 28, 750)), show_input=False))
    print(summary(model, torch.zeros((1, 1, 22, 1125)), show_input=False))