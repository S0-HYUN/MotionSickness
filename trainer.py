from numpy.lib.function_base import average
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

class trainer:
    def __init__(self, args, model, epoch, lr, optimizer, weight_decay = None, criterion = 'CrossEntropyLoss'):
            self.writer = SummaryWriter(log_dir='./runs/lr{}_wd{}'.format(lr, weight_decay))
            ##tensorboard --logdir=C:/Users/soso/Desktop/tensorboard/runs --port 6006
            optimizer = getattr(torch.optim, optimizer)
            criterion = getattr(torch.nn, criterion)
            self.model = model
            self.optimizer = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
            self.epoch = epoch
            self.criterion = criterion()
            self.lr = lr
            self.wd = weight_decay
            self.channel_num = args.channel_num

    def train(self, data, data_v, batch_size, shuffle=True, interval=1000):
        # train_data, valid_data = torch.utils.data.random_split(data, [int(len(data) * 0.8), (len(data) - int(len(data) * 0.8))]) # data.split(split_ratio = 0.8)
        train_data = data
        valid_data = data_v

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # print(data.y[:int(0.8*len(data))].shape)
        # train_data = Dataset(x = data.x[ : int(0.8 * len(data))], y = data.y[ : int(0.8 * len(data))])
        # valid_data = Dataset(x = data.x[int(0.8 * len(data)) : ], y = data.y[int(0.8 * len(data)) : ])
        prev_v = -10
        data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = shuffle)

        for e in tqdm(range(self.epoch)):
            epoch_loss = 0
            self.model.train() # train 시키기
            pred_list = []
            true_label = []
            for idx, data in enumerate(data_loader): # enumerate -> batch size 정한만큼 꺼내오기.
                x, y = data # 튜플에서 x, y 하나씩 넣기
                true_label.append(y)
                b = x.shape[0] # 30 --> 1?
                
                self.optimizer.zero_grad() # optimizer 항상 초기화 
                #x = x.reshape(b,1,30,-1) ############## checking [1, 1, 30, 700] 미츼ㅣㅣ
                x = x.reshape(b, 1, self.channel_num, -1) # [1, 1, 25, 750]
                pred = self.model(x.to(device=device).float())

                # pred, (hidden_state, cell_state) = network(x.view(b, 1, -1).float().to(device=device), (hidden_state[:,:b].detach().contiguous(), cell_state[:,:b].detach().contiguous())) # view함수 -> 밑에 적음
                loss = self.criterion(pred, y.flatten().long().to(device=device)) 
                if (idx+1) % interval == 0: print('[Epoch{}, Step({}/{})] Loss:{:.4f}'.format(e+1, idx+1, len(data_loader.dataset)//batch_size, epoch_loss / (idx +1)))

                loss.backward()
                epoch_loss += loss.item()
                self.optimizer.step()
                pred_prob = F.softmax(pred, dim=-1)
                pred_label = torch.argmax(pred_prob, dim=-1).cpu().numpy()
                pred_list.append(pred_label) # pred_list에 prediction 넣어주기

            pred_list = np.concatenate(pred_list)
            true_label = np.concatenate(true_label)
            
            # print("true_label:", true_label)
            # print(true_label)
            # print("\n\n\n", pred_list)
            f1 = f1_score(true_label, pred_list, average='macro') 
            acc = accuracy_score(true_label, pred_list)
            cm = confusion_matrix(true_label, pred_list)
            epoch_loss = epoch_loss / (idx+1)

            print('\nEpoch{} Training, f1:{:.4f}, acc:{:.4f}, Loss:{:.4f}'.format(e+1, f1, acc, epoch_loss))
            print(cm)

            f1_v, acc_v, cm_v, loss_v = self.prediction(valid_data, batch_size)
            if f1_v > prev_v : 
                prev_v = f1_v
                if not os.path.exists('./param/lr{}_wd{}'.format(self.lr, self.wd)): ################################################ 위치
                    os.mkdir('./param/lr{}_wd{}'.format(self.lr, self.wd))
                torch.save(self.model.state_dict(), './param/lr{}_wd{}/eegnet_f1_{:.2f}'.format(self.lr, self.wd, f1_v))
                # torch.save(self.model.state_dict(), '/opt/workspace/soxo/code_ms/param/lr{}_wd{}/eegnet_f1_{:.2f}'.format(self.lr, self.wd, f1_v))

            # writer.add_scalar('Learning Rate', lr.get_last_lr()[-1], e)
            self.writer.add_scalar('Train/Loss', epoch_loss, e)
            self.writer.add_scalar('Train/F1', f1, e)
            self.writer.add_scalar('Train/Acc', acc, e)

            self.writer.add_scalar('Valid/Loss', loss_v, e)
            self.writer.add_scalar('Valid/F1', f1_v, e)
            self.writer.add_scalar('Valid/Acc', acc_v, e)
            self.writer.flush()
        return acc, f1, cm, epoch_loss


    def only_train(self, data, batch_size, shuffle=True, interval=1000):
        # train_data, valid_data = torch.utils.data.random_split(data, [int(len(data) * 0.8), (len(data) - int(len(data) * 0.8))]) # data.split(split_ratio = 0.8)
        train_data = data
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # print(data.y[:int(0.8*len(data))].shape)
        # train_data = Dataset(x = data.x[ : int(0.8 * len(data))], y = data.y[ : int(0.8 * len(data))])
        # valid_data = Dataset(x = data.x[int(0.8 * len(data)) : ], y = data.y[int(0.8 * len(data)) : ])
        prev_v = -10
        data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = shuffle)

        for e in tqdm(range(self.epoch)):
            epoch_loss = 0
            self.model.train() # train 시키기
            pred_list = []
            true_label = []
            for idx, data in enumerate(data_loader): # enumerate -> batch size 정한만큼 꺼내오기.
                x, y = data # 튜플에서 x, y 하나씩 넣기

                true_label.append(y)
                b = x.shape[0] # 30 --> 1?
                
                self.optimizer.zero_grad() # optimizer 항상 초기화 
                #x = x.reshape(b,1,30,-1) ############## checking [1, 1, 30, 700] 미츼ㅣㅣ
                x = x.reshape(b, 1, self.channel_num, -1) # [1, 1, 25, 750]
                pred = self.model(x.to(device=device).float())

                # pred, (hidden_state, cell_state) = network(x.view(b, 1, -1).float().to(device=device), (hidden_state[:,:b].detach().contiguous(), cell_state[:,:b].detach().contiguous())) # view함수 -> 밑에 적음
                loss = self.criterion(pred, y.flatten().long().to(device=device))
                if (idx+1) % interval == 0: print('[Epoch{}, Step({}/{})] Loss:{:.4f}'.format(e+1, idx+1, len(data_loader.dataset)//batch_size, epoch_loss / (idx +1)))

                loss.backward()
                epoch_loss += loss.item()
                self.optimizer.step()
                pred_prob = F.softmax(pred, dim=-1)
                pred_label = torch.argmax(pred_prob, dim=-1).cpu().numpy()
                pred_list.append(pred_label) # pred_list에 prediction 넣어주기

            pred_list = np.concatenate(pred_list)
            true_label = np.concatenate(true_label)
            
            # print("true_label:", true_label)
            # print(true_label)
            # print("\n\n\n", pred_list)
            f1 = f1_score(true_label, pred_list, average='macro') 
            acc = accuracy_score(true_label, pred_list)
            cm = confusion_matrix(true_label, pred_list)
            epoch_loss = epoch_loss / (idx+1)

            print('\nEpoch{} Training, f1:{:.4f}, acc:{:.4f}, Loss:{:.4f}'.format(e+1, f1, acc, epoch_loss))
            print(cm)

            if f1 > prev_v : 
                prev_v = f1
                if not os.path.exists('./param/lr{}_wd{}'.format(self.lr, self.wd)): ################################################ 위치
                    os.mkdir('./param/lr{}_wd{}'.format(self.lr, self.wd))
                torch.save(self.model.state_dict(), './param/lr{}_wd{}/eegnet_f1_{:.2f}'.format(self.lr, self.wd, f1))
                # torch.save(self.model.state_dict(), '/opt/workspace/soxo/code_ms/param/lr{}_wd{}/eegnet_f1_{:.2f}'.format(self.lr, self.wd, f1_v))


            # writer.add_scalar('Learning Rate', lr.get_last_lr()[-1], e)
            self.writer.add_scalar('Train/Loss', epoch_loss, e)
            self.writer.add_scalar('Train/F1', f1, e)
            self.writer.add_scalar('Train/Acc', acc, e)

            self.writer.flush()
        return acc, f1, cm, epoch_loss


    def prediction(self, data, batch_size, interval=1000, test=False):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        data_loader = DataLoader(data, batch_size = batch_size)

        with torch.no_grad(): # gradient 안함
            self.model.eval() # dropout은 training일 때만, evaluation으로 하면 dropout 해제
            pred_list = []
            true_label = []
            valid_loss = 0
            
            for idx, data in enumerate(data_loader):
                x, y = data
                true_label.append(y)
                b = x.shape[0]

                # 밑에 두개 중에뭐지?
                pred = self.model(x.transpose(1,2).reshape(b,1,self.channel_num,-1).to(device=device).float())
                #pred = self.model(x.reshape(b,1,30,-1).to(device=device).float()) #self.model(x.transpose(1,2).reshape(b,1,30,-1).to(device=device).float())
                loss = self.criterion(pred, y.flatten().long().to(device=device)) # pred.shape
                valid_loss += loss
                if (idx+1) % interval == 0: print('[Epoch, Step({}/{})] Valid Loss:{:.4f}'.format(idx+1, len(data)//batch_size, loss / (idx +1)))
                pred_prob = F.softmax(pred, dim=-1)
                pred_label = torch.argmax(pred_prob, dim = -1).cpu().numpy()
                pred_list.append(pred_label)

            valid_loss = valid_loss / (idx+1)            
            pred_list = np.concatenate(pred_list)
            true_label = np.concatenate(true_label)
            
            f1 = f1_score(true_label, pred_list, average='macro')
            acc = accuracy_score(true_label, pred_list)
            cm = confusion_matrix(true_label, pred_list)
            if not test:
                print('\nEpoch Validation, f1:{:.4f}, acc:{:.4f}, Loss:{:.4f}'.format(f1, acc, valid_loss))
            else:
                print('\nEpoch Test, f1:{:.4f}, acc:{:.4f}'.format(f1, acc))
            print(cm)

        return f1, acc, cm, valid_loss
     