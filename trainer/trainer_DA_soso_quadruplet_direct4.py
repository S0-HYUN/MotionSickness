from random import sample
from numpy.lib.function_base import average
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
# from torch.utils.tensorboard import SummaryWriter
import os
from utils import *
from loss.ND_Crossentropy import CrossentropyND, TopKLoss
from loss.focalloss import *
from loss.label_smoothing import LabelSmoothingCrossEntropy
# from loss.quadruplet import QuadrupletLoss
from loss.quadruplet2 import sosoLoss
from utils import gpu_checking
from collections import defaultdict
import wandb
from scipy.spatial import distance

from utils_drawing import visualizer
'''
distance 벡터 안쓰고 
그냥 quadruplet + MSE 씀
'''

class TrainMaker:
    def __init__(self, args, model, data, data_v=None, data_r=None, data_v_r=None, DA=False):
        self.args = args
        self.model = model
        self.data = data
        self.data_v = None
        
        # can be None
        if data_v : self.data_v = data_v 
        if data_r : self.data_r = data_r
        if data_v_r : self.data_v_r = data_v_r

        self.history = defaultdict(list)
        self.history_mini_batch = defaultdict(list)
        self.optimizer = getattr(torch.optim, self.args.optimizer)
        self.channel_num = self.args.channel_num
        self.device = gpu_checking(args)

        if DA == False:
            # self.writer = SummaryWriter(log_dir=f'./runs/{self.args.test_subj}/lr{self.args.lr}_wd{self.args.wd}')
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
            self.epoch = self.args.epoch
            self.lr = self.args.lr
            self.wd = self.args.wd

        elif DA == True:
            # self.writer = SummaryWriter(log_dir=f'./runs/lr{self.args.lr}_wd{self.args.wd}')
            self.optimizer = self.optimizer(self.model.parameters(), lr=self.args.da_lr)
            self.epoch = self.args.da_epoch
            self.lr = self.args.da_lr
            self.wd = 0 ##이게 맞나?

        self.scheduler = self.__set_scheduler(args, self.optimizer)
        self.cal = Calculate()

        # self.criterion = self.__set_criterion(self.args.criterion)   
        self.criterion_mse = nn.MSELoss()
        self.criterion_quad = sosoLoss()

    def training(self, shuffle=True, interval=1000):
        if self.args.standard == 'loss': prev_v = 1000
        else : prev_v = -10
        
        sampler = self.data.bins
        data_loader = DataLoader(self.data, batch_size=1, sampler=sampler)
        # sampler = torch.utils.data.WeightedRandomSampler(self.data.in_weights, replacement=True, num_samples=self.data.x.shape[0])
        # data_loader = DataLoader(self.data, batch_size=self.args.batch_size)
        data_rest_loader = DataLoader(self.data_r, batch_size=self.args.batch_size, sampler=sampler) #근데 사실 필요없는 게 아닐까.
        
        for e in tqdm(range(self.epoch)):
            epoch_loss = 0
            self.model.train()

            pred_label_acc = None; true_label_acc = None
            pred_list = None; true_label = None

            # f1, acc, cm, loss = self.evaluation(self.data, state="train") # train데이터 eval 끄고 evaluation
            for idx, data in enumerate(data_loader):
                x, y, subj = data
                x = x.squeeze(0); y = y.squeeze(0); subj = subj.squeeze(0)

                true_label = y.numpy(); y_vector = change_vector(y) #label extension
                if x.shape[0] != self.args.batch_size : continue
                if len(set(true_label)) != 3 : continue
                
                self.optimizer.zero_grad()
                x = x.reshape(self.args.batch_size, 1, self.channel_num, -1) # [1, 1, 25, 750]

                #---# MS data #---#
                MS_feature, pred = self.model(x.to(device=self.device).float(), subj.to(self.device), embedding=True)

                #---# rest data #---#
                random_vec = np.random.randint(0, len(data_rest_loader.dataset), size=self.args.batch_size)
                x_rest, y_rest, subj_rest = data_rest_loader.dataset[random_vec]
                x_rest = x_rest.reshape(self.args.batch_size, 1, self.channel_num, -1)
                rest_feature, rest_clf = self.model(x_rest.to(device=self.device).float())#######################
                rest_proto = rest_feature.mean(0) # calculate prototype
                rest_proto_vec = rest_proto.repeat(pred.shape[0]).view(self.args.batch_size, -1)

                # distance_vec = torch.norm(torch.dstack([pred, rest_clf]), dim=2)
                pred_label = (pred>0.5).cumprod(axis=1).sum(axis=1)-1; pred_label[pred_label<0] = 1

                #---# Calculate loss #---#
                loss_quad = self.criterion_quad(true_label, MS_feature, rest_proto)
                loss_mse = self.criterion_mse(pred, y_vector.float().to(device=self.device))
            
                if (idx+1) % interval == 0: print('[Epoch{}, Step({}/{})] Loss:{:.4f}'.format(e+1, idx+1, len(data_loader.dataset) // self.args.batch_size, epoch_loss / (idx + 1)))
                
                loss = 0.7*loss_quad + loss_mse
                loss.backward()
                epoch_loss += loss
                self.optimizer.step()

                if pred_label_acc is None:
                    pred_label_acc = pred_label
                    true_label_acc = y.to(self.device)
                else:
                    # pred_label_acc = np.concatenate((pred_label_acc, pred_label.cpu()), axis=None)
                    pred_label_acc = torch.cat([pred_label_acc, pred_label], dim=0)
                    true_label_acc = torch.cat([true_label_acc, y.to(self.device)], dim=0)

                # Calculate log per mini-batch
                log = self.cal.calculator(metrics=self.args.metrics, loss=loss, y_true=y, y_pred=pred_label, acc_count=True)
                
                # Record history per mini-batch
                self.record_history(log, self.history_mini_batch, phase='train')
           
            f1 = f1_score(true_label_acc.detach().cpu(), pred_label_acc.detach().cpu(), average='macro') 
            acc = self.cal.get_acc(true_label_acc, pred_label_acc); cm = 0
            # cm = confusion_matrix(true_label_acc, pred_label_acc)
            epoch_loss = epoch_loss / (idx+1)
        

            if self.data_v != None:
                f1_v, acc_v, cm_v, loss_v = self.evaluation(self.data_v)
                if self.args.standard == 'loss':
                    if loss_v < prev_v :
                        prev_v = loss_v
                        create_folder(f'./param/lr{self.lr}_wd{self.wd}')
                        torch.save(self.model.state_dict(), f'./param/lr{self.lr}_wd{self.wd}/{self.args.model}_loss_{loss_v:.2f}')
                        self.save_checkpoint(epoch=len(self.history['train_loss']))

                elif self.args.standard == 'f1':
                    if f1_v > prev_v : 
                        prev_v = f1_v 
                        create_folder(f'./param/lr{self.lr}_wd{self.wd}')
                        torch.save(self.model.state_dict(), f'./param/lr{self.lr}_wd{self.wd}/{self.args.model}_f1_{f1_v:.2f}')
                        self.save_checkpoint(epoch=len(self.history['train_loss']))

                elif self.args.standard == 'acc':
                    if acc_v > prev_v : 
                        prev_v = f1_v 
                        create_folder(f'./param/lr{self.lr}_wd{self.wd}')
                        torch.save(self.model.state_dict(), f'./param/lr{self.lr}_wd{self.wd}/{self.args.model}_f1_{acc_v:.2f}')
                        self.save_checkpoint(epoch=len(self.history['train_loss']))
            else:
                self.save_checkpoint(epoch=len(self.history['train_loss']))
            
            self.write_history(self.history_mini_batch)

            f1, acc, cm, loss = self.evaluation(self.data, state="train") # train데이터 eval 끄고 evaluation
            if self.args.mode == "train":
                wandb.log({"loss": epoch_loss,
                            "acc": acc,
                            "f1":f1,
                            "vloss": loss_v,
                            "vacc": acc_v,
                            "vf1":f1_v,
                            "loss_quad":loss_quad,
                            "loss_mse":loss_mse
                            # "lr": self.optimizer.state_dict().get('param_groups')[0].get("lr")
                })

            if self.args.scheduler != None:
                self.scheduler.step()

        return acc, f1, cm, epoch_loss
        # return f1_v, acc_v, cm_v, loss_v

    def evaluation(self, data, interval=1000, state=None):
        # flag = list(self.model._modules)[2] #################
        # final_layer = self.model._modules.get(flag)
        # activated_features = FeatureExtractor(final_layer) #############################

        data_loader = DataLoader(data, batch_size=self.args.batch_size)
        data_rest_loader = DataLoader(self.data_v_r, batch_size=self.args.batch_size)
        with torch.no_grad():
            self.model.eval() 

            pred_label_acc = None; true_label_acc = None
            pred_list = None; true_label = None
            valid_loss = 0
            features = None # for save features

            # test_embeds = None
            for idx, data in enumerate(data_loader):
                x, y, subj = data
                true_label = y.numpy(); y_vector = change_vector(y)
                b = x.shape[0]
                if x.shape[0] != self.args.batch_size : continue
                x = x.reshape(b, 1, self.channel_num, -1)

                #---# MS data #---#
                MS_feature, pred = self.model(x.to(device=self.device).float(), subj.to(self.device).float())
                if features == None : features = MS_feature
                else : features = torch.cat([features, MS_feature], dim = 0)
                
                #---# rest data #---#
                # rest_clf = self.model(x_rest.to(device=self.device).float(), fc=True)
                rest_proto = get_meanPrototype(feature=True)
                rest_proto = torch.tensor(rest_proto).to(self.device)
                rest_proto_vec = rest_proto.repeat(pred.shape[0]).view(self.args.batch_size, -1)

                # distance_vec = torch.norm(torch.dstack([pred, rest_proto_vec]), dim=2)
                pred_label = (pred>0.5).cumprod(axis=1).sum(axis=1)-1; pred_label[pred_label<0] = 1
                # loss = self.criterion(distance_vec, y_vector.float().to(device=self.device))
                
                loss_quad = self.criterion_quad(true_label, MS_feature, rest_proto)
                loss_mse = self.criterion_mse(pred, y_vector.float().to(self.device))
                
                loss = 0.7*loss_quad + loss_mse
                valid_loss += loss

                if pred_label_acc is None:
                    pred_label_acc = pred_label
                    true_label_acc = y.to(self.device)
                else:
                    # pred_label_acc = np.concatenate((pred_label_acc, pred_label.cpu()), axis=None)
                    pred_label_acc = torch.cat([pred_label_acc, pred_label], dim=0)
                    true_label_acc = torch.cat([true_label_acc, y.to(self.device)], dim=0)

                #---# for t-sne #---#
                # embeds = activated_features.features_before # [256, 200, 1, 4]; embeds =  embeds.squeeze(2) # [256, 200, 4]
                # if test_embeds == None: test_embeds = embeds
                # else : test_embeds = torch.cat((test_embeds, embeds), dim=0) # [103, 800]
       
                log = self.cal.calculator(metrics=self.args.metrics, loss=loss, y_true=y, y_pred=pred_label, acc_count=True)
                
                # Record history per mini-batch
                if state == None:
                    self.record_history(log, self.history_mini_batch, phase='val')
                    
            f1 = f1_score(true_label_acc.detach().cpu(), pred_label_acc.detach().cpu(), average='macro')
            acc = self.cal.get_acc(true_label_acc, pred_label_acc)
            cm = confusion_matrix(true_label_acc.cpu(), pred_label_acc.cpu())
            if not self.args.mode == "test" and state == None:
                print('\nEpoch Validation, f1:{:.4f}, acc:{:.4f}, Loss:{:.4f}'.format(f1, acc, valid_loss))
            elif self.args.mode == "test":
                print('\nEpoch Test, f1:{:.4f}, acc:{:.4f}'.format(f1, acc))
            
            if self.args.mode == "test":
                # savefeature
                current_time = get_time()
                create_folder(f"./features/features_{self.args.model}_{self.args.standard}_{self.args.class_num}")
                np.savez(f"./features/features_{self.args.model}_{self.args.standard}_{self.args.class_num}/{current_time}_original_subj{str(self.args.test_subj).zfill(2)}", features.detach().cpu().numpy(), true_label_acc.detach().cpu().numpy())
        return f1, acc, cm, valid_loss

    def record_history(self, log, history, phase):
        for metric in log:
            history[f'{phase}_{metric}'].append(log[metric])
       
    def write_history(self, history):
        for metric in history:
            if metric.endswith('acc'):
                n_samples = self.data.x[0].shape[0]
                # n_samples = len(getattr(self.data, f"{metric.split('_')[0]}_loader").dataset.y)
                self.history[metric].append((sum(history[metric]) / n_samples))
            else:
                self.history[metric].append(sum(history[metric]) / len(history[metric]))
        
        # if self.args.mode == 'train':
        #     write_json(os.path.join(self.args.save_path, "history.json"), self.history)
        # else:
        #     write_json(os.path.join(self.args.save_path, "history_test.json"), self.history)

    def save_checkpoint(self, epoch):
        create_folder(os.path.join(self.args.save_path, "checkpoints"))
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # 'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None
        }, os.path.join(self.args.save_path, f"checkpoints/{epoch}.tar"))
        # if epoch >= 6:
        #     os.remove(os.path.join(self.args.save_path, f"checkpoints/{epoch - 5}.tar"))
    
    def __set_criterion(self, criterion):
        if criterion == "MSE":
            criterion = nn.MSELoss()
        elif criterion == "CEE":
            criterion = nn.CrossEntropyLoss()
        elif criterion == "Focal":
            criterion = FocalLoss(gamma=2)
        elif criterion == "ND":
            criterion = CrossentropyND()
        elif criterion == "TopK":
            criterion = TopKLoss()
        elif criterion == "LS":
            criterion = LabelSmoothingCrossEntropy()
        elif criterion == "cosine":
            criterion = nn.CosineEmbeddingLoss()
        # elif criterion == "quad":
        #     criterion = QuadrupletLoss()
            
        return criterion
    
    def __set_scheduler(self, args, optimizer):
        if args.scheduler is None:
            return None
        elif args.scheduler == 'exp':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma)
        elif args.scheduler == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        elif args.scheduler == 'multi_step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
        elif args.scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20,
                                                             threshold=0.1, threshold_mode='abs', verbose=True)
        elif args.scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             T_max=args.T_max if args.T_max else args.epochs,
                                                             eta_min=args.eta_min if args.eta_min else 0)
        elif args.scheduler == 'one_cycle':
            scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=args.max_lr, 
                                                    steps_per_epoch=args.steps_per_epoch,
                                                    epochs=args.cycle_epochs)
        else:
            raise ValueError(f"Not supported {args.scheduler}.")
        return scheduler

    def enable_dropout(self, model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()


    # def __make_trainer(self, **kwself.args):
    #     module = importlib.import_module(f"trainers.{kwself.args['self.args'].model}_trainer")
    #     trainer = getattr(module, 'Trainer')(**kwargs)
    #     return trainer

class Calculate:
    def calculator(self, metrics, loss, y_true, y_pred, numpy=True, **kwargs):
        if numpy:
            y_true = self.guarantee_numpy(y_true)
            y_pred = self.guarantee_numpy(y_pred)

        history = defaultdict(list)
        for metric in metrics:
            history[metric] = getattr(self, f"get_{metric}")(loss=loss, y_true=y_true, y_pred=y_pred, **kwargs)
        return history

    def get_loss(self, loss, **kwargs):
            return float(loss)

    def get_acc(self, y_true, y_pred, acc_count=False, **kwargs):
        if acc_count:
            return sum(y_true == y_pred)
        else:
            return sum(y_true == y_pred) / len(y_true)
    
    def get_f1(self, y_true, y_pred):
        tp = (y_true * y_pred).sum()
        tn = ((1 - y_true) * (1 - y_pred)).sum()
        fp = ((1 - y_true) * y_pred).sum()
        fn = (y_true * (1 - y_pred)).sum()
        
        epsilon = 1e-7
        
        precision = tp / (tp + fp + epsilon)
        recall = tp / (tp + fn + epsilon)
        
        f1 = 2 * (precision*recall) / (precision + recall + epsilon)
        return f1

    def guarantee_numpy(self, data):
        data_type = type(data)
        if data_type == torch.Tensor:
            device = data.device.type
            if device == 'cpu':
                data = data.detach().numpy()
            else:
                data = data.detach().cpu().numpy()
            return data
        elif data_type == np.ndarray or data_type == list:
            return data
        else:
            raise ValueError("Check your data type.")