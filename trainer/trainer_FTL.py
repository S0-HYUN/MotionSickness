##################################################################################################
# FTL Draft Code for Subject-adaptive Analysis
# Author：Ce Ju, Dashan Gao
# Date  : July 29, 2020
# Paper : Ce Ju et al., Federated Transfer Learning for EEG Signal Classification, IEEE EMBS 2020.
# Description: Source domain inlcudes all good subjects, target domain is the bad subject.
##################################################################################################
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
from torch.utils.tensorboard import SummaryWriter
import os
from utils import *
from loss.ND_Crossentropy import CrossentropyND, TopKLoss
from loss.focalloss import *
from loss.label_smoothing import LabelSmoothingCrossEntropy
from utils import gpu_checking
from collections import defaultdict
import wandb

import warnings
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from Model.SPDNet import SPDNetwork, SPDNetwork_1, SPDNetwork_2

warnings.filterwarnings('ignore')

class TrainMaker:
    def __init__(self, args, data1, data2, test, DA=False):
        self.args = args
        # 4. Initialize Model
        self.model_1 = SPDNetwork_1()
        self.model_2 = SPDNetwork_2()
        self.data1 = data1
        self.data2 = data2
        self.test = test

        self.history = defaultdict(list)
        self.history_mini_batch = defaultdict(list)
        self.optimizer = getattr(torch.optim, self.args.optimizer)
        self.channel_num = self.args.channel_num
        self.device = gpu_checking(args)

        if DA == False:
            self.writer = SummaryWriter(log_dir=f'./runs/{self.args.test_subj}/lr{self.args.lr}_wd{self.args.wd}')
            # self.optimizer = self.optimizer(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.wd)
            self.epoch = self.args.epoch
            self.lr = self.args.lr
            self.wd = self.args.wd

        self.scheduler = self.__set_scheduler(args, self.optimizer)
        self.criterion = self.__set_criterion(self.args.criterion)   

    def training(self, shuffle=True, interval=1000):
        if self.args.standard == 'loss': prev_v = 1000
        else : prev_v = -10

        # sampler = torch.utils.data.WeightedRandomSampler(self.data.in_weights, replacement=True, num_samples=self.data.x.shape[0])
        # data_loader = DataLoader(self.data, batch_size=self.args.batch_size, sampler=sampler)
        # data_loader = DataLoader(self.data, batch_size=self.args.batch_size)
        
        self.data1_x = self.data1.x; self.data1_y = self.data1.y
        self.data2_x = self.data2.x; self.data2_y = self.data2.y
        self.data_test_x = self.test.x; self.data_test_y = self.test.y
        
        old_loss = 0
        lr, lr_1, lr_2 = 0.1, 0.1, 0.1
        train_accuracy_1, train_accuracy_2, test_accuracy_2 = 0, 0, 0

        for e in tqdm(range(self.epoch)):
            epoch_loss = 0

            pred_label_acc = None; true_label_acc = None
            pred_list = None; true_label = None

            output_1, feat_1 = self.model_1(self.data1_x)
            output_2, feat_2 = self.model_2(self.data2_x)

            feat_1_positive, feat_1_negative = split_class_feat(feat_1, self.data1_y)
            feat_2_positive, feat_2_negative = split_class_feat(feat_2, self.data2_y)

            mmd = MMD('rbf', kernel_mul=2.0)
            loss = F.nll_loss(output_1, self.data1_y.long()) + F.nll_loss(output_2, self.data2_y.long()) + \
                mmd.forward(feat_1_positive, feat_2_positive) + \
                mmd.forward(feat_1_negative, feat_2_negative)
            loss.backward()
            
            self.model_1.update_manifold_reduction_layer(lr_1)
            self.model_2.update_manifold_reduction_layer(lr_2)

            average_grad = (self.model_1.fc_w.grad.data + self.model_2.fc_w.grad.data) / 2

            self.model_1.update_federated_layer(lr, average_grad)
            self.model_2.update_federated_layer(lr, average_grad)

            if e % 1 == 0:
                pred_1 = output_1.data.max(1, keepdim=True)[1]
                pred_2 = output_2.data.max(1, keepdim=True)[1]
                train_accuracy_1 = pred_1.eq(self.data1_y.data.view_as(pred_1)).long().cpu().sum().float() / pred_1.shape[0]
                train_accuracy_2 = pred_2.eq(self.data2_y.data.view_as(pred_2)).long().cpu().sum().float() / pred_2.shape[0]
                print('Iteration {}: Trainning Accuracy for Source Task Model: {:.4f} / Target Task Model: {:.4f}'.format(
                    e,
                    train_accuracy_1,
                    train_accuracy_2))

                logits_2, _ = self.model_2(self.data_test_x)
                output_2 = F.log_softmax(logits_2, dim=-1)
                loss_2 = F.nll_loss(output_2, self.data_test_y)
                pred_2 = output_2.data.max(1, keepdim=True)[1]
                test_accuracy_2 = pred_2.eq(self.data_test_y.data.view_as(pred_2)).long().cpu().sum().float() / pred_2.shape[0]
                print('Testing Accuracy for Model 2: {:.4f}'.format(test_accuracy_2))

            if np.abs(loss.item() - old_loss) < 1e-4:break
            old_loss = loss.item()

            if e % 50 == 0:
                lr = max(0.98 * lr, 0.01)
                lr_1 = max(0.98 * lr_1, 0.01)
                lr_2 = max(0.98 * lr_2, 0.01)

        return test_accuracy_2
        # return f1_v, acc_v, cm_v, loss_v

    def evaluation(self, data, interval=1000, state=None):
        flag = list(self.model._modules)[-1] #################
        final_layer = self.model._modules.get(flag)
        activated_features = FeatureExtractor(final_layer) #############################

        data_loader = DataLoader(data, batch_size=self.args.batch_size)

        with torch.no_grad(): # gradient 안함
            self.model.eval() # dropout은 training일 때만, evaluation으로 하면 dropout 해제
            
            pred_label_acc = None
            true_label_acc = None
            pred_list = None
            true_label = None
            valid_loss = 0
            
            test_embeds = None
            for idx, data in enumerate(data_loader):
                x, y, subj = data
                true_label = y.numpy()
                b = x.shape[0]

                x = x.reshape(b, 1, self.channel_num, -1)
                pred = self.model(x.to(device=self.device).float(), subj.to(device=self.device)); pred = pred.reshape(b, -1) # [256, 200, 4]
                # pred = self.model(x.to(device=self.device).float())
                
                loss = self.criterion(pred, y.flatten().long().to(device=self.device)) # pred.shape
                valid_loss += loss
                # if (idx+1) % interval == 0: print('[Epoch, Step({}/{})] Valid Loss:{:.4f}'.format(idx+1, len(data)//self.args.batch_size, loss / (idx +1)))
                # pred_prob = F.softmax(pred, dim=-1)
                pred_label = torch.argmax(pred, dim = -1).cpu().numpy()
                # print(pred_prob)
                if pred_label_acc is None:
                    pred_label_acc = pred_label
                    true_label_acc = true_label
                else:
                    pred_label_acc = np.concatenate((pred_label_acc, pred_label), axis=None)
                    true_label_acc = np.concatenate((true_label_acc, true_label), axis=None)

                #---# for t-sne #---#
                embeds = activated_features.features_before # [256, 200, 1, 4]; embeds =  embeds.squeeze(2) # [256, 200, 4]
                if test_embeds == None: test_embeds = embeds
                else : test_embeds = torch.cat((test_embeds, embeds), dim=0) # [103, 800]
            
                self.cal = Calculate()
                log = self.cal.calculator(metrics=self.args.metrics, loss=loss, y_true=y, y_pred=pred_label, acc_count=True)
                
                # Record history per mini-batch
                if state == None:
                    self.record_history(log, self.history_mini_batch, phase='val')
            # valid_loss = valid_loss / (idx+1)            
            # pred_list = np.concatenate(pred_list)
            # true_label = np.concatenate(true_label)
            # print(true_label_acc)
            # print(pred_label_acc)
            f1 = f1_score(true_label_acc, pred_label_acc, average='macro')
            acc = accuracy_score(true_label_acc, pred_label_acc)
            cm = confusion_matrix(true_label_acc, pred_label_acc)
            if not self.args.mode == "test" and state == None:
                print('\nEpoch Validation, f1:{:.4f}, acc:{:.4f}, Loss:{:.4f}'.format(f1, acc, valid_loss))
            elif self.args.mode == "test":
                print('\nEpoch Test, f1:{:.4f}, acc:{:.4f}'.format(f1, acc))
            # print(cm)
            
            if self.args.mode == "test":
                # savefeature
                current_time = get_time()
                create_folder(f"./features_{self.args.model}")
                np.savez(f"./features_{self.args.model}/original_subj{str(self.args.test_subj).zfill(2)}", test_embeds, true_label_acc)
        if state == None:
            print('{}____{}'.format(acc.item(), valid_loss.item()))
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


def transfer_SPD(cov_data_1, cov_data_2, labels_1, labels_2):
    """
    Train the proposed Federated Transfer Learning model over two participants.
    :param cov_data_1: data of participant 1
    :param cov_data_2: data of participant 2
    :param labels_1: labels of participant 1
    :param labels_2: labels of participant 2
    :return: The final test accuracy of participant2, which is the target domain of the federated transfer learning.
    """

    np.random.seed(0)

    # 1. Shuffle data
    cov_data_1, labels_1 = shuffle_data(cov_data_1, labels_1)
    cov_data_2, labels_2 = shuffle_data(cov_data_2, labels_2)
    print(cov_data_1.shape, cov_data_2.shape)

    # 2. Train test split
    train_data_1_num = cov_data_1.shape[0]
    cov_data_train_1 = cov_data_1[0:cov_data_1.shape[0], :, :]
    train_data_2_num = int(np.floor(cov_data_2.shape[0] * 0.8))
    cov_data_train_2 = cov_data_2[0:train_data_2_num, :, :]
    cov_data_test_2 = cov_data_2[train_data_2_num:cov_data_2.shape[0], :, :]
    print('split_num_for_test: ', train_data_2_num)
    print('rest_num_for_test: ', labels_2.shape[0] - train_data_2_num)
    print('-------------------------------------------------------')

    # 3. Convert training data to torch variables.
    data_train_1 = Variable(torch.from_numpy(cov_data_train_1)).double()
    data_train_2 = Variable(torch.from_numpy(cov_data_train_2)).double()
    data_test_2 = Variable(torch.from_numpy(cov_data_test_2)).double()

    target_train_1 = Variable(torch.LongTensor(labels_1[0:train_data_1_num]))
    target_train_2 = Variable(torch.LongTensor(labels_2[0:train_data_2_num]))
    target_test_2 = Variable(torch.LongTensor(labels_2[train_data_2_num:labels_2.shape[0]]))

    # 4. Initialize Model
    model_1 = SPDNetwork_1()
    model_2 = SPDNetwork_2()

    # Start training
    old_loss = 0
    lr, lr_1, lr_2 = 0.1, 0.1, 0.1
    train_accuracy_1, train_accuracy_2, test_accuracy_2 = 0, 0, 0
    for iteration in range(200):

        output_1, feat_1 = model_1(data_train_1)
        output_2, feat_2 = model_2(data_train_2)

        # 1. Index of positive/negative labels
        feat_1_positive, feat_1_negative = split_class_feat(feat_1, target_train_1)
        feat_2_positive, feat_2_negative = split_class_feat(feat_2, target_train_2)

        # 2. MMD knowledge transfer via MMD loss
        mmd = MMD('rbf', kernel_mul=2.0)

        loss = F.nll_loss(output_1, target_train_1) + F.nll_loss(output_2, target_train_2) + \
               1 * mmd.forward(feat_1_positive, feat_2_positive) + \
               1 * mmd.forward(feat_1_negative, feat_2_negative)
        loss.backward()

        # 3. Update local model components.
        model_1.update_manifold_reduction_layer(lr_1)
        model_2.update_manifold_reduction_layer(lr_2)

        # 4. Compute the average of global component.
        average_grad = (model_1.fc_w.grad.data + model_2.fc_w.grad.data) / 2

        # 5. Update local model of each participant.
        model_1.update_federated_layer(lr, average_grad)
        model_2.update_federated_layer(lr, average_grad)

        # 6. Evaluate model performance
        if iteration % 1 == 0:
            # Accuracy of two models
            pred_1 = output_1.data.max(1, keepdim=True)[1]
            pred_2 = output_2.data.max(1, keepdim=True)[1]
            train_accuracy_1 = pred_1.eq(target_train_1.data.view_as(pred_1)).long().cpu().sum().float() / pred_1.shape[
                0]
            train_accuracy_2 = pred_2.eq(target_train_2.data.view_as(pred_2)).long().cpu().sum().float() / pred_2.shape[
                0]
            print('Iteration {}: Trainning Accuracy for Source Task Model: {:.4f} / Target Task Model: {:.4f}'.format(
                iteration,
                train_accuracy_1,
                train_accuracy_2))

            logits_2, _ = model_2(data_test_2)
            output_2 = F.log_softmax(logits_2, dim=-1)
            loss_2 = F.nll_loss(output_2, target_test_2)
            pred_2 = output_2.data.max(1, keepdim=True)[1]
            test_accuracy_2 = pred_2.eq(target_test_2.data.view_as(pred_2)).long().cpu().sum().float() / pred_2.shape[0]
            print('Testing Accuracy for Model 2: {:.4f}'.format(test_accuracy_2))

        # 7. Check stopping criteria
        if np.abs(loss.item() - old_loss) < 1e-4:
            break
        old_loss = loss.item()

        # 8. Update learning rates
        if iteration % 50 == 0:
            lr = max(0.98 * lr, 0.01)
            lr_1 = max(0.98 * lr_1, 0.01)
            lr_2 = max(0.98 * lr_2, 0.01)

    return test_accuracy_2


def load_data(data_file, label_file, good_subjects, bad_subject):
    """
    Load data training data
    :param data_file: training samples of all subjects
    :param label_file: labels of training samples of all subjects
    :return: data and labels of the good subjects as well as one specific bad subject.
    """
    data = np.load(data_file)
    label = np.load(label_file)

    # Good Subjects
    good_subj_data = np.concatenate(data[good_subjects], axis=0)
    good_subj_label = np.concatenate(label[good_subjects], axis=0)

    # Bad Subject
    bad_subj_data = data[bad_subject]
    bad_subj_label = label[bad_subject]

    return good_subj_data, good_subj_label, bad_subj_data, bad_subj_label


def split_class_feat(feat, target):
    """
    Split the features according to the true label of the training samples. This is meant to apply MMD of the
    features of each class.
    :param feat: features
    :param target: targets/ labels
    :return: features of positive calss and features of negative class
    """
    positive_index, negative_index = np.array(target) == 1, np.array(target) == 0
    positive_feat = feat[positive_index].detach().numpy()
    negative_feat = feat[negative_index].detach().numpy()
    # Convert to Variable for further training.
    positive_feat = Variable(torch.from_numpy(positive_feat)).double()
    negative_feat = Variable(torch.from_numpy(negative_feat)).double()
    return positive_feat, negative_feat


def shuffle_data(x, y):
    """
    Shuffle the data and labels.
    :param x: data
    :param y: targets
    :return: shuffled data adn labels
    """
    idx = np.random.permutation(x.shape[0])
    return x[idx, :, :], y[idx]


if __name__ == '__main__':

    np.random.seed(0)

    GOOD_SUBJECT_IDS = [0, 1, 6, 7, 14, 28, 30, 32, 33, 34, 41, 47, 51, 53, 54, 55, 59, 61, 69, 70, 71, 72,
                        79, 84, 85, 92, 99, 103]

    # Train a model using federated transfer learning to boost the performance of one bad subject.
    # bad_subject_index = [2, 8, 16, 17, 22, 23, 27, 35, 37, 38, 39, 40, 44, 46, 57, 62, 63, 66, 73, 75, 76, 77, 89,
    # 95, 96, 98, 100, 101]
    BAD_SUBJECT_ID = 100  # Select a bad subject ID here.

    # Load data of good subjects and bad subjects.
    good_subj_data, good_subj_label, bad_subj_data, bad_subj_label = \
        load_data('raw_data/normalized_original_train_sample.npy', 'raw_data/train_label.npy',
                  GOOD_SUBJECT_IDS, BAD_SUBJECT_ID)

    accuracy_recorder = []
    for _ in range(10):
        # Conduct federated transfer learning over good and bad subjects.
        accuracy = transfer_SPD(good_subj_data, bad_subj_data, good_subj_label, bad_subj_label)
        accuracy_recorder.append(accuracy)

    print('All Accuracy: ', accuracy_recorder)
    print('SPD Riemannian Average Classification Accuracy: {:4f}.'.format(np.array(accuracy_recorder).mean()))


##################################################################################################
# Compute MMD distance using pytorch
# Author：Ce Ju, Dashan Gao
# Date  : July 29, 2020
# Paper : Ce Ju et al., Federated Transfer Learning for EEG Signal Classification, IEEE EMBS 2020.
##################################################################################################

import torch
import torch.nn as nn


class MMD(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5):
        super(MMD, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0 - total1) ** 2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            with torch.no_grad():
                XX = torch.mean(kernels[:batch_size, :batch_size])
                YY = torch.mean(kernels[batch_size:, batch_size:])
                XY = torch.mean(kernels[:batch_size, batch_size:])
                YX = torch.mean(kernels[batch_size:, :batch_size])
                loss = torch.mean(XX + YY - XY - YX)
            torch.cuda.empty_cache()
            return loss

    def backward():
        pass