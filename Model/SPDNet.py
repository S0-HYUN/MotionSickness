##################################################################################################
# SPDNet model definition
# Author：Ce Ju, Dashan Gao
# Date  : July 29, 2020
# Paper : Ce Ju et al., Federated Transfer Learning for EEG Signal Classification, IEEE EMBS 2020.
# Description: Source domain includes all good subjects, target domain is the bad subject.
##################################################################################################

import torch
from torch.autograd import Variable
import Model.SPDNet_utils
import torch.nn.functional as F

torch.manual_seed(0)


class SPDNetwork_1(torch.nn.Module):
    """
    A sub-class of SPDNetwork with network structure of manifold reduction layers: [(32, 32), (32, 16), (16, 4)]
    """

    def __init__(self):
        super(SPDNetwork_1, self).__init__()

        # self.w_1_p = Variable(torch.randn(32, 32).double(), requires_grad=True)
        # self.w_2_p = Variable(torch.randn(32, 16).double(), requires_grad=True)
        # self.w_3_p = Variable(torch.randn(16, 4).double(), requires_grad=True)
        # self.fc_w = Variable(torch.randn(16, 2).double(), requires_grad=True)

        self.w_1_p = Variable(torch.randn(28, 28).double(), requires_grad=True)
        self.w_2_p = Variable(torch.randn(28, 14).double(), requires_grad=True)
        self.w_3_p = Variable(torch.randn(14, 3).double(), requires_grad=True)
        self.fc_w = Variable(torch.randn(9, 3).double(), requires_grad=True)

    def forward(self, input):
        """
        Forward propagation
        :param input:
        :return:
                output: the predicted probability of the model.
                feat: feature in the common subspace for feature alignment.
        """
        batch_size = input.shape[0]

        output = input
        # Forward propagation of local model
        for idx, w in enumerate([self.w_1_p, self.w_2_p]): #w=(28,28)
            w = w.contiguous().view(1, w.shape[0], w.shape[1])
            w_tX = torch.matmul(torch.transpose(w, dim0=1, dim1=2), output) #w=(28,28)
            w_tXw = torch.matmul(w_tX, w) #(1,28,28)
            output = torch.tensor(RecFunction.apply(w_tXw)) #(1,14,14)

        w_3 = self.w_3_p.contiguous().view([1, self.w_3_p.shape[0], self.w_3_p.shape[1]]); print(w_3.shape) #(1,14,4)
        w_tX = torch.matmul(torch.transpose(w_3, dim0=1, dim1=2), output); print("w_tX", w_tX.shape) #(1,14,14)
        w_tXw = torch.matmul(w_tX, w_3); print("w_tXw", w_tXw.shape) #(3,3)
        X_3 = LogFunction.apply(w_tXw)

        # feat = X_3.view([w_tXw.shape[0]*w_tXw.shape[1], -1])  # [batch_size, d]
        feat = X_3.view([batch_size, -1])#(3,3)->9
        logits = torch.matmul(feat, self.fc_w)  # [batch_size, num_class] (28,3)
        output = F.log_softmax(logits, dim=-1)
        return output, feat

    def update_all_layers(self, lr):
        """
        Update all layers for local single party training.
        :param lr: learning rate
        :return: None
        """
        update_manifold_reduction_layer(lr, [self.w_1_p, self.w_2_p, self.w_3_p])
        self.fc_w.data -= lr * self.fc_w.grad.data
        self.fc_w.grad.data.zero_()

    def update_manifold_reduction_layer(self, lr):
        """
        Update the manifold reduction layers
        :param lr: learning rate
        :return: None
        """
        update_manifold_reduction_layer(lr, [self.w_1_p, self.w_2_p, self.w_3_p])

    def update_federated_layer(self, lr, average_grad):
        """
        Update the federated layer.
        :param lr: Learning rate
        :param average_grad: the average gradient of the federated layer of all participants
        :return: None
        """
        self.fc_w.data -= lr * average_grad
        self.fc_w.grad.data.zero_()


class SPDNetwork_2(torch.nn.Module):
    """
    A sub-class of SPDNetwork with network structure of manifold reduction layers: [(32, 4), (4, 4), (4, 4)]
    """
    def __init__(self):
        super(SPDNetwork_2, self).__init__()

        self.w_1_p = Variable(torch.randn(28, 14).double(), requires_grad=True)
        self.w_2_p = Variable(torch.randn(14, 14).double(), requires_grad=True)
        self.w_3_p = Variable(torch.randn(14, 3).double(), requires_grad=True)
        self.fc_w = Variable(torch.randn(9, 3).double(), requires_grad=True)

        # self.w_1_p = Variable(torch.randn(32, 4).double(), requires_grad=True)
        # self.w_2_p = Variable(torch.randn(4, 4).double(), requires_grad=True)
        # self.w_3_p = Variable(torch.randn(4, 4).double(), requires_grad=True)
        # self.fc_w = Variable(torch.randn(16, 2).double(), requires_grad=True)

    def forward(self, input):
        """
        Forward propagation
        :param input:
        :return:
                output: the predicted probability of the model.
                feat: feature in the common subspace for feature alignment.
        """
        batch_size = input.shape[0]
        output = input
        # Forward propagation of local model
        for idx, w in enumerate([self.w_1_p, self.w_2_p]):
            w = w.contiguous().view(1, w.shape[0], w.shape[1])
            w_tX = torch.matmul(torch.transpose(w, dim0=1, dim1=2), output)
            w_tXw = torch.matmul(w_tX, w)
            output = torch.tensor(RecFunction.apply(w_tXw)) #(1,16,16)

        w_3 = self.w_3_p.contiguous().view([1, self.w_3_p.shape[0], self.w_3_p.shape[1]])
        w_tX = torch.matmul(torch.transpose(w_3, dim0=1, dim1=2), output)
        w_tXw = torch.matmul(w_tX, w_3)
        X_3 = LogFunction.apply(w_tXw)

        # feat = X_3.view([w_tXw.shape[0]*w_tXw.shape[1], -1])  # [batch_size, d]
        feat = X_3.view([batch_size, -1]); print(feat.shape)
        logits = torch.matmul(feat, self.fc_w)  # [batch_size, num_class]
        output = F.log_softmax(logits, dim=-1)
        return output, feat

    def update_all_layers(self, lr):
        """
        Update all layers for local single party training.
        :param lr: learning rate
        :return: None
        """
        update_manifold_reduction_layer(lr, [self.w_1_p, self.w_2_p, self.w_3_p])
        self.fc_w.data -= lr * self.fc_w.grad.data
        self.fc_w.grad.data.zero_()

    def update_manifold_reduction_layer(self, lr):
        """
        Update the manifold reduction layers
        :param lr: learning rate
        :return: None
        """
        update_manifold_reduction_layer(lr, [self.w_1_p, self.w_2_p, self.w_3_p])

    def update_federated_layer(self, lr, average_grad):
        """
        Update the federated layer.
        :param lr: Learning rate
        :param average_grad: the average gradient of the federated layer of all participants
        :return: None
        """
        self.fc_w.data -= lr * average_grad
        self.fc_w.grad.data.zero_()


# Define the SPDNetwork the same as SPDNetwork_2 for convenience.
SPDNetwork = SPDNetwork_2


def update_manifold_reduction_layer(lr, params_list):
    """
    Update parameters of the participant-specific parameters, here are [self.w_1_p, self.w_2_p, self.w_3_p]
    :param lr: learning rate
    :param params_list: parameter list
    :return: None
    """
    for w in params_list:
        grad_w_np = w.grad.data.numpy()
        w_np = w.data.numpy()
        updated_w = Model.SPDNet_utils.update_para_riemann(w_np, grad_w_np, lr)
        w.data.copy_(torch.DoubleTensor(updated_w))
        # Manually zero the gradients after updating weights
        w.grad.data.zero_()


###
from torch.autograd import Function
class RecFunction(Function):

    def forward(self, input):
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.shape[0], input.shape[1])).double()
        max_Ss = torch.zeros_like(input)
        max_Ids = torch.zeros_like(input)
        for i in range(input.shape[0]):
            U, S, V = torch.svd(input[i, :, :])
            eps = 0.0001
            max_S = torch.clamp(S, min=eps)
            max_Id = torch.ge(S, eps).float()
            Ss[i, :] = S
            Us[i, :, :] = U
            max_Ss[i, :, :] = torch.diag(max_S)
            max_Ids[i, :, :] = torch.diag(max_Id)

        result = torch.matmul(Us, torch.matmul(max_Ss, torch.transpose(Us, 1, 2)))
        self.Us = Us
        self.Ss = Ss
        self.max_Ss = max_Ss
        self.max_Ids = max_Ids
        self.save_for_backward(input)
        return result

    def backward(self, grad_output):
        Ks = torch.zeros_like(grad_output)

        dLdC = grad_output
        dLdC = 0.5 * (dLdC + torch.transpose(dLdC, 1, 2))  # checked
        Ut = torch.transpose(self.Us, 1, 2)
        dLdV = 2 * torch.matmul(torch.matmul(dLdC, self.Us), self.max_Ss)
        dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), self.Us)
        dLdS = torch.matmul(self.max_Ids, dLdS_1)  # checked

        diag_dLdS = torch.zeros_like(grad_output)
        for i in range(grad_output.shape[0]):
            diagS = self.Ss[i, :]
            diagS = diagS.contiguous()
            vs_1 = diagS.view([diagS.shape[0], 1])
            vs_2 = diagS.view([1, diagS.shape[0]])
            K = 1.0 / (vs_1 - vs_2)
            K[K >= float("Inf")] = 0.0
            Ks[i, :, :] = K
            diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))

        tmp = torch.transpose(Ks, 1, 2) * torch.matmul(Ut, dLdV)
        tmp = 0.5 * (tmp + torch.transpose(tmp, 1, 2)) + diag_dLdS
        grad = torch.matmul(self.Us, torch.matmul(tmp, Ut))  # checked

        return grad


class LogFunction(Function):
    def forward(self, input):
        Us = torch.zeros_like(input)
        Ss = torch.zeros((input.shape[0], input.shape[1])).double()
        logSs = torch.zeros_like(input)
        invSs = torch.zeros_like(input)
        for i in range(input.shape[0]):
            U, S, V = torch.svd(input[i, :, :])
            Ss[i, :] = S
            Us[i, :, :] = U
            logSs[i, :, :] = torch.diag(torch.log(S))
            invSs[i, :, :] = torch.diag(1.0 / S)
        result = torch.matmul(Us, torch.matmul(logSs, torch.transpose(Us, 1, 2)))
        self.Us = Us
        self.Ss = Ss
        self.logSs = logSs
        self.invSs = invSs
        self.save_for_backward(input)
        return result

    def backward(self, grad_output):
        grad_output = grad_output.double()
        Ks = torch.zeros_like(grad_output)
        dLdC = grad_output
        dLdC = 0.5 * (dLdC + torch.transpose(dLdC, 1, 2))  # checked
        Ut = torch.transpose(self.Us, 1, 2)
        dLdV = 2 * torch.matmul(dLdC, torch.matmul(self.Us, self.logSs))  # [d, ind]
        dLdS_1 = torch.matmul(torch.matmul(Ut, dLdC), self.Us)  # [ind, ind]
        dLdS = torch.matmul(self.invSs, dLdS_1)
        diag_dLdS = torch.zeros_like(grad_output)
        for i in range(grad_output.shape[0]):
            diagS = self.Ss[i, :]
            diagS = diagS.contiguous()
            vs_1 = diagS.view([diagS.shape[0], 1])
            vs_2 = diagS.view([1, diagS.shape[0]])
            K = 1.0 / (vs_1 - vs_2)
            # K.masked_fill(mask_diag, 0.0)
            K[K >= float("Inf")] = 0.0
            Ks[i, :, :] = K

            diag_dLdS[i, :, :] = torch.diag(torch.diag(dLdS[i, :, :]))

        tmp = torch.transpose(Ks, 1, 2) * torch.matmul(Ut, dLdV)
        tmp = 0.5 * (tmp + torch.transpose(tmp, 1, 2)) + diag_dLdS
        grad = torch.matmul(self.Us, torch.matmul(tmp, Ut))  # checked
        return grad


def rec_mat(input):
    return RecFunction()(input)


def log_mat(input):
    return LogFunction()(input)


def update_para_riemann(X, U, t):
    Up = cal_riemann_grad(X, U)
    new_X = cal_retraction(X, Up, t)
    return new_X


def cal_riemann_grad(X, U):
    """
    Calculate Riemann gradient.
    :param X: the parameter
    :param U: the eculidean gradient
    :return: the riemann gradient
    """
    # XtU = X'*U;
    XtU = np.matmul(np.transpose(X), U)
    # symXtU = 0.5 * (XtU + XtU');
    symXtU = 0.5 * (XtU + np.transpose(XtU))
    # Up = U - X * symXtU;
    Up = U - np.matmul(X, symXtU)
    return Up


def cal_retraction(X, rU, t):
    """
    Calculate the retraction value
    :param X: the parameter
    :param rU: the riemann gradient
    :param t: the learning rate
    :return: the retraction:
    """
    # Y = X + t * U;
    # [Q, R] = qr(Y, 0);
    # Y = Q * diag(sign(diag(R)));
    Y = X - t * rU
    Q, R = np.linalg.qr(Y, mode='reduced')
    sR = np.diag(np.sign(np.diag(R)))
    Y = np.matmul(Q, sR)

    return Y