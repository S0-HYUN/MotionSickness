import torch
import torch.nn.functional as F

class CroppedLoss:
    def __init__(self, loss_function):
        self.loss_function = loss_function

    def __call__(self, preds, targets):
        avg_preds = torch.mean(preds, dim=2)
        avg_preds = avg_preds.squeeze(dim=1)
        return self.loss_function(avg_preds, targets)

def train(log_interval, model, device, train_loader, optimizer,scheduler, cuda, gpuidx, epoch=1):
    criterion = torch.nn.NLLLoss()
    lossfn = CroppedLoss(criterion)
    
    model.train()
    for batch_idx, datas in enumerate(train_loader):
        data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64)

        optimizer.zero_grad()

        data = data.reshape(data.shape[0], 1, 22, 1000)
        output = model(data) # [8,4,467]
        
        loss = lossfn(output,target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    scheduler.step()

def eval(model, device, test_loader):
    model.eval()
    test_loss = []
    correct = []

    with torch.no_grad():
        for datas in test_loader:
            data, target = datas[0].to(device), datas[1].to(device, dtype=torch.int64) # [8,22,1125]
            outputs = []
            # data = data.reshape(data.shape[0], 1, 22, 1000) ##
            for i in range(2):
                d = data[:,:,i*125:i*125 + 1000]
                d = d.reshape(d.shape[0], 1, 22, 1000)
                outputs.append(model(d)) # outputs[0] / [1] -> [8,4,467]

            result = torch.cat([outputs[0],outputs[1][:,:,model.out_size-125:model.out_size]],dim=2) # [8,4,592] #out_size=467
            y_preds_per_trial = result.mean(dim=2) # [8,4]

            # import numpy as np
            # result = outputs[0] + outputs[1]
            # result = torch.divide(result, 2)
            # y_preds_per_trial = model(data)

            test_loss.append(F.nll_loss(y_preds_per_trial, target, reduction='sum').item()) # sum up batch loss
            pred = y_preds_per_trial.argmax(dim=1,keepdim=True)# get the index of the max log-probability
            correct.append(pred.eq(target.view_as(pred)).sum().item())

    loss = sum(test_loss)/len(test_loader.dataset)
    #print('{:.0f}'.format(100. * correct / len(test_loader.dataset)))

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
        loss, sum(correct), len(test_loader.dataset),
        100. * sum(correct) / len(test_loader.dataset)))


    return loss, 100. * sum(correct) / len(test_loader.dataset)

