import numpy as np
import pandas as pd
from statistics import mean
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from Model.model_maker import ModelMaker
# add
import data_loader.data_loader_5fold
from get_args import Args
from trainer import TrainMaker
from sklearn.metrics import f1_score
import wandb
import time
 
if __name__ == '__main__': 
    # wandb.init(project="my-test-project", entity="sohyun")

    args_class = Args()
    args = args_class.args
    
    data = data_loader.data_loader_5fold.Dataset(args, phase="train")
    data_pool = data_loader.data_loader_5fold.Dataset(args, phase="pool")        
    data_valid = data_loader.data_loader_5fold.Dataset(args, phase="valid")
    data_test = data_loader.data_loader_5fold.Dataset(args, phase="test")

    # time
    tm = time.localtime(time.time())
    string = time.strftime('%Y%m%d_%H%M%S', tm)
    
    # Save a file
    df = pd.DataFrame(columns = ['test_subj', 'lr', 'wd', 'acc', 'f1', 'loss']); idx = 0

    ac1, ac2 = [], []  # arrays to store accuracy of different models
    acc = []

    budget = int(data_pool.x.shape[0]/20)
    print("budget", budget)

    for i in range(151):
        print(f"==={i}===")
        print(data.x.shape, data_pool.x.shape, data_valid.x.shape, data_test.x.shape)

        args.mode = "train"
        model = ModelMaker(args_class).model
        trainer = TrainMaker(args, model, data, data_valid)
        # fitting
        trainer.training()
        
        # args.mode = "test"
        # model_pool = ModelMaker(args_class).model # 방금 train한걸로 pseudo label을 뽑아.
        # trainer_pool = TrainMaker(args, model_pool, data, data_valid)
        print(data_pool.x.shape, "===============")
        max_idx, pseudo_labeling = trainer.predict_proba(data_pool.x, random=True) # mcdo=True
        uncrt_pt_ind = max_idx

        data.x = np.append(data_pool.x[uncrt_pt_ind, :], data.x, axis = 0)
        # data.y = np.append(data_pool.y[uncrt_pt_ind], data.y)
        data.y = np.append(pseudo_labeling, data.y) # 방금 얻은 걸 집어넣어
        data_pool.x = np.delete(data_pool.x, uncrt_pt_ind, axis = 0)
        data_pool.y = np.delete(data_pool.y, uncrt_pt_ind)
        # print(unlabel.shape)


        args.mode = "test"
        model_test = ModelMaker(args_class).model # 방금 train한걸로 pseudo label을 뽑아.
        trainer_test = TrainMaker(args, model_test, data_test)
        f1_v, acc_v, cm_v, loss_v = trainer.evaluation(data_test)

        # score = f1_score(data.y, trainer.pseudo_label(data), average='macro')
        acc.append(acc_v)

        df.loc[idx] = [args.test_subj, args.lr, args.wd, acc_v, f1_v, loss_v.cpu().numpy()]
        df.to_csv('./csvs/{}_active_Random_results_{}_subj{}_choose.csv'.format(string, args.model, args.test_subj), header = True, index = False)
        idx += 1
    
    # figure
    from matplotlib import pyplot as plt
    plt.plot(np.arange(len(acc)), acc)
    plt.savefig(f"./{string}_eegnet.png")

    # print("Accuracy by active model :", mean(ac1)*100)
    # print("Accuracy by random sampling :", mean(ac2)*100)
