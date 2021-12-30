from Model.DeepConvNet_models import *
from Model.EEGNet_models import *
from utils import *
import pickle
import importlib
import re

class ModelMaker:
    def __init__(self, args_class=None):
        print("\nModel setting...")
        self.args_class = args_class
        args = args_class.args
        self.flag = 0 
        # create_folder(args.save_path) # 필요없음
    
        self.device = gpu_checking()
        self.model = self.__build_model(args)
    
    def __build_model(self, args):
        model = ''
        if args.mode == 'train':
            if self.flag == 0:
                self.args_class.set_save_path() # setting save_path 
                self.flag == 1
            if args.model == 'DeepConvNet':
                model = DeepConvNet(args.class_num, args.channel_num, args.one_bundle).to(device = self.device) 
            elif args.model == 'ShallowConvNet':
                model = ShallowConvNet_dk(args.class_num, args.channel_num).to(device = self.device)
            elif args.model == 'EEGNet':
                model = EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device = self.device)
            
            write_pickle(os.path.join(args.save_path, "model.pk"), model)
        else:
            self.args_class.get_load_path()
            print(f"{args.load_path}에서 pretrained_model을 부르옵니다아ㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏㅏ")           
            model = pretrained_model(args.load_path) #load_path
        return model

def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def read_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def pretrained_model(save_path):
    # print(f"S{args['subject']:02} is loaded.")
    try:
        model = read_pickle(os.path.join(save_path, 'model.pk'))
    except FileNotFoundError:
        raise FileNotFoundError
    save_path, model = set_pretrained_path(save_path, model)
    # model = load_model(model, save_path)
    return model

def sort(array):
    '''
    sort exactly for list or array which element is string
    example: [1, 10, 2, 4] -> [1, 2, 4, 10]
    '''
    str2int = lambda string: int(string) if string.isdigit() else string
    key = lambda key: [str2int(x) for x in re.split("([0-9]+)", key)]
    return sorted(array, key=key)

def listdir_sort(path):
    return sort(os.listdir(path))

def set_pretrained_path(path, model):
    if not path.endswith('.tar'):
        tar = listdir_sort(os.path.join(path, 'checkpoints'))[-1]
        path = os.path.join(path, 'checkpoints', tar)
        checkpoint = torch.load(path, map_location='cpu')
        
        new_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(new_state_dict)
    return path, model

# def load_model(model, path, load_range='all'):
#     checkpoint = torch.load(path, map_location='cpu')
#     if next(iter(checkpoint.keys())).startswith('model_state_dict'):
#         print("==================================================kdy")
#         new_state_dict = dict()
#         for k, v in checkpoint['model_state_dict'].items():
#             print(k, v)
#             new_key = k[7:]
#             new_state_dict[new_key] = v
#         model.load_state_dict(new_state_dict, strict=True)
#         print(model.load_state_dict(new_state_dict, strict=True))
#     raise
#     return model