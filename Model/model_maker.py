from Model.DeepConvNet_models import *
from Model.EEGNet_models import *
from utils import *
import pickle
import importlib

class ModelMaker:
    def __init__(self, args):
        print("\nModel setting...")
        self.device = gpu_checking()
        self.model = args.__build_model(args)

    def __build_model(self, args):
        if args.mode == 'train':
            if args.model == 'DeepConvNet':
                model = DeepConvNet(args.class_num, args.channel_num, args.one_bundle).to(device = self.device) 
            elif args.model == 'ShallowConvNet':
                model = ShallowConvNet_dk(args.class_num, args.channel_num).to(device = self.device)
            elif args.model == 'EEGNet':
                model = EEGNet(args.class_num, args.channel_num, args.one_bundle).to(device = self.device)

            model = import_model(args.model, args.cfg)
            write_pickle(os.path.join(args.save_path, "model.pk"), model)
        else:
            model = pretrained_model(args.load_path)
        return model


def import_model(model_name, config):
    module = importlib.import_module(f'models.{model_name}_model')
    model = getattr(module, config.name)(**config)
    return model

def write_pickle(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        
def read_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def pretrained_model(save_path):
    args = get_args(save_path)
    print(f"S{args['subject']:02} is loaded.")
    try:
        model = read_pickle(os.path.join(save_path, 'model.pk'))
    except FileNotFoundError:
        raise FileNotFoundError
    save_path = set_pretrained_path(save_path)
    model = load_model(model, save_path)
    return model

# def sort(array):
#     '''
#     sort exactly for list or array which element is string
#     example: [1, 10, 2, 4] -> [1, 2, 4, 10]
#     '''
#     str2int = lambda string: int(string) if string.isdigit() else string
#     key = lambda key: [str2int(x) for x in re.split("([0-9]+)", key)]
#     return sorted(array, key=key)

# def listdir_sort(path):
#     return sort(os.listdir(path))

def set_pretrained_path(path):
    if not path.endswith('.tar'):
        tar = listdir_sort(os.path.join(path, 'checkpoints'))[-1]
        path = os.path.join(path, 'checkpoints', tar)
    return path

# def load_model(model, path, load_range='all'):
#     checkpoint = torch.load(path, map_location='cpu')
#     if next(iter(checkpoint['model_state_dict'].keys())).startswith('module'):
#         new_state_dict = dict()
#         for k, v in checkpoint['model_state_dict'].items():
#             new_key = k[7:]
#             new_state_dict[new_key] = v
#         model.load_state_dict(new_state_dict, strict=True)

# def get_args(save_path):
#     args = read_json(os.path.join(save_path, "args.json"))
#     return args


### active learning 예시

from modAL.models import ActiveLearner
from sklearn.ensemble import RandomForestClassifier

# initializing the learner
learner = ActiveLearner(
    estimator=RandomForestClassifier(),
    query_strategy = entropy_sampling, # random_sampling
    X_training=X_training, y_training=y_training
)

# query for labels
query_idx, query_inst = learner.query(X_pool)

# ...obtaining new labels from the Oracle...

# supply label for queried instance
learner.teach(X_pool[query_idx], y_new)
