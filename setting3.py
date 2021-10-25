from sys import stderr
from scipy.signal.windows.windows import exponential
from torch.optim import optimizer
import torch
import data_loader
import trainer
import os
import pandas as pd
import Network.EEGNet_models
import Network.DeepConvNet_models
import argparse
import subprocess