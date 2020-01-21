
import argparse
import numpy as np 
import pandas as pd
from simplenet import *

parser = argparse.ArgumentParser(description='Run PyTorch ANN with .csv file')
parser.add_argument('input_file', help='Path of .csv data file')
parser.add_argument('x_columns', type=int, help="No. of columns corresponding to inputs")
parser.add_argument('y_columns', type=int, help="No. of columns corresponding to outputs")
parser.add_argument('-l', '--layers', nargs="+", type=int, help="No. of hidden layers")
parser.add_argument('-a', '--activation', nargs="+", default='elu', help="Activation function for hidden layers")
parser.add_argument('-d', '--dropout', default=0.0, type=float, help="Training dropout probability")
parser.add_argument('-n', '--norm', default=False, type=bool, help="Use batch normalization?")
parser.add_argument('-r', '--learnrate', default=0.01, type=float, help="Learning rate for optimizer")
parser.add_argument('-b', '--batches', default=20, type=int, help="No. of batches (for testing)")
parser.add_argument('-e', '--epochs', default=30, type=int, help="Epochs per batch")
parser.add_argument('-t', '--trainratio', default=0.7, type=float, help="Training set ratio")

args = parser.parse_args()

checkerr(args.input_file[-4:] == ".csv", "Please use .csv file")
df = pd.read_csv(args.input_file)
n_in = args.x_columns
n_out = args.y_columns

checkerr(df.shape[1] == n_in+n_out, "Inputs + outputs size do not match .csv file")
x_vals = df.iloc[:,:n_in].values
y_vals = df.iloc[:,n_in:].values

T = Trainer(x_vals, y_vals, train_ratio=args.trainratio, test_inc_train=True)

if args.layers is None:
    layers = [n_in * 8, (n_in*2+n_out*2) ,n_out*3]
else:
    layers = args.layers

if args.activation is None:
    af = 'elu'
else:
    af = args.activation

dp = args.dropout
norm = args.norm
lr = args.learnrate

T.set_model(layers, af=af, lr=lr, dropout=dp, norm=norm)
T.train_plot_model(args.batches,args.epochs)
T.model.summary()
T.save_model()

print("Testing jacobian")
y, jac = T(x_vals[0:1,:], jac=True)
print(jac)