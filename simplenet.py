import torch
from torch import nn
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

def checkerr(cond, message):
    if not (cond):
        raise Exception(message)

class Trainer(object):
    af_string = {
        'none': None,
        'elu': nn.ELU,
        'hardshrink': nn.Hardshrink,
        'hardtanh': nn.Hardtanh,
        'leakyrelu': nn.LeakyReLU,
        'logsigmoid': nn.LogSigmoid,
        'prelu': nn.PReLU,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'rrelu': nn.RReLU,
        'selu': nn.SELU,
        'celu': nn.CELU,
        'sigmoid': nn.Sigmoid,
        'softplus': nn.Softplus,
        'softshrink': nn.Softshrink,
        'softsign': nn.Softsign,
        'tanh': nn.Tanh,
        'tanhshrink': nn.Tanhshrink,
        'softmin': nn.Softmin,
        'softmax': nn.Softmax,
        'logsoftmax': nn.LogSoftmax
    }
    optim_string = {
        'sgd': optim.SGD,
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
        'sparseadam': optim.SparseAdam,
        'adamax': optim.Adamax,
        'asgd': optim.ASGD,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop
    }
    loss_string = {
        'l1loss': nn.L1Loss,
        'mseloss': nn.MSELoss,
        'crossentropyloss': nn.CrossEntropyLoss,
        'ctcloss': nn.CTCLoss,
        'nllloss': nn.NLLLoss,
        'poissonnllloss': nn.PoissonNLLLoss,
        'kldivloss': nn.KLDivLoss,
        'bceloss': nn.BCELoss,
        'bcewithlogitsloss': nn.BCEWithLogitsLoss,
        'marginrankingloss': nn.MarginRankingLoss,
        'hingeembeddingloss': nn.HingeEmbeddingLoss,
        'multilabelmarginloss': nn.MultiLabelMarginLoss,
        'smoothl1loss': nn.SmoothL1Loss,
        'softmarginloss': nn.SoftMarginLoss,
        'multilabelsoftmarginloss': nn.MultiLabelSoftMarginLoss,
        'cosineembeddingloss': nn.CosineEmbeddingLoss,
        'multimarginloss': nn.MultiMarginLoss,
        'tripletmarginloss': nn.TripletMarginLoss
    }
    def __init__(self, input_data, output_data, train_ratio=0.7, test_inc_train=False):
        checkerr(isinstance(input_data, np.ndarray) and input_data.ndim == 2, "Please use a 2D-numpy array as input data for Trainer")
        checkerr(isinstance(output_data, np.ndarray) and input_data.ndim == 2, "Please use a 2D-numpy array as output data for Trainer")
        checkerr(np.size(input_data, axis=0) == np.size(output_data, axis=0), "Rows of input data should match rows of output data")

        # Input and output dimensions
        self.n_inputs = np.size(input_data, axis=1)
        self.n_outputs = np.size(output_data, axis=1)
        self.input_data = input_data
        self.output_data = output_data
        self.data_size = np.size(output_data, axis=0)

        # Getting Standard Scaling parameters and lambda functions
        self.x_mean = np.mean(input_data, axis=0)
        self.x_std = np.std(input_data, axis=0)

        self.y_mean = np.mean(output_data, axis=0)
        self.y_std = np.std(output_data, axis=0)

        self.transform_x = lambda x: (x - self.x_mean) / self.x_std
        self.invtransform_x = lambda x: x * self.x_std + self.x_mean
        self.transform_y = lambda y: (y - self.y_mean) / self.y_std
        self.invtransform_y = lambda y: y * self.y_std + self.y_mean

        # Train test split:
        alldata = np.column_stack([input_data, output_data])
        np.random.shuffle(alldata)
        train_cut = int(self.data_size * train_ratio)
        self.x_train = alldata[:train_cut, :self.n_inputs]
        self.y_train = alldata[:train_cut, self.n_inputs:]

        if test_inc_train:
            self.x_test = alldata[:, :self.n_inputs]
            self.y_test = alldata[:, self.n_inputs:]
        else:
            self.x_test = alldata[train_cut:, :self.n_inputs]
            self.y_test = alldata[train_cut:, self.n_inputs:]

        # Initialising self.model to check if model is ready
        self.model = None

    def set_model(self, layers, af='elu', opti='sgd', lr=0.01, optim_params=None, loss='mseloss', dropout=0.0, norm=False):
        if optim_params is None: optim_params = dict()

        checkerr(isinstance(layers, (tuple, list, np.ndarray)) and len(layers) >= 1, "Layers must be a list of no. of hidden nodes of size 1 or more")
        nlayers = len(layers)
        if isinstance(af, (tuple,list)):
            act_func = []
            for i in range(len(af)):
                checkerr(isinstance(af[i], str), "Use string to get the name of the activation function")
                f = Trainer.af_string[af[i].lower()]
                if f is None:
                    act_func.append(None)
                else:
                    act_func.append(f())
        else:
            checkerr(isinstance(af, str), "Use string to get the name of the activation function")
            act_func = [ Trainer.af_string[af.lower()]() ] * nlayers

        # Defining model
        self.model = SimpleNet((self.n_inputs, self.n_outputs), layers, act_func, dropout, norm)

        # Defining optimizer
        checkerr(isinstance(opti, str), "Use string to get the name of the optimizer function")
        optim_fn = Trainer.optim_string[opti.lower()]
        self.optimizer = optim_fn(self.model.parameters(), lr=lr, **optim_params)

        # Defining loss criterion
        loss_fn = Trainer.loss_string[loss.lower()]
        self.criterion = loss_fn()

        self._x = torch.from_numpy(self.transform_x(self.x_train)).float()
        self._y = torch.from_numpy(self.transform_y(self.y_train)).float()
        self._xv = torch.from_numpy(self.transform_x(self.x_test)).float()
        self._yv = torch.from_numpy(self.transform_y(self.y_test)).float()

        self.epochs = 0

    def train_model(self, epochs=1, get_ys=False, get_trainloss=False):
        checkerr(self.model is not None, "Please set model before training model")
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            y_ = self.model(self._x)
            loss = self.criterion(y_, self._y)

            loss.backward()
            self.optimizer.step()
            self.epochs += 1

        train_loss = loss.item()
        self.model.eval()
        with torch.no_grad():
            yv_ = self.model(self._xv)
            loss = self.criterion(yv_, self._yv)


        if get_trainloss and get_ys:
            return (loss.item(), train_loss, self.invtransform_y(yv_.numpy()))
        elif get_ys:
            return (loss.item(), self.invtransform_y(yv_.numpy()))
        elif get_trainloss:
            return (loss.item(), train_loss)
        return loss.item()

    def train_plot_model(self, n_batches, epochs_per_batch, print_all=True):
        checkerr(self.model is not None, "Please set model before training model")
        ydim = self.n_outputs
        if ydim > 5:
            print("Warning: outputs more than 5, only first 5 will be displayed on deviation plots")
            ydim = 5
        rows = ydim // 3 + 1
        cols = 2 if ydim == 1 else 3
        spid = rows*100+cols*10
        fig = plt.figure(figsize=(7*cols,5*rows))
        ax_loss = fig.add_subplot(spid+1)

        ax_ys = []
        for i in range(ydim):
            ax_ys.append(fig.add_subplot(spid+i+2))

        pend = "\n" if print_all else "\r"
        batch = np.array([])
        test_loss = np.array([])
        train_loss = np.array([])
        for i in range(n_batches):
            (loss, trainl, yv) = self.train_model(epochs_per_batch, get_ys=True, get_trainloss=True)
            print("Batch {:d}, Total epochs {:d}: training loss = {:0.4f}, testing loss = {:0.4f}".format(i+1, (i+1)*epochs_per_batch, loss, trainl), end=pend)
            ax_loss.cla()
            batch = np.append(batch, i+1)
            test_loss = np.append(test_loss, loss)
            train_loss = np.append(train_loss, trainl)
            ax_loss.plot(batch, train_loss, linewidth=1.5, label="training set loss")
            ax_loss.plot(batch, test_loss, linewidth=1.5, label="test set loss")

            ax_loss.set_ylabel('Reduced Loss Function')
            ax_loss.set_xlabel('No. of Batches')
            ax_loss.set_title('Loss function = test:{:0.4f}, train:{:0.4f}'.format(loss, trainl))
            ax_loss.legend()

            for yd in range(ydim):
                ax = ax_ys[yd]
                ax.cla()
                ys = self.y_test[:,yd]
                ymin = np.min(ys)
                ymax = np.max(ys)
                ax.plot(ys, yv[:,yd], 'ro', alpha=0.2)
                ax.plot([ymin, ymax], [ymin,ymax], 'k-', linewidth=1.5)

                ax.set_ylabel('Predicted Outputs')
                ax.set_xlabel('Actual Ouputs')
                ax.set_title(f'Deviation Plot for y({yd+1:d})for Test Set')
            
            plt.pause(0.1)

        plt.show(block=True)
        return

    def evaluate(self, x=None, jac=False):
        # Using current model, evaluate given x. If no x is given, evaluates all input
        checkerr(self.model is not None, "Please set model before evaluating model")
        if x is None:
            x = self.input_data
        checkerr(isinstance(x, np.ndarray) and np.size(x, axis=1) == self.n_inputs, f"Inputs must be numpy array with {self.n_inputs:d} columns")
        x_scaled = self.transform_x(x)

        if not jac:
            x_torch = torch.from_numpy(x_scaled).float()
            self.model.eval()
            with torch.no_grad():
                y = self.model(x_torch)
            return self.invtransform_y(y.numpy())

        checkerr(np.size(x, axis=0) == 1, "Inputs have to be only one row in order to evaluate jacobian")
        x_torch = torch.tensor(x_scaled, requires_grad=True).float()
        self.model.eval()
        self.model.zero_grad()
        y = self.model(x_torch)
        j = torch.zeros(self.n_outputs, self.n_inputs)
        for i in range(self.n_outputs):
            j[i, :] = torch.autograd.grad(y[:,i], x_torch, retain_graph=True)[0].data

        jacobian = (j.numpy().T * self.y_std).T / self.x_std

        return (self.invtransform_y(y.detach().numpy()), jacobian)

    def __call__(self, x=None, jac=False):
        return self.evaluate(x, jac)

    def summary(self):
        # print / return the details of the model and training 
        return

    def save_model(self, path="model.pt"):
        # saves the model
        torch.save(self.model.state_dict(), path)
        return self

    def load_model(self, path='model.pt'):
        checkerr(self.model is not None, "Please set the model before loading saved model. Set model should have the same layers and activation functions as loaded model.")
        try:
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        except:
            print("Error loading model, path not found or model doesn't correspond to data size")
        return self

    def get_params(self):
        sdict = dict()
        statedict = self.model.state_dict()
        for param_tensor in statedict:
            sdict[param_tensor] = statedict[param_tensor].numpy()
        return sdict

class SimpleNet(nn.Module):
    def __init__(self, input_output, layers, activation_func=None, dropout=0.0, norm=False):
        super().__init__()
        input_dim = input_output[0]
        output_dim = input_output[1]
        self.beta = nn.ModuleList()
        self.sig = nn.ModuleList() if activation_func is not None else None
        self.norm1d = nn.ModuleList() if norm else None
        self.drop = nn.ModuleList() if dropout > 0.0 else None
        self.dropout_p = dropout
        # while (activation_func is not None) and len(activation_func) < len(layers):
        #     activation_func.append(activation_func[-1])
        self.afskip = []
        # skipped = 0
        for i in range(len(layers)):
            if i == 0:
                self.beta.append(nn.Linear(input_dim, layers[i]))
                if self.sig is not None:
                    if activation_func[i] is None:
                        self.afskip.append(i)
                    else:
                        self.sig.append(activation_func[i])
                if self.norm1d is not None:
                    self.norm1d.append(nn.BatchNorm1d(layers[i]))
                if self.drop is not None:
                    self.drop.append(nn.Dropout(p=dropout))

            if i < len(layers)-1:
                self.beta.append(nn.Linear(layers[i], layers[i+1]))
                if self.sig is not None and len(activation_func) > i+1:
                    if activation_func[i+1] is None:
                        self.afskip.append(i+1)
                    else:
                        self.sig.append(activation_func[i+1])
                if self.norm1d is not None:
                    self.norm1d.append(nn.BatchNorm1d(layers[i+1]))
                if self.drop is not None:
                    self.drop.append(nn.Dropout(p=dropout))
                
            else:
                self.beta.append(nn.Linear(layers[i],output_dim))
        self.layers = len(self.beta)

    def forward(self, x):
        skipped = 0
        for i in range(self.layers):
            f = self.beta[i]
            skip = lambda x : x
            if i == 0:# or i == self.layers - 1:
                x = f(x)
            else:
                n = skip if self.norm1d is None else self.norm1d[i-1]
                d = skip if self.drop is None else self.drop[i-1]
                if self.sig is None or len(self.sig) <= i-1-skipped:
                    a = skip
                elif (i-1) in self.afskip:
                    a = skip
                    skipped += 1
                else:
                    a = self.sig[i-1-skipped]
                x = f(a(d(n(x))))
        return x

    def summary(self):
        s = ""
        s += "-"*70 + "\n"
        s += "{:<35s}{:<20s}{:<10s}".format("Layer", "(Input, Output)", "Params #") + "\n"
        s += "="*70 + "\n"

        print("-"*70)
        print("{:<35s}{:<20s}{:<10s}".format("Layer", "(Input, Output)", "Params #"))
        print("="*70)
        total_params = 0
        skipped = 0
        for i in range(self.layers):
            inputs = self.beta[i].in_features
            outputs = self.beta[i].out_features
            params = self.beta[i].weight.size()[0] * self.beta[i].weight.size()[1] + self.beta[i].bias.size()[0]
            s += "{:<35s}{:<20s}{:<10s}".format(f"dense_{i+1:d}", f"({inputs:d},{outputs:d})", f"{params:d}") + "\n"
            print("{:<35s}{:<20s}{:<10s}".format(f"dense_{i+1:d}", f"({inputs:d},{outputs:d})", f"{params:d}"))
            if self.norm1d is not None and i < len(self.norm1d):
                s += "{:<35s}{:<20s}{:<10s}".format(f"norm1d_{i+1:d}", f"({outputs:d},{outputs:d})", f"{0:d}") + "\n"
                print("{:<35s}{:<20s}{:<10s}".format(f"norm1d_{i+1:d}", f"({outputs:d},{outputs:d})", f"{0:d}"))
            if self.drop is not None and i < len(self.drop):
                s += "{:<35s}{:<20s}{:<10s}".format(f"dropout_{i+1:d}(p={self.dropout_p:0.2f})", f"({outputs:d},{outputs:d})", f"{0:d}") + "\n"
                print("{:<35s}{:<20s}{:<10s}".format(f"dropout_{i+1:d}(p={self.dropout_p:0.2f})", f"({outputs:d},{outputs:d})", f"{0:d}"))
            if self.sig is not None:
                if i in self.afskip:
                    skipped += 1
                elif i-skipped < len(self.sig):
                    s += "{:<35s}{:<20s}{:<10s}".format(repr(self.sig[i-skipped]).lower()+f"_{i+1:d}", f"({outputs:d},{outputs:d})", f"{0:d}") + "\n"
                    print("{:<35s}{:<20s}{:<10s}".format(repr(self.sig[i-skipped]).lower()+f"_{i+1:d}", f"({outputs:d},{outputs:d})", f"{0:d}"))
            if i < self.layers-1: 
                s += "-"*70 + "\n"
                print("-"*70)

            total_params += params
        s += "="*70 + "\n"
        s += "{:<55s}{:<10d}".format("Total no. of parameters: ", total_params) + "\n"
        s += "-"*70
        print("="*70)
        print("{:<55s}{:<10d}".format("Total no. of parameters: ", total_params))
        print("-"*70)

        return s

def main():
    # return
    test_in = np.random.rand(5000,6) * 90 + 10
    test_out = test_in[:,0] ** 0.31 * test_in[:,1]**1.31 - test_in[:,2]**0.84 * test_in[:,3]**-0.91 / test_in[:,4]**0.13 + test_in[:,5]**1.21
    test_out2 = test_in[:,0] ** 0.91 / test_in[:,2]**0.86 - test_in[:,1]**1.21 * test_in[:,4]**0.26 / test_in[:,3]**0.13 + test_in[:,0]**0.61 / test_in[:,5]**0.88
    test_out3 = test_in[:,0] ** 0.47 * test_in[:,2]**0.166 + test_in[:,1]**0.53 * test_in[:,4]**0.96 / test_in[:,3]**0.73 + test_in[:,0]**0.21 / test_in[:,5]**0.42
    test_out4 = test_in[:,0] ** 0.63 / test_in[:,2]**0.56 - test_in[:,1]**0.94 / test_in[:,4]**0.56 * test_in[:,3]**0.293 + test_in[:,0]**0.61 / test_in[:,5]**0.58
    test_out5 = test_in[:,0] ** 0.391 * test_in[:,2]**1.16 + test_in[:,1]**0.71 * test_in[:,4]**1.16 / test_in[:,3]**0.83 - test_in[:,0]**0.421 / test_in[:,5]**0.78
    test_out6 = test_in[:,0] ** 0.251 / test_in[:,2]**0.76 - test_in[:,1]**1.11 * test_in[:,4]**0.69 * test_in[:,3]**0.13 + test_in[:,0]**0.791 / test_in[:,5]**0.58
    test_out = np.column_stack([test_out, test_out2, test_out3]) * (1 + np.random.normal(scale=0.05,size=(5000,3)))
    T = Trainer(test_in, test_out, test_inc_train=True)
    T.set_model([64, 48, 32, 16], af=['none','softplus','elu'], lr=0.05, dropout=0.2, norm=True)
    T.train_plot_model(5,10)
    s = T.model.summary()

    # testdx = torch.tensor(T.transform_x(test_in[1:2,:]), requires_grad=True).float()
    # print(testdx.requires_grad)
    
    # T.model.eval()
    # # T.model.zero_grad()
    # y = T.model(testdx)
    # print(T.invtransform_y(y.detach().numpy()))
    # print(T(test_in[1:2, :]))
    # g = torch.zeros(6,6)
    # g2 = torch.zeros(6,6)
    # for i in range(6):
    #     g[:,i] = torch.autograd.grad(y[:,i], testdx, retain_graph=True)[0].data
    #     g2[:,i] = g[:,i] * T.y_std[i]
    #     for j in range(6):
    #         g2[j,i] = g2[j,i] / T.x_std[j]
    # print(g2.numpy().T)
    # jac1 = (g.numpy() * T.y_std).T / T.x_std
    # print(jac1)
    x = test_in[1:2,:]
    # xp = x.copy()
    # xm = x.copy()
    # jac = np.zeros((6,6))
    # for i in range(6):
    #     change = 0.01
    #     xp[0,i] = x[0,i] * (1+change)
    #     xm[0,i] = x[0,i] * (1-change)
    #     dx = x[0,i] * change

    #     yp = T(xp)
    #     ym = T(xm)

    #     jac[:, i] = (yp - ym)/(2*dx)

    #     xp[0,i] = x[0,i]
    #     xm[0,i] = x[0,i]

    # print()
    # print(jac)
    ty, tj = T(x, jac=True)
    print(tj)
    # print(testdx.grad)
    # params = T.get_params()
    # for param_tensor in params:
    #     print(param_tensor, "\t", params[param_tensor])

    # T2 = Trainer(test_in, test_out)
    # T2.set_model([64, 48, 32, 16], af='relu', lr=0.05, dropout=0.5, norm=True)
    # T2.load_model()

    # fig, ax = plt.subplots()
    # ax.plot(T.evaluate(), T2.evaluate(), 'ro', alpha=0.2)
    # plt.show()
    # plt.ion()
    # fig = plt.figure(figsize=(15,6))
    # ax = fig.add_subplot(121)
    # x = np.array([])
    # y = np.array([])
    # line, = ax.plot(x, y)
    # for i in range(50):
    #     loss = T.train_model(10)
    #     print('batch {:d}: current loss = {:0.6f}'.format(i, loss))
    #     x = np.append(x, i+1)
    #     y = np.append(y, loss)
    #     ax.cla()
    #     ax.plot(x,y)
    #     # plt.cla()
    #     # plt.plot(x,y)
    #     # line.set_xdata(x)
    #     # line.set_ydata(y)
    #     # fig.canvas.draw()
    #     # fig.canvas.flush_events()
    #     plt.pause(0.1)

    # plt.show(block=True)
    

if __name__ == '__main__':
    main()