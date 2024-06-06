import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from GetData_Dy import GetDataSet,GetDataSet2,GetDataSet3

class client1(object):
    def __init__(self,dev):
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate1(self, localEpoch, localBatchSize, Net, lossFun, opti, local_parameters):
        gds = GetDataSet()
        x_tr = torch.from_numpy(gds.train_data).float()  # from numpy --> to tensor
        y_tr = torch.from_numpy(gds.train_label)
        torch_dataset = TensorDataset(x_tr, y_tr)
        Net.load_state_dict(local_parameters, strict=True)
        self.train_dl = DataLoader(dataset=torch_dataset, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label.long())
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def local_val(self):
        pass


class client2(object):
    def __init__(self,dev):
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate2(self, localEpoch, localBatchSize, Net, lossFun, opti, local_parameters):
        gds = GetDataSet2()
        x_tr = torch.from_numpy(gds.train_data).float()
        y_tr = torch.from_numpy(gds.train_label)
        torch_dataset = TensorDataset(x_tr, y_tr)
        Net.load_state_dict(local_parameters, strict=True)
        self.train_dl = DataLoader(dataset=torch_dataset, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label.long())
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def local_val(self):
        pass


class client3(object):
    def __init__(self,dev):
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    def localUpdate3(self, localEpoch, localBatchSize, Net, lossFun, opti, local_parameters):
        gds = GetDataSet3()
        x_tr = torch.from_numpy(gds.train_data).float()
        y_tr = torch.from_numpy(gds.train_label)
        torch_dataset = TensorDataset(x_tr, y_tr)
        Net.load_state_dict(local_parameters, strict=True)
        self.train_dl = DataLoader(dataset=torch_dataset, batch_size=localBatchSize, shuffle=True)

        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)
                preds = Net(data)
                loss = lossFun(preds, label.long())
                loss.backward()
                opti.step()
                opti.zero_grad()

        return Net.state_dict()

    def local_val(self):
        pass







