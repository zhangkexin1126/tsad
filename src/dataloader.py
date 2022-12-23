import torch
import  numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
from torch.utils.data import DataLoader

class TSDataLoader():
    def __init__(self, dataname, win_size, step, mode="train"):

        self.dataname = dataname
        self.win_size = win_size
        self.step = step
        self.mode = mode
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = self._getdata(dataname)

    def _getdata(self, dataname):
        if dataname == "SMD":
            prefix = './data/SMD'
            """Train_data"""
            train_data = np.load(prefix + "/SMD_train.npy")
            train_label = np.zeros(len(train_data), dtype=int)
            scaler = StandardScaler()
            scaler.fit(train_data)
            train_data = scaler.transform(train_data)
            """Valid_data"""
            valid_data = train_data[(int)(len(train_data) * 0.8):]
            valid_label = train_label[(int)(len(train_data) * 0.8):]
            """Test_data"""
            test_data = np.load(prefix + "/SMD_test.npy")
            test_label = np.load(prefix + "/SMD_test_label.npy")
            test_data = scaler.transform(test_data)
            return train_data, train_label, valid_data, valid_label, test_data, test_label
        elif dataname == "MSL":
            prefix = './data/MSL'
            """Train_data"""
            train_data = np.load(prefix + "/MSL_train.npy")
            train_label = np.zeros(len(train_data), dtype=int)
            scaler = StandardScaler()
            scaler.fit(train_data)
            train_data = scaler.transform(train_data)
            """Valid_data"""
            valid_data = train_data[(int)(len(train_data) * 0.8):]
            valid_label = train_label[(int)(len(train_data) * 0.8):]
            """Test_data"""
            test_data = np.load(prefix + "/MSL_test.npy")
            test_label = np.load(prefix + "/MSL_test_label.npy")
            test_data = scaler.transform(test_data)
            return train_data, train_label, valid_data, valid_label, test_data, test_label
        elif dataname == "PSM":
            prefix = './data/PSM'
            """Train_data"""
            train_data = pd.read_csv(prefix + "/train.csv")
            train_data = train_data.values[:, 1:]
            train_label = np.zeros(len(train_data), dtype=int)
            train_data = np.nan_to_num(train_data)
            scaler = StandardScaler()
            scaler.fit(train_data)
            train_data = scaler.transform(train_data)
            """Valid_data"""
            valid_data = train_data[(int)(len(train_data) * 0.8):]
            valid_label = train_label[(int)(len(train_data) * 0.8):]
            """Test_data"""
            test_data = pd.read_csv(prefix + '/test.csv')
            test_data = test_data.values[:, 1:]
            test_data = np.nan_to_num(test_data)
            test_label = pd.read_csv(prefix + '/test_label.csv').values[:, 1:]
            test_label = test_label.flatten()
            return train_data, train_label, valid_data, valid_label, test_data, test_label
        elif dataname == "SMAP":
            prefix = './data/SMAP'
            """Train_data"""
            train_data = np.load(prefix + "/SMAP_train.npy")
            train_label = np.zeros(len(train_data), dtype=int)
            scaler = StandardScaler()
            scaler.fit(train_data)
            train_data = scaler.transform(train_data)
            """Valid_data"""
            valid_data = train_data[(int)(len(train_data) * 0.8):]
            valid_label = train_label[(int)(len(train_data) * 0.8):]
            """Test_data"""
            test_data = np.load(prefix + "/SMAP_test.npy")
            test_label = np.load(prefix + "/SMAP_test_label.npy")
            return train_data, train_label, valid_data, valid_label, test_data, test_label
        elif dataname == "SWAT":
            pass
        else:
            pass

    def __len__(self):
        if self.mode == "train":
            return (self.train_x.shape[0] - self.win_size) // self.step + 1 # can drop last
        elif self.mode == "valid":
            return (self.valid_x.shape[0] - self.win_size) // self.step + 1
        elif self.mode == "test":
            return (self.test_x.shape[0] - self.win_size) // self.step + 1
        else:
            pass

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            x = np.float32(self.train_x[index:index + self.win_size])
            y = np.float32(self.train_y[index:index + self.win_size])
            return x, y
        elif self.mode == "valid":
            x = np.float32(self.valid_x[index:index + self.win_size])
            y = np.float32(self.valid_y[index:index + self.win_size])
            return x, y
        elif self.mode == "test":
            x = np.float32(self.test_x[index:index + self.win_size])
            y = np.float32(self.test_y[index:index + self.win_size])
            return x, y


def get_loader(dataname="PSM", batch_size=256, win_size=100, step=100, mode='train'):
    dataset = TSDataLoader(dataname, win_size, step, mode)
    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)
    return data_loader, dataset


if __name__ == "__main__":
    pass
    # dl, ds = get_loader()
    # print(ds[0][0].shape, ds[0][0])

