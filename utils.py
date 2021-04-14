import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import StandardScaler

from config import CONTINUITY, CAT_DROP


class DataProcessor():
    def __init__(self):
        super().__init__()

    def _shuffle_split(self, test_ratio):
        data_size = self.x.shape[0]
        test_size = int(data_size * test_ratio)
        idx_list = list(range(data_size))
        np.random.shuffle(idx_list)
        idx_tr, idx_ts = idx_list[test_size:], idx_list[:test_size]
        self.x_tr, self.x_ts = self.x[idx_tr], self.x[idx_ts]
        self.y_tr, self.y_ts = self.y[idx_tr], self.y[idx_ts]

    def feature_z_norm(self):
        dims = []
        for d, c in enumerate(self.continuity):
            if c == 1:
                dims.append(d)
        for d in dims:
            x = np.concatenate((self.x_tr[:, d], self.x_ts[:, d]))
            self.x_tr[:, d] = (self.x_tr[:, d] - x.mean()) / x.std()
            self.x_ts[:, d] = (self.x_ts[:, d] - x.mean()) / x.std()

    def feature_minmax_scale(self):
        dims = []
        for d, c in enumerate(self.continuity):
            if c == 1:
                dims.append(d)
        for d in dims:
            x = np.concatenate((self.x_tr[:, d], self.x_ts[:, d]))
            self.x_tr[:, d] = (self.x_tr[:, d] - x.min()) / (x.max() - x.min() + 1e-6)
            self.x_ts[:, d] = (self.x_ts[:, d] - x.min()) / (x.max() - x.min() + 1e-6)

    def label_encode(self):
        encoder = LabelEncoder()
        encoder.fit(np.concatenate([self.y_tr, self.y_ts]))
        self.y_tr = encoder.transform(self.y_tr).astype(np.int32)
        self.y_ts = encoder.transform(self.y_ts).astype(np.int32)

    def get_full(self):
        return self.x_tr, self.y_tr, self.x_ts, self.y_ts

    def get_miss(self, miss_dims):
        feat_size = self.x_tr.shape[1]
        # perform random missing if a float missing ratio provided
        if isinstance(miss_dims, float):
            miss_size = int(feat_size * miss_dims)
            feat_dims = list(range(feat_size))
            np.random.shuffle(feat_dims)
            miss_dims = sorted(feat_dims[:miss_size])
        # construct keep_dims from miss_dims
        keep_dims = list(range(feat_size))
        for d in miss_dims:
            keep_dims.remove(d)
        # missing introduce
        x_tr_miss = self.x_tr[:, keep_dims]
        x_ts_miss = self.x_ts[:, keep_dims]

        return x_tr_miss, self.y_tr, x_ts_miss, self.y_ts, miss_dims, keep_dims

    def get_miss_indicator(self, miss_dims):
        feat_size = self.x_tr.shape[1]
        # perform random missing if a float missing ratio provided
        if isinstance(miss_dims, float):
            miss_size = int(feat_size * miss_dims)
            feat_dims = list(range(feat_size))
            np.random.shuffle(feat_dims)
            miss_dims = sorted(feat_dims[:miss_size])
        # construct keep_dims from miss_dims
        keep_dims = list(range(feat_size))
        for d in miss_dims:
            keep_dims.remove(d)

        return miss_dims, keep_dims

class LetterDataProcessor(DataProcessor):
    def __init__(self, test_ratio):
        super().__init__()
        self.continuity = np.array(CONTINUITY['letter'])
        self._read_file()
        self._shuffle_split(test_ratio=test_ratio)
        self.feature_minmax_scale()
        self.label_encode()

    def _read_file(self):
        df = pd.read_csv('data/uci/letter/data.txt', header=None)
        self.x = df.iloc[:, 1:].values.astype(np.float32)
        self.y = df.iloc[:, 0].values.astype(np.str)

def get_data_processor(dataset_name, test_ratio=0.2):
    if dataset_name == 'letter':
        return LetterDataProcessor(test_ratio=test_ratio)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    letter_dp = LetterDataProcessor(test_ratio=0.2)
    x_tr, y_tr, x_ts, y_ts, miss_dims, keep_dims = letter_dp.get_miss(0.2)
    print(x_tr.shape, y_tr.shape)
    print(x_ts.shape, y_ts.shape)
    print(x_tr[0])
    print(y_tr[:10])
    print(miss_dims)
    print(letter_dp.get_miss_indicator(miss_dims))
