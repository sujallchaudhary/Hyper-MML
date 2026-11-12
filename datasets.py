import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import os
import pickle, pandas as pd
import numpy

# Root of the feature folders; override with env var HYPERMML_ROOT (e.g. a Modal volume mount)
DATA_ROOT = os.environ.get('HYPERMML_ROOT', '/root/autodl-tmp/dataset')


class SoftSelectMixin:
    """Access to raw modality arrays and in-place feature selection (used by rough-set reduction).
    MODS maps a modality key (eeg/aud/vis) to the (train_attr, test_attr) holding its features."""
    MODS = {}

    def mod_array(self, mod, train=True):
        tr, te = self.MODS[mod]
        return numpy.asarray(getattr(self, tr if train else te), dtype=numpy.float32)

    def labels(self, train=True):
        return numpy.asarray(self.tr_lab if train else self.te_lab).astype(int).ravel()

    def groups(self, train=True):
        """Optional trial ids of the samples (windows of one trial must not be split across train/valid)."""
        g = getattr(self, 'tr_grp' if train else 'te_grp', None)
        return None if g is None else numpy.asarray(g).astype(int).ravel()

    def select_features(self, mod, idx):
        for attr in self.MODS[mod]:
            setattr(self, attr, numpy.asarray(getattr(self, attr), dtype=numpy.float32)[:, idx])

class EAVDataset(SoftSelectMixin, Dataset):
    MODS = {'eeg': ('tr_eeg', 'te_eeg'), 'aud': ('tr_aud', 'te_aud'), 'vis': ('tr_vis', 'te_vis')}

    def __init__(self, train=True, subject=1):
        '''
            label index mapping = {'Neutral':0, 'Sadness':1, 'Anger':2, 'Happiness':3, 'Calmness':4}
        '''
        self.subject = subject
        self.tr_aud, self.te_aud = pickle.load(open(f'{DATA_ROOT}/EAV/features/Audio_openSMILE/subject_{subject:02d}_aud.pkl', 'rb'))
        self.tr_vis, self.te_vis = pickle.load(open(f'{DATA_ROOT}/EAV/features/Vision_manet/subject_{subject:02d}_vis.pkl', 'rb'))
        self.tr_eeg, self.te_eeg = pickle.load(
            open(f'{DATA_ROOT}/EAV/features/EEG_NESTA/subject_{subject:02d}_eeg.pkl', 'rb'))
        self.tr_lab, self.te_lab = pickle.load(
            open(f'{DATA_ROOT}/EAV/features/lab/subject_{subject:02d}_lab.pkl', 'rb'))
        self.train = train
        if self.train:
            self.len = len(self.tr_lab)
        else:
            self.len = len(self.te_lab)

    def __getitem__(self, index):
        if self.train:
            return torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_vis[index])), \
                torch.FloatTensor(numpy.array(self.tr_aud[index])), \
                torch.FloatTensor(numpy.array([1,0])), \
                torch.FloatTensor(numpy.array([1])), \
                torch.FloatTensor(numpy.array([self.tr_lab[index]])),\
                index
        else:
            return torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_vis[index])), \
                torch.FloatTensor(numpy.array(self.te_aud[index])), \
                torch.FloatTensor(numpy.array([1,0])), \
                torch.FloatTensor(numpy.array([1])), \
                torch.FloatTensor(numpy.array([self.te_lab[index]])), \
                index

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]

class AFFECDataset(SoftSelectMixin, Dataset):
    # slot naming follows train.py: 'aud' slot = GSR, 'vis' slot = eye-tracking
    MODS = {'eeg': ('tr_eeg', 'te_eeg'), 'aud': ('tr_gsr', 'te_gsr'), 'vis': ('tr_eye', 'te_eye')}

    def __init__(self, train=True, subject='1'):

        self.subject = subject
        if isinstance(subject, int):  # 1-based index into the alphabetically sorted subjects found on disk (1 -> 'acl')
            subs = sorted(d[4:] for d in os.listdir(f'{DATA_ROOT}/AFFEC/feature') if d.startswith('sub-'))
            user = subs[subject - 1]
        else:
            user = subject
        self.tr_eye, self.te_eye = pickle.load(open(f'{DATA_ROOT}/AFFEC/feature/sub-{user}/sub-{user}_eye.pkl', 'rb'))
        self.tr_gsr, self.te_gsr = pickle.load(open(f'{DATA_ROOT}/AFFEC/feature/sub-{user}/sub-{user}_gsr.pkl', 'rb'))
        self.tr_eeg, self.te_eeg = pickle.load(
            open(f'{DATA_ROOT}/AFFEC/feature/sub-{user}/sub-{user}_eeg.pkl', 'rb'))
        _, _, self.tr_lab, self.tr_grp, _, _, self.te_lab, self.te_grp = pickle.load(
            open(f'{DATA_ROOT}/AFFEC/feature/sub-{user}/sub-{user}_lab.pkl', 'rb'))
        self.train = train
        if self.train:
            self.len = len(self.tr_lab)
        else:
            self.len = len(self.te_lab)

    def __getitem__(self, index):
        if self.train:
            return torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eeg[index])), \
                torch.FloatTensor(numpy.array(self.tr_eye[index])), \
                torch.FloatTensor(numpy.array(self.tr_gsr[index])), \
                torch.FloatTensor(numpy.array([1,0])), \
                torch.FloatTensor(numpy.array([1])), \
                torch.FloatTensor(numpy.array([self.tr_lab[index]])),\
                index
        else:
            return torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eeg[index])), \
                torch.FloatTensor(numpy.array(self.te_eye[index])), \
                torch.FloatTensor(numpy.array(self.te_gsr[index])), \
                torch.FloatTensor(numpy.array([1,0])), \
                torch.FloatTensor(numpy.array([1])), \
                torch.FloatTensor(numpy.array([self.te_lab[index]])), \
                index

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i<7 else pad_sequence(dat[i], True) if i<9 else dat[i].tolist() for i in dat]
