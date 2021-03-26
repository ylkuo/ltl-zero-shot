import glob
import GPUtil
import numpy as np
import os
import random
import torchvision


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)


def choose_gpu(threshold=0.50):
    """Automatically choose the most available GPU"""
    gpus = GPUtil.getGPUs()
    gpus = [gpu for gpu in gpus if gpu.load < threshold and gpu.memoryUtil < threshold]
    gpu = random.choice(gpus).id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    return 'cuda:' + str(gpu)


def make_filter_image(layer, use_color=True, scale_each=True):
    """Build an image of the weights of the filters in a given convolutional layer."""
    weights = layer.weight.data.to("cpu")
    if not use_color:
        n_input_channels = weights.size()[1]
        weights = weights.view([weights.size()[0], 1, weights.size()[1]*weights.size()[2], weights.size()[3]])
    img = torchvision.utils.make_grid(weights, normalize=True, scale_each=scale_each)
    return img


def pad_slice(array, slice_r, slice_c):
    assert len(array.shape) >= 2

    r1, r2 = slice_r
    c1, c2 = slice_c
    assert r2 > r1
    assert c2 > c1

    pr1 = max(r1, 0)
    pc1 = max(c1, 0)

    sl = array[pr1:r2, pc1:c2, :]
    slr, slc = sl.shape[:2]

    padded_sl = np.zeros((r2 - r1, c2 - c1) + array.shape[2:])
    pad_fr_r = pr1 - r1
    pad_to_r = pad_fr_r + slr
    pad_fr_c = pc1 - c1
    pad_to_c = pad_fr_c + slc

    padded_sl[pad_fr_r:pad_to_r, pad_fr_c:pad_to_c, :] = sl

    return padded_sl


class Index:
    def __init__(self):
        self.contents = dict()
        self.ordered_contents = []
        self.reverse_contents = dict()

    def __getitem__(self, item):
        if item not in self.contents:
            return None
        return self.contents[item]

    def index(self, item):
        if item not in self.contents:
            idx = len(self.contents) + 1
            self.ordered_contents.append(item)
            self.contents[item] = idx
            self.reverse_contents[idx] = item
        idx = self[item]
        assert idx != 0
        return idx

    def get(self, idx):
        if idx == 0:
            return "*invalid*"
        return self.reverse_contents[idx]

    def __len__(self):
        return len(self.contents) + 1

    def __iter__(self):
        return iter(self.ordered_contents)
