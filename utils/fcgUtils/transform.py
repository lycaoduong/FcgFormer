import numpy as np
import torch
import cv2


class AddNoise(object):
    def __init__(self, p=0.2, target_snr_db=[2, 10], mean_noise=0):
        self.p = p
        self.snr = target_snr_db
        self.mean_noise = mean_noise

    def __call__(self, sample):
        signal, label, fn = sample['signal'], sample['label'], sample['fn']
        if self.p > np.random.rand():
            target_snr_db = np.random.randint(self.snr[0], self.snr[1])
            _, signal_len = signal.shape

            x_watts = signal ** 2
            sig_avg_watts = np.mean(x_watts)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            noise_avg_db = sig_avg_db - target_snr_db
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            noise = np.random.normal(self.mean_noise, np.sqrt(noise_avg_watts), signal_len)
            signal = signal + noise
        sample = {'signal': signal, 'label': label, 'fn': fn}
        return sample


class Revert(object):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, sample):
        signal, label, fn = sample['signal'], sample['label'], sample['fn']
        if self.p > np.random.rand():
            signal = signal[:, ::-1]
        sample = {'signal': signal, 'label': label, 'fn': fn}
        return sample


class MaskZeros(object):
    def __init__(self, p=0.2, mask_p=[0.1, 0.3]):
        self.p = p
        self.mask_p = mask_p

    def __call__(self, sample):
        signal, label, fn = sample['signal'], sample['label'], sample['fn']
        if self.p > np.random.rand():
            _, signal_len = signal.shape
            target_mask_p = np.random.uniform(self.mask_p[0], self.mask_p[1])
            mask_size = int(target_mask_p * signal_len)
            target_mask = np.random.randint(0, signal_len-1, mask_size)
            signal[:, target_mask] = 0.0
        sample = {'signal': signal, 'label': label, 'fn': fn}
        return sample


class ShiftLR(object):
    def __init__(self, p=0.2, shift_p=[0.01, 0.05]):
        self.p = p
        self.shift_p = shift_p

    def __call__(self, sample):
        signal, label, fn = sample['signal'], sample['label'], sample['fn']
        if self.p > np.random.rand():
            _, signal_len = signal.shape
            target_shift_p = np.random.uniform(self.shift_p[0], self.shift_p[1])
            shift_size = int(target_shift_p * signal_len)
            shift_signal = np.zeros_like(signal)
            if np.random.rand() > 0.5:
                shift_signal[:, shift_size:] = signal[:, :-shift_size]
            else:
                shift_signal[:, :-shift_size] = signal[:, shift_size:]
            sample = {'signal': shift_signal, 'label': label, 'fn': fn}
        else:
            sample = {'signal': signal, 'label': label, 'fn': fn}
        return sample

class ShiftUD(object):
    def __init__(self, p=0.2, shift_p=[0.01, 0.05]):
        self.p = p
        self.shift_p = shift_p

    def __call__(self, sample):
        signal, label, fn = sample['signal'], sample['label'], sample['fn']
        if self.p > np.random.rand():
            max_value = np.max(signal)
            target_shift_p = np.random.uniform(self.shift_p[0], self.shift_p[1])
            offset_value = max_value * target_shift_p
            if np.random.rand() > 0.5:
                signal += offset_value
            else:
                signal -= offset_value
        sample = {'signal': signal, 'label': label, 'fn': fn}
        return sample


class Normalizer(object):
    def __init__(self, with_std=False, mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])
        self.with_std = with_std

    def __call__(self, sample):
        signal, label, fn = sample['signal'], sample['label'], sample['fn']
        max = np.max(signal, axis=1)
        min = np.min(signal, axis=1)
        if self.with_std:
            signal = (((signal.astype(np.float32) - min) / (max - min)) - self.mean) / self.std
        else:
            signal = ((signal.astype(np.float32) - min) / (max - min))
        sample = {'signal': signal, 'label': label, 'fn': fn}
        return sample


class Resizer(object):
    def __init__(self, signal_size=1024):
        self.signal_size = signal_size

    def __call__(self, sample):
        signal, label, fn = sample['signal'], sample['label'], sample['fn']

        signal = cv2.resize(signal, (self.signal_size, 1), interpolation=cv2.INTER_CUBIC)

        sample = {'signal': torch.from_numpy(signal).to(torch.float32), 'label': torch.from_numpy(label).to(torch.long), 'fn': fn}
        return sample


def collater(data):
    signal = [s['signal'] for s in data]
    labels = [s['label'] for s in data]
    fns = [s['fn'] for s in data]
    signal = torch.from_numpy(np.stack(signal, axis=0))
    labels = torch.from_numpy(np.stack(labels, axis=0))
    sample = {'signal': signal, 'label': labels, 'fn': fns}
    return sample
