# encoding=utf-8
# Author:fuli
# Date:2021/9/19
from torch.utils.data import Dataset, DataLoader
import librosa
import torch
import os
import numpy as np
import utils


class AudioDataSet(Dataset):
    # 通过该类的get_dataset()方法直接返回需要的训练、测试数据集，可以实现数据集打乱
    def __init__(self, lines, audio_dir, is_train, train_ratio, is_silence):
        """
        :param lines: 打乱之后的文件标注数组，每一项由（labels，audio_name）组成
        :param audio_dir: 语音文件根目录
        :param is_train: 训练or测试
        :param train_ratio: 训练数据集占总数据集的比例
        :param is_silence: 是否进行降噪，去除前后空白段处理
        """
        labels = [line.strip("\n").split(",")[0] for line in lines]
        audio_names = [line.strip("\n").split(",")[1] for line in lines]

        tag = int(len(labels) * train_ratio)
        if is_train:
            self.labels = labels[:tag]
            self.audio_names = audio_names[:tag]
        else:
            self.labels = labels[tag:]
            self.audio_names = audio_names[tag:]
        self.audio_dir = audio_dir
        self.classes = ["A", "B", "C", "D"]
        self.is_silence = is_silence

    def __getitem__(self, index):
        """
        :param index:
        :return: 返回：标签的下标， 二维tensor
        """
        filePath = os.path.join(self.audio_dir, self.audio_names[index])
        data = AudioDataSet.audio2data(filePath, self.is_silence)

        return data, self.classes.index(self.labels[index])

    def __len__(self):
        return len(self.labels)

    @classmethod
    def get_dataset(cls, annotation_file, audio_dir, train_ratio=0.9, is_shuffle=False):
        """
        返回生成的训练集和测试集
        :param annotation_file: 带标注的文件地址，每一行包含(标签，语音文件名)，并以“,”分割
        :param audio_dir: 语音文件根目录
        :param train_ratio: 训练数据集占总数据集的比例
        :param is_shuffle: 表示原始数据集是否打乱
        :return: train, test数据集
        """
        with open(annotation_file, encoding="utf-8") as f:
            lines = f.readlines()
        f.close()

        if is_shuffle:
            np.random.shuffle(lines)

        train = AudioDataSet(lines, audio_dir, is_train=True, train_ratio=train_ratio, is_silence=False)
        test = AudioDataSet(lines, audio_dir, is_train=False, train_ratio=train_ratio, is_silence=False)
        return train, test

    @classmethod
    def audio2data(cls, filePath, is_silence=False):
        """
        将语音文件转换成二维数组
        :param filePath: 语音文件地址
        :param is_silence: 是否进行降噪、去除前后空白段
        :return: 二维tensor
        """
        y, sr = librosa.load(filePath, sr=None)

        # 语音文件降噪，并去除前后空白段
        if is_silence:
            y = utils.trim_silence(y)

        # 提取Log-MelSpectrogram特征
        mel_spec = librosa.feature.melspectrogram(y, sr, n_fft=2048, n_mels=128)
        log_mel_spec = librosa.power_to_db(mel_spec)
        resampled = librosa.resample(y=log_mel_spec, orig_sr=sr, target_sr=np.floor(50 * sr / len(log_mel_spec[1])),
                                     fix=True)
        return torch.tensor(resampled)
