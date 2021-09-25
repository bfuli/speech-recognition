# encoding=utf-8
# Author:fuli
# Date:2021/9/23
import numpy as np
import librosa.display
import matplotlib.pyplot as plt


def trim_silence(audio, n_noise_samples=1000, noise_factor=1.0, mean_filter_size=100):
    """ 语音文件降噪，并去除前后空白段
    :param audio: numpy array of audio
    :return: a trimmed numpy array
    """
    start = 0
    end = len(audio) - 1
    mag = abs(audio)

    noise_sample_period = mag[end - n_noise_samples:end]
    noise_threshold = noise_sample_period.max() * noise_factor

    mag_mean = np.convolve(mag, [1 / float(mean_filter_size)] * mean_filter_size, 'same')

    # find onset
    for idx, point in enumerate(mag_mean):
        if point > noise_threshold:
            start = idx
            break

    # Reverse the array for trimming the end
    for idx, point in enumerate(mag_mean[::-1]):
        if point > noise_threshold:
            end = len(audio) - idx
            break

    return audio[start:end]


def draw(audio_file):
    data, sample_rate = librosa.load(audio_file)
    plt.subplot(1, 2, 1)
    librosa.display.waveplot(data, sr=sample_rate)

    trim_data = trim_silence(data)
    plt.subplot(1, 2, 2)
    librosa.display.waveplot(trim_data)
    plt.show()


if __name__ == '__main__':
    draw("res/转换后/B.wav")