# encoding=utf-8
# Author:fuli
# Date:2021/9/21
from pydub import AudioSegment
import os.path as path


def get_name(audio_path):
    return path.basename(audio_path).split(".")[0]


def m4a2wav(audio_path, dest_path):
    """
    将m4a格式的语音文件转成wav格式，转换后的文件名与转换前相同，知识后缀不同
    :param audio_path: 原语音文件位置
    :param dest_path: 转换后的文件目录
    :return:
    """
    song = AudioSegment.from_file(audio_path, format="m4a")
    audio_name = get_name(audio_path)
    song.export(path.join(dest_path, audio_name + ".wav"), format="wav")
