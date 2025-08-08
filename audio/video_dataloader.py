import os
from torch.utils import data
import torchaudio
import numpy as np


class AudioRecord(object):
    """仅存储音频路径和标签的记录类"""
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]  # 原始视频路径（用于提取音频ID）

    @property
    def label(self):
        return float(self._data[2])  # 标签值


class AudioOnlyDataset(data.Dataset):
    def __init__(self, list_file, mode='train'):
        """
        仅加载完整音频的数据集（不进行分割）
        :param list_file: 数据列表文件路径，格式为 [video_id, num_frames, label]
        :param mode: 训练/测试模式（用于区分音频路径）
        """
        self.list_file = list_file
        self.mode = mode
        self._parse_list()

    def _parse_list(self):
        """解析数据列表，生成音频记录列表"""
        with open(self.list_file, 'r') as f:
            lines = [line.strip().split() for line in f if line.strip()]
        self.audio_list = [AudioRecord(line) for line in lines if len(line) >= 3]
        print(f"加载音频样本数量: {len(self.audio_list)}")

    def _wav2fbank(self, audio_path):
        """提取完整音频的FBANK特征（不分割）"""
        # 加载音频波形
        waveform, sr = torchaudio.load(audio_path)
        waveform = waveform - waveform.mean()  # 去均值

        # 提取FBANK特征
        fbank = torchaudio.compliance.kaldi.fbank(
            waveform,
            htk_compat=True,
            sample_frequency=sr,
            use_energy=False,
            window_type='hanning',
            num_mel_bins=128,
            dither=0.0,
            frame_shift=10  # 10ms帧移
        )

        # 固定特征长度为512帧（与原有逻辑保持一致）
        target_length = 512
        n_frames = fbank.shape[0]
        p = target_length - n_frames

        if p > 0:
            fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p), mode='constant')
        elif p < 0:
            fbank = fbank[:target_length, :]

        return fbank

    def __getitem__(self, index):
        record = self.audio_list[index]
        
        # 从视频路径中提取音频文件名（假设视频文件夹名即音频ID）
        video_folder = os.path.basename(os.path.dirname(record.path))
        audio_filename = f"{video_folder}.wav"
        
        if self.mode == 'train':
            audio_path = os.path.join('/root/autodl-tmp/AVEC/wav/train', audio_filename)
        else:
            audio_path = os.path.join('/root/autodl-tmp/AVEC/wav/dev', audio_filename)

        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        fbank = self._wav2fbank(audio_path)
        
        fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
        
        fbank = fbank.unsqueeze(0)

        return record.label, fbank

    def __len__(self):
        return len(self.audio_list)


# 数据加载器创建函数
def train_data_loader(list_file):
    return AudioOnlyDataset(list_file, mode='train')

def test_data_loader(list_file):
    return AudioOnlyDataset(list_file, mode='test')