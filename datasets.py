import glob
import os

import librosa
import numpy as np

from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, root, n_fft, hop_length, power, mels_samples, n_mels, sr=22050, mode="train"):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.mels_samples = mels_samples
        self.n_mels = n_mels
        self.sr = sr

        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*"))
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*"))

    def __getitem__(self, index):
        audio_A, _ = librosa.load(self.files_A[index % len(self.files_A)])
        audio_B, _ = librosa.load(self.files_B[index % len(self.files_B)])

        return {"A": self.to_logspec(audio_A), "B": self.to_logspec(audio_B)}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    
    def to_logspec(self, audio):
        """Convert an audio time serie to a croped/padded log1p melspectrogram"""

        spec = librosa.feature.melspectrogram(audio, sr=self.sr,
            n_fft=self.n_fft, hop_length=self.hop_length, power=self.power, 
            n_mels=self.n_mels)
        spec_crop = np.log1p(spec[:, :self.mels_samples])
        spec_pad = np.zeros((self.n_mels, self.mels_samples))
        spec_pad[:, :spec_crop.shape[1]] = spec_crop
        
        return spec_pad[np.newaxis, :] # [channel, heigh, width]

    def to_mp3(self, S, path):
        """Saves a spectrogram as an mp3 file"""
        
        S = S.cpu().detach().numpy()
        S = S.squeeze() # remove channel dim
        S = np.expm1(S)
        
        audio = librosa.feature.inverse.mel_to_audio(S, sr=self.sr,
            n_fft=self.n_fft, hop_length=self.hop_length, power=self.power)
        librosa.output.write_wav(path, audio, self.sr)
