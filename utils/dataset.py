import os
import random
import numpy as np
import torch

from text import text_to_sequence


class TextMelDataset(torch.utils.data.Dataset):
    """
        1) loads filepath,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) loads mel-spectrograms from mel files
    """
    def __init__(self, fname, hparams):
        self.text_cleaners  = hparams.text_cleaners
        self.symbols_lang   = hparams.symbols_lang
        self.n_mel_channels = hparams.n_mel_channels
        self.f_list = self.files_to_list(fname)
        random.seed(hparams.seed)
        random.shuffle(self.f_list)

    def files_to_list(self, file_path):
        f_list = []
        with open(file_path, encoding = 'utf-8') as f:
            for line in f:
                parts = line.strip().strip('\ufeff').split('|') #remove BOM
                # mel_file_path
                path  = parts[0]
                # text
                text  = parts[1]
                f_list.append([text, path])
        return f_list

    def get_mel_text_pair(self, text, file_path):
        text = self.get_text(text)
        mel = self.get_mel(file_path)
        return (text, mel)

    def get_mel(self, file_path):
        #stored melspec: np.ndarray [shape=(T_out, num_mels)]
        #in Pytorch [shape=(num_mels, T_out)]
        melspec = torch.from_numpy(np.load(file_path).T)
        assert melspec.size(0) == self.n_mel_channels, (
            'Mel dimension mismatch: given {}, expected {}'.format(melspec.size(0), self.n_mel_channels))

        return melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners, self.symbols_lang))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(*self.f_list[index])

    def __len__(self):
        return len(self.f_list)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per step
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths
