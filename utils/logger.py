import random
import torch
from torch.utils.tensorboard import SummaryWriter
from .plot import plot_alignment_to_numpy, plot_spectrogram_to_numpy
from .plot import plot_gate_outputs_to_numpy


class TacotronLogger(SummaryWriter):
    def __init__(self, logdir):
        super(TacotronLogger, self).__init__(logdir)

    def log_training(self, loss, grad_norm, learning_rate, duration, iteration):
        self.add_scalar("training.loss", loss, iteration)
        self.add_scalar("grad.norm", grad_norm, iteration)
        self.add_scalar("learning.rate", learning_rate, iteration)
        self.add_scalar("duration", duration, iteration)

    def log_validation(self, loss, model, targets, predicts, iteration):
        self.add_scalar("validation.loss", loss, iteration)

        _, spec_predicts, stop_predicts, alignments = predicts
        if len(targets) == 3:
            _, spec_targets, stop_targets  = targets
        else:
            spec_targets, stop_targets  = targets

        # plot distribution of parameters
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)

        # plot alignment, mel target and predicted, stop_token target and predicted
        idx = random.randint(0, alignments.size(0) - 1)
        self.add_image(
            "alignment",
            plot_alignment_to_numpy(alignments[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "spec_target",
            plot_spectrogram_to_numpy(spec_targets[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "spec_predicted",
            plot_spectrogram_to_numpy(spec_predicts[idx].data.cpu().numpy().T),
            iteration, dataformats='HWC')
        self.add_image(
            "stop_token",
            plot_gate_outputs_to_numpy(
                stop_targets[idx].data.cpu().numpy(),
                torch.sigmoid(stop_predicts[idx]).data.cpu().numpy()),
            iteration, dataformats='HWC')
