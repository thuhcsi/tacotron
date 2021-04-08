import os
import time
import argparse
import json

import torch
from torch.utils.data import DataLoader

from model.tacotron import Tacotron, TacotronLoss
from model.tacotron2 import Tacotron2, Tacotron2Loss
from utils.dataset import TextMelDataset, TextMelCollate
from utils.logger import TacotronLogger
from utils.utils import data_parallel_workaround
from hparams import create_hparams


def prepare_datasets(hparams):
    # Get data, data loaders and collate function ready
    trainset = TextMelDataset(hparams.training_files, hparams)
    valset = TextMelDataset(hparams.validation_files, hparams)
    collate_fn = TextMelCollate(hparams.r)
    #
    return trainset, valset, collate_fn


def create_model(hparams):
    # Model config
    with open(hparams.tacotron_config, 'r') as f:
        model_cfg = json.load(f)
    if hparams.tacotron_version == "1":
        # Tacotron model
        model = Tacotron(n_vocab=hparams.num_symbols,
                         embed_dim=hparams.symbols_embed_dim,
                         mel_dim=hparams.mel_dim,
                         linear_dim=hparams.mel_dim,
                         n_speaker=hparams.num_speakers,
                         speaker_embed_dim=hparams.speaker_embed_dim,
                         max_decoder_steps=hparams.max_decoder_steps,
                         stop_threshold=hparams.stop_threshold,
                         r=hparams.r,
                         model_cfg=model_cfg
                         )
        # Loss criterion
        criterion = TacotronLoss(hparams.speaker_loss_weight)
    elif hparams.tacotron_version == "2":
        # Tacotron2 model
        model = Tacotron2(n_vocab=hparams.num_symbols,
                          embed_dim=hparams.symbols_embed_dim,
                          mel_dim=hparams.mel_dim,
                          n_speaker=hparams.num_speakers,
                          speaker_embed_dim=hparams.speaker_embed_dim,
                          max_decoder_steps=hparams.max_decoder_steps,
                          stop_threshold=hparams.stop_threshold,
                          r=hparams.r,
                          model_cfg=model_cfg
                          )
        # Loss criterion
        criterion = Tacotron2Loss(hparams.speaker_loss_weight)
    else:
        raise ValueError("Unsupported Tacotron version: {} ".format(hparams.tacotron_version))
    #
    return model, criterion


def warm_start_model(checkpoint_path, model, ignore_layers):
    assert os.path.isfile(checkpoint_path)
    print("Warm starting model from checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model_dict = checkpoint_dict['state_dict']
    if len(ignore_layers) > 0:
        model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        dummy_dict = model.state_dict()
        dummy_dict.update(model_dict)
        model_dict = dummy_dict
    model.load_state_dict(model_dict)
    return model


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    learning_rate = checkpoint['learning_rate']
    iteration = checkpoint['iteration']

    print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(checkpoint_path, model, optimizer, learning_rate, iteration):
    print("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path))
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)


def validate(model, criterion, iteration, device, valset, batch_size, collate_fn, logger):
    """Evaluate on validation set, get validation loss and printing
    """
    model.eval()
    with torch.no_grad():
        valdata_loader = DataLoader(valset, sampler=None, num_workers=1,
                                shuffle=False, batch_size=batch_size,
                                pin_memory=False, collate_fn=collate_fn)

        val_loss = 0.0
        for i, batch in enumerate(valdata_loader):
            inputs, targets = model.parse_data_batch(batch)
            predicts = model(inputs)

            # Loss
            loss = criterion(predicts, targets)

            val_loss += loss.item()
        val_loss = val_loss / (i + 1)

    model.train()
    print("Validation loss {}: {:9f}  ".format(iteration, val_loss))
    logger.log_validation(val_loss, model, targets, predicts, iteration)


def train(output_dir, log_dir, checkpoint_path, warm_start, hparams):
    """Training and validation logging results to tensorboard and stdout

    Params
    ------
    output_dir (string): directory to save checkpoints
    log_dir (string): directory to save tensorboard logs
    checkpoint_path (string): path to load checkpoint
    warm_start (bool): load model weights only, ignore specified layers
    hparams (object): comma separated list of "name=value" pairs
    """

    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed(hparams.seed)

    # Prepare device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    # Decide parallel running on GPU
    parallel_run = hparams.distributed_run and torch.cuda.device_count() > 1

    # Instantiate Tacotron Model
    print("\nInitialising Tacotron Model...\n")
    model, criterion = create_model(hparams)
    model = model.to(device)

    # Initialize the optimizer
    learning_rate = hparams.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=hparams.weight_decay)

    # Prepare directory and logger
    os.makedirs(output_dir, exist_ok=True)
    logger = TacotronLogger(log_dir)

    # Prepare dataset and dataloader
    trainset, valset, collate_fn = prepare_datasets(hparams)
    train_loader = DataLoader(trainset, sampler=None, num_workers=1,
                              shuffle=True, batch_size=hparams.batch_size,
                              pin_memory=False, drop_last=True, collate_fn=collate_fn)

    # Load checkpoint if one exists
    iteration    = -1 # will add 1 in main loop
    epoch_offset = 0
    if checkpoint_path is not None:
        if warm_start:
            model = warm_start_model(checkpoint_path, model, hparams.ignore_layers)
        else:
            model, optimizer, _learning_rate, iteration = load_checkpoint(checkpoint_path, model, optimizer)
            if hparams.use_saved_learning_rate:
                learning_rate = _learning_rate
            epoch_offset = max(0, int(iteration / len(train_loader)))

    # ================ MAIN TRAINNIG LOOP! ===================
    model.train()
    for epoch in range(epoch_offset, hparams.epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            start = time.perf_counter()
            iteration += 1

            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate

            # prepare data
            inputs, targets = model.parse_data_batch(batch)

            # Forward pass
            # Parallelize model onto GPUS using workaround due to python bug
            if parallel_run:
                predicts = data_parallel_workaround(model, inputs)
            else:
                predicts = model(inputs)

            # Loss
            loss = criterion(predicts, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.grad_clip_thresh)
            optimizer.step()

            # Logs
            duration = time.perf_counter() - start
            print("Train loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(iteration, loss, grad_norm, duration))
            logger.log_training(loss, grad_norm, learning_rate, duration, iteration)

            # Validation
            if iteration % hparams.iters_per_validation == 0:
                validate(model, criterion, iteration, device, valset, hparams.batch_size, collate_fn, logger)

            # Save checkpoint
            if iteration % hparams.iters_per_checkpoint == 0:
                checkpoint_path = os.path.join(output_dir, "checkpoint_{}".format(iteration))
                save_checkpoint(checkpoint_path, model, optimizer, learning_rate, iteration)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_dir', type=str, default='out',
                        help='directory to save checkpoints')
    parser.add_argument('-l', '--log_dir', type=str, default='log',
                        help='directory to save tensorboard logs')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
    parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')

    args = parser.parse_args()
    hparams = create_hparams()

    torch.backends.cudnn.enabled = hparams.cudnn_enabled
    torch.backends.cudnn.benchmark = hparams.cudnn_benchmark

    print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
    print("Distributed Run:", hparams.distributed_run)
    print("cuDNN Enabled:", hparams.cudnn_enabled)
    print("cuDNN Benchmark:", hparams.cudnn_benchmark)

    train(args.output_dir, args.log_dir, args.checkpoint_path, args.warm_start, hparams)
