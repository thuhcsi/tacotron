# PyTacotron

PyTorch implementation of [Tacotron: Towards End-to-End Speech Synthesis](https://arxiv.org/abs/1703.10135), and  
PyTorch implementation of [Natural TTS synthesis by conditioning Wavenet on MEL spectogram predictions](https://arxiv.org/abs/1712.05884).

## Features

- Easy switch between [Tacotron](https://arxiv.org/abs/1703.10135) and [Tacotron2](https://arxiv.org/abs/1712.05884)
- Detailed model structure configuration with json
    - For Tacotron: [tacotron1.json](tacotron1.json)
    - For Tacotron2: [tacotron2.json](tacotron2.json)
- Dynamic reduction factor (r) changing along with training schedule

## Setup
1. Prepare `DATASET` directory
    - Prepare `train.csv.txt` and `val.csv.txt` files
    - Change `training_files` and `validation_files` in [hparams.py](hparams.py) to the above two files respectively
    - Make necessary modifications to `files_to_list` to retrieve **'mel_file_path'** and **'text'** in [utils/dataset.py](utils/dataset.py)
2. Install PyTorch
3. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`


## Training

### Training from scratch
1. `python train.py -o outdir -l logdir`
2. (OPTIONAL) `tensorboard --logdir=logdir`

### Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are ignored

1. Download the published Tacotron model
2. `python train.py -o outdir -l logdir -c tacotron_statedict.pt --warm_start`

### Multi-GPU (distributed) Training
1. `python train.py -o outdir -l logdir --hparams=distributed_run=True`


## Inference demo
1. Download published Tacotron model
2. Download published WaveGAN model
3. `jupyter notebook --ip=127.0.0.1 --port=31337`
4. Load inference.ipynb 

***Note***: When performing Mel-Spectrogram to Audio synthesis, make sure Tacotron
and the Mel decoder were trained on the same mel-spectrogram representation. 


## Acknowledgements
This implementation uses code from the following repos as described in our code.

- [Tacotron2 by NVIDIA](https://github.com/NVIDIA/tacotron2)
- [Tacotron by blue-fish in Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/b5ba6d0371882dbab595c48deb2ff17896547de7/synthesizer)
- [Tacotron by r9y9](https://github.com/r9y9/tacotron_pytorch)
- [Tacotron by keithito](https://github.com/keithito/tacotron)
