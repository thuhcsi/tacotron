from text import symbols


class Hparams:
    def __init__(self):
        ################################
        # Experiment Parameters        #
        ################################
        self.seed = 1234
        self.dynamic_loss_scaling = True
        self.fp16_run = False
        self.distributed_run = False
        self.cudnn_enabled = True
        self.cudnn_benchmark = False
        self.ignore_layers = ["embedding.weight"]

        ################################
        # Data Parameters              #
        ################################
        self.training_files = "DATASET/train.csv.txt"
        self.validation_files = "DATASET/val.csv.txt"
        self.text_cleaners = ["basic_cleaners"]
        self.symbols_lang = "en"  # en: English characters; py: Chinese Pinyin symbols

        ################################
        # Model Parameters             #
        ################################
        self.tacotron_version = "2"  # 1: Tacotron; 2: Tacotron-2
        self.tacotron_config = "tacotron2.json"

        self.num_symbols = len(symbols(self.symbols_lang))
        self.symbols_embed_dim = 512
        self.mel_dim = 80
        self.max_decoder_steps = 1000
        self.stop_threshold = 0.5

        ################################
        # Model training Parameters    #
        ################################
        self.schedule = [(10, 1e-3,  10_000,  32),   # Progressive training schedule
                         (3,  1e-3,  20_000,  32),   # (r, lr, iters, batch_size)
                         (3,  5e-4,  40_000,  32),   #
                         (2,  2e-4,  80_000,  32),   # r = reduction factor (# of mel frames
                         (2,  1e-4, 120_000,  32),   #     synthesized for each decoder iteration)
                         (1,  3e-5, 160_000,  32),   # lr = learning rate
                         (1,  1e-5, 200_000,  32)]   # iters = maximum iteration steps

        self.grad_clip_thresh = 1.0                  # Clip the gradient norm to prevent explosion
        self.iters_per_checkpoint = 1000             # Number of iterations between model checkpoint saving
        self.iters_per_validation = 1000             # Number of iterations between model validation

    def __str__(self):
        return "\n".join(
            ["Hyper Parameters:"]
            + ["{}:{}".format(key, getattr(self, key, None)) for key in self.__dict__]
        )


def create_hparams():
    """Create model hyperparameters. Parse nondefault from object args."""
    return Hparams()