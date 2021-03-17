from text import symbols


class Hparams:
    def __init__(self):
        ################################
        # Experiment Parameters        #
        ################################
        self.epochs = 500
        self.iters_per_checkpoint = 1000
        self.iters_per_validation = 1000
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
        self.tacotron_version = "1"  # 1: Tacotron; 2: Tacotron-2
        self.tacotron_config = "tacotron1.json"

        self.num_symbols = len(symbols(self.symbols_lang))
        self.symbols_embed_dim = 512
        self.mel_dim = 80
        self.r = 3
        self.max_decoder_steps = 1000
        self.stop_threshold = 0.5

        ################################
        # Optimization Hyperparameters #
        ################################
        self.use_saved_learning_rate = False
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.grad_clip_thresh = 1.0
        self.batch_size = 32
        self.mask_padding = True  # set model's padded outputs to padded values

    def __str__(self):
        return "\n".join(
            ["Hyper Parameters:"]
            + ["{}:{}".format(key, getattr(self, key, None)) for key in self.__dict__]
        )


def create_hparams():
    """Create model hyperparameters. Parse nondefault from object args."""
    return Hparams()