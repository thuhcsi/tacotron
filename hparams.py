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
        # Audio Parameters             #
        ################################
        self.n_mel_channels = 80

        ################################
        # Model Parameters             #
        ################################
        self.tacotron_version = "1"  # 1: Tacotron; 2: Tacotron-2
        self.tacotron_config = "tacotron1.json"

        self.n_symbols = len(symbols(self.symbols_lang))
        self.symbols_embedding_dim = 512

        # Encoder parameters
        self.encoder_kernel_size = 5
        self.encoder_n_convolutions = 3
        self.encoder_embedding_dim = 512

        # Decoder parameters
        self.n_frames_per_step = 3
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        # Attention parameters
        self.attention_rnn_dim = 1024
        self.attention_dim = 128

        # Location Layer parameters
        self.attention_location_n_filters = 32
        self.attention_location_kernel_size = 31

        # Mel-post processing network parameters
        self.postnet_embedding_dim = 512
        self.postnet_kernel_size = 5
        self.postnet_n_convolutions = 5

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