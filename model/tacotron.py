""" adapted from https://github.com/r9y9/tacotron_pytorch """
""" with reference to https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/b5ba6d0371882dbab595c48deb2ff17896547de7/synthesizer """

import torch
from torch import nn

from .attention import BahdanauAttention, AttentionWrapper
from .attention import get_mask_from_lengths
from .modules import Prenet, CBHG


class Encoder(nn.Module):
    def __init__(self, embed_dim,
                 prenet_dims=[256, 128], prenet_dropout=0.5,
                 K=16, conv_channels=128, pool_kernel_size=2,
                 proj_channels=[128, 128], proj_kernel_size=3,
                 num_highways=4, highway_units=128, rnn_units=128):
        # initialize
        super(Encoder, self).__init__()
        self.prenet = Prenet(embed_dim, prenet_dims, prenet_dropout)
        self.cbhg = CBHG(prenet_dims[-1], K, conv_channels, pool_kernel_size,
                         proj_channels, proj_kernel_size,
                         num_highways, highway_units, rnn_units)

    def forward(self, inputs):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs)


class Decoder(nn.Module):
    # Class variable because its value doesn't change between classes
    # yet ought to be scoped by class because its a property of a Decoder
    max_r = 10

    def __init__(self, mel_dim, encoder_output_dim,
                 prenet_dims=[256, 128], prenet_dropout=0.5,
                 attention_dim=256, attention_rnn_units=256,
                 decoder_rnn_units=256, decoder_rnn_layers=2,
                 max_decoder_steps=1000, stop_threshold=0.5):
        super(Decoder, self).__init__()

        self.register_buffer("r", torch.tensor(1, dtype=torch.int))

        self.mel_dim = mel_dim
        self.attention_context_dim = attention_context_dim = encoder_output_dim
        self.attention_rnn_units = attention_rnn_units
        self.decoder_rnn_units = decoder_rnn_units
        self.max_decoder_steps = max_decoder_steps
        self.stop_threshold = stop_threshold

        # Prenet
        self.prenet = Prenet(mel_dim, prenet_dims, prenet_dropout)

        # Attention RNN
        # (prenet_out + attention context) = attention_rnn_in -> attention_rnn_out
        self.attention_rnn = AttentionWrapper(
            nn.GRUCell(prenet_dims[-1] + attention_context_dim, attention_rnn_units),
            BahdanauAttention(attention_rnn_units, attention_dim)
        )
        # Process encoder_output as attention key
        self.memory_layer = nn.Linear(encoder_output_dim, attention_dim, bias=False)

        # Decoder RNN
        # (attention_rnn_out + attention context) -> decoder_rnn_in
        self.project_to_decoder_in = nn.Linear(attention_rnn_units + attention_context_dim, decoder_rnn_units)
        # decoder rnn
        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(decoder_rnn_units, decoder_rnn_units) for _ in range(decoder_rnn_layers)])

        # Project to mel
        self.mel_proj = nn.Linear(decoder_rnn_units + attention_context_dim, mel_dim * self.max_r)

        # Stop token prediction
        self.stop_proj = nn.Linear(decoder_rnn_units + attention_context_dim, 1)

    def forward(self, encoder_outputs, inputs=None, memory_lengths=None):
        """
        Decoder forward step.

        If decoder inputs are not given (e.g., at testing time), greedy decoding is adapted.

        Args:
            encoder_outputs: Encoder outputs. (B, T_encoder, dim)
            inputs: Decoder inputs (i.e., mel-spectrogram).
                    If None (at eval-time), previous decoder outputs are used as decoder inputs.
            memory_lengths: Encoder output (memory) lengths. If not None, used for attention masking.

        Returns:
            mel_outputs: mel outputs from the decoder.
            stop_tokens: stop token outputs from the decoder.
            attn_scores: sequence of attention weights from the decoder.
        """
        B = encoder_outputs.size(0)

        # Get processed memory for attention key
        processed_memory = self.memory_layer(encoder_outputs)
        if memory_lengths is not None:
            mask = get_mask_from_lengths(processed_memory, memory_lengths)
        else:
            mask = None

        # Run greedy decoding if inputs is None
        greedy = inputs is None

        # Time first: (B, T, mel_dim) -> (T, B, mel_dim)
        if inputs is not None:
            inputs = inputs.transpose(0, 1)
            T_decoder = inputs.size(0)

        # <GO> frames
        initial_input = encoder_outputs.data.new(B, self.mel_dim).zero_()

        # Init decoder states
        attention_rnn_hidden = encoder_outputs.data.new(B, self.attention_rnn_units).zero_()
        decoder_rnn_hiddens = [encoder_outputs.data.new(B, self.decoder_rnn_units).zero_()
                               for _ in range(len(self.decoder_rnns))]
        attention_context = encoder_outputs.data.new(B, self.attention_context_dim).zero_()

        # To save the result
        mel_outputs, attn_scores, stop_tokens = [], [], []

        # Run the decoder loop
        t = 0
        current_input = initial_input
        while True:
            if t > 0:
                current_input = mel_outputs[-1][:, -1, :] if greedy else inputs[t - 1]
            t += self.r

            # Prenet
            current_input = self.prenet(current_input)

            # Attention RNN
            attention_rnn_hidden, attention_context, attention_score = self.attention_rnn(
                current_input, attention_context, attention_rnn_hidden,
                encoder_outputs, processed_memory=processed_memory, mask=mask)

            # Concat RNN output and attention context vector
            decoder_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, attention_context), -1))

            # Pass through the decoder RNNs
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    decoder_input, decoder_rnn_hiddens[idx])
                # Residual connectinon
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input

            # Contact RNN output and context vector to form projection input
            proj_input = torch.cat((decoder_input, attention_context), -1)

            # Project to mel
            # (B, mel_dim*max_r) -> (B, max_r, mel_dim) -> (B, r, mel_dim)
            output = self.mel_proj(proj_input)
            output = output.view(B, -1, self.mel_dim)[:, :self.r, :]

            # Stop token prediction
            stop = self.stop_proj(proj_input)
            stop = torch.sigmoid(stop)

            # Store predictions
            mel_outputs.append(output)
            attn_scores.append(attention_score.unsqueeze(1))
            stop_tokens.extend([stop] * self.r)

            if greedy:
                if stop > self.stop_threshold:
                    break
                elif t > 1 and is_end_of_frames(output):
                    print("Warning: End with low power.")
                    break
                elif t > self.max_decoder_steps:
                    print("Warning: Reached max decoder steps.")
                    break
            else:
                if t >= T_decoder:
                    break

        # To tensor
        mel_outputs = torch.cat(mel_outputs, dim=1) # (B, T_decoder, mel_dim)
        attn_scores = torch.cat(attn_scores, dim=1) # (B, T_decoder/r, T_encoder)
        stop_tokens = torch.cat(stop_tokens, dim=1) # (B, T_decoder)

        # Validation check
        assert greedy or mel_outputs.size(1) == T_decoder

        return mel_outputs, stop_tokens, attn_scores


def is_end_of_frames(output, eps=-3.4):
    return (output.data <= eps).all()


class Tacotron(nn.Module):
    def __init__(self, model_cfg, n_vocab, embed_dim=256, mel_dim=80, linear_dim=1025,
                 max_decoder_steps=1000, stop_threshold=0.5):
        super(Tacotron, self).__init__()

        self.mel_dim = mel_dim
        self.linear_dim = linear_dim

        # Embedding
        self.embedding = nn.Embedding(n_vocab, embed_dim)
        # Trying smaller std
        self.embedding.weight.data.normal_(0, 0.3)

        # Encoder
        encoder_cfg = model_cfg["encoder"]
        encoder_out_dim = encoder_cfg["rnn_units"] * 2
        self.encoder = Encoder(embed_dim, **encoder_cfg)

        # Decoder
        decoder_cfg = model_cfg["decoder"]
        self.decoder = Decoder(mel_dim, encoder_out_dim, **decoder_cfg,
            max_decoder_steps=max_decoder_steps, stop_threshold=stop_threshold)

        # Postnet
        postnet_cfg = model_cfg["postnet"]
        postnet_out_dim = postnet_cfg["rnn_units"] * 2
        assert mel_dim == postnet_cfg["proj_channels"][-1]
        self.postnet = CBHG(mel_dim, **postnet_cfg)
        self.last_linear = nn.Linear(postnet_out_dim, linear_dim)

    @property
    def r(self):
        return self.decoder.r.item()

    @r.setter
    def r(self, value):
        self.decoder.r = self.decoder.r.new_tensor(value, requires_grad=False)

    def parse_data_batch(self, batch):
        """Parse data batch to form inputs and targets for model training/evaluating
        """
        # use same device as parameters
        device = next(self.parameters()).device

        text, text_length, mel, stop, _ = batch
        text = text.to(device).long()
        text_length = text_length.to(device).long()
        mel = mel.to(device).float()
        stop = stop.to(device).float()

        return (text, mel, text_length), (mel, mel, stop)

    def forward(self, inputs):
        inputs, targets, input_lengths = inputs

        B = inputs.size(0)

        # (B, T)
        inputs = self.embedding(inputs)

        # (B, T, embed_dim)
        encoder_outputs = self.encoder(inputs)

        # (B, T, mel_dim)
        mel_outputs, stop_tokens, alignments = self.decoder(
            encoder_outputs, targets, memory_lengths=input_lengths)

        # Postnet processing
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, stop_tokens, alignments

    def inference(self, inputs):
        # Only text inputs
        inputs = inputs, None, None
        return self.forward(inputs)


class TacotronLoss(nn.Module):
    def __init__(self):
        super(TacotronLoss, self).__init__()

    def forward(self, predicts, targets):
        mel_target, linear_target, stop_target = targets
        mel_target.requires_grad = False
        linear_target.requires_grad = False
        stop_target.requires_grad = False

        mel_predict, linear_predict, stop_predict, _ = predicts

        mel_loss = nn.MSELoss()(mel_predict, mel_target)
        linear_loss = nn.MSELoss()(linear_predict, linear_target)
        stop_loss = nn.BCELoss()(stop_predict, stop_target)

        return mel_loss + linear_loss + stop_loss
