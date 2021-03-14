import torch


def get_mask_from_lengths(lengths, total_length):
    # use total_length of a batch for data parallelism
    device = lengths.device
    ids = torch.arange(0, total_length, out=torch.cuda.LongTensor(total_length)).to(device)
    mask = (ids < lengths.unsqueeze(1)).bool().to(device)
    return mask


def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)


_output_ref = None
_replicas_ref = None

def data_parallel_workaround(model, *input, output_device=None):
    """Pytorch parallel processing

    Code from:
        https://github.com/CorentinJ/Real-Time-Voice-Cloning/tree/b5ba6d0371882dbab595c48deb2ff17896547de7/synthesizer
    """
    global _output_ref
    global _replicas_ref

    device_ids = list(range(torch.cuda.device_count()))

    if output_device is None:
        output_device = device_ids[0]

    replicas = torch.nn.parallel.replicate(model, device_ids)

    # input.shape = (num_args, batch, ...)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    # inputs.shape = (num_gpus, num_args, batch/num_gpus, ...)

    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)

    y_hat = torch.nn.parallel.gather(outputs, output_device)

    _output_ref = outputs
    _replicas_ref = replicas

    return y_hat
