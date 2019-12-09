"""An extremely ad-hoc test harness for basic sanity checking.
Make sure things don't revert /too/ badly when I commit things.

To add a test, just add a new function annotated with the decorator `@test`.
"""

import traceback
from termcolor import colored
from collections import defaultdict
import argparse
import os
import sys

import torch
import torch.nn as nn

sys.path.append('waveglow/')

from ref_model import ConvBlock, RnnBlock, ReferenceEncoder
from train import load_model
from convert_model import update_model
from denoiser import Denoiser
from torch.utils.data import DataLoader
from train import prepare_dataloaders

from hparams import create_hparams

### DEFINITIONS ###
test_registry = {}

def test(fn):
    test_registry[fn.__name__] = fn
    return fn

hparams = create_hparams()

### TESTS ###
@test
def _test_conv_block(test_params):
    """Just feed some numbers through a ConvBlock and make sure it doesn't
    crash.
    """
    input_shape = (80, 200)
    # leading 1s for batch size and number of channels (I think? Confirm.)

    input_tensor = torch.randn(1, 1, *input_shape)
    cb = ConvBlock(3, 2, [32, 32, 64])
    if test_params['cuda']:
        input_tensor = input_tensor.cuda()
        cb = cb.cuda()

    output = cb(input_tensor)

@test
def _test_rnn_block(test_params):
    """Just feed some numbers through an RnnBlock and make sure it doesn't
    crash.
    """
    LR_reduced = 36  # an arbitrary number for LR / 64
    dR_reduced = 24  # an arbitrary number for dR / 64
    input_dim = 128 * dR_reduced
    cnn_channels = 128
    embedding_dim = 128
    batch_size = 32

    input_tensor = torch.randn(batch_size, cnn_channels, dR_reduced, LR_reduced)
    rb = RnnBlock(input_dim, hparams.ref_embedding_dim, batch_size)
    if test_params['cuda']:
        input_tensor = input_tensor.cuda()
        rb = rb.cuda()

    output = rb(input_tensor)
    assert output.shape == torch.Size([batch_size, embedding_dim])

@test
def _test_reference_encoder(test_params):
    input_spec = torch.randn(hparams.batch_size, 80, 200)
    re = ReferenceEncoder(hparams)
    if test_params['cuda']:
        input_spec = input_spec.cuda()
        re = re.cuda()

    embedding = re(input_spec)
    assert embedding.shape == torch.Size([hparams.batch_size, 128])

# @test
def _test_reference_encoder_real_spectrogram():
    pass

@test
def _test_inference(test_params):
    # checkpoint_path = "tacotron2_statedict.pt"
    checkpoint_path = 'new-test-outdir/checkpoint_27000'
    checkpoint_name = os.path.basename(checkpoint_path)
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval()

    waveglow_path = 'waveglow_256channels_ljs_v2.pt'
    waveglow = update_model(torch.load(waveglow_path)['model'])
    waveglow.cuda().eval()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    train_loader, valset, collate_fn = prepare_dataloaders(hparams)

    val_loader = DataLoader(valset, sampler=None, num_workers=1,
                            shuffle=False, batch_size=4,
                            pin_memory=False, collate_fn=collate_fn)
    x, y = model.parse_batch(next(iter(val_loader)))

    embedded_inputs = model.embedding(x[0]).transpose(1, 2)
    encoder_outputs = model.encoder.inference(embedded_inputs)

    length = encoder_outputs.shape[1]
    # reshaping encoder outputs:
    # need shape (N, L, ref_embedding_dim)
    ref_encoder_outputs = model.ref_encoder(y[0]).unsqueeze(1).repeat(1, length, 1)

    mel_outputs, gate_outputs, alignments = model.decoder.inference(
                torch.cat((encoder_outputs[1,:,:], ref_encoder_outputs[1,:,:]), -1))


### RUN THE TESTS ###
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true')
    namespace = parser.parse_args(sys.argv[1:])

    test_params = defaultdict(lambda: False)
    test_params.update(vars(namespace))

    for name, fn in test_registry.items():
        try:
            fn(test_params)
            print(colored("Passed test {}".format(name), 'green'))
        except Exception :  # Here, pokemon exception handling is a-ok
            print(colored("Tests failed in the execution of {}\n\n".format(name), 'red'),
                traceback.format_exc())
            exit(1)
