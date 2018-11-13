import base64
import glob
import json
import logging
import subprocess
import sys
import tarfile
import traceback
import uuid
import wave
from os import unlink, environ, makedirs
from os.path import basename
from pickle import load
from random import randint
from shutil import copy2, rmtree
from urllib.request import urlretrieve

import mxnet as mx
import numpy as np
from mxnet import autograd, nd, gluon
from mxnet.gluon import Trainer
from mxnet.gluon.loss import SoftmaxCrossEntropyLoss
from mxnet.gluon.nn import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Sequential
from mxnet.initializer import Xavier


def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])


install("opencv-python")
install("argparse")
install("msgpack")
install("librosa")
install("tqdm")

import librosa
import numpy as np
from tqdm import tqdm
import os

environ["PATH"] += ":/tmp"

# from pydub import AudioSegment

logging.basicConfig(level=logging.INFO)

voices = ['five', 'happy', 'two', 'bed', 'one', 'six', 'zero', 'four', 'three', 'cat', 'eight', 'nine', 'seven']


def train(hyperparameters, channel_input_dirs, num_gpus, hosts):
    batch_size = hyperparameters.get("batch_size", 64)
    epochs = hyperparameters.get("epochs", 3)

    mx.random.seed(42)

    training_dir = channel_input_dirs['training']

    with open("{}/train/data.p".format(training_dir), "rb") as pickle:
        train_nd = load(pickle)
    with open("{}/validation/data.p".format(training_dir), "rb") as pickle:
        validation_nd = load(pickle)

    train_data = gluon.data.DataLoader(train_nd, batch_size, shuffle=True)
    validation_data = gluon.data.DataLoader(validation_nd, batch_size, shuffle=True)

    net = Sequential()
    # http: // gluon.mxnet.io / chapter03_deep - neural - networks / plumbing.html  # What's-the-deal-with-name_scope()?
    with net.name_scope():
        net.add(Conv2D(channels=32, kernel_size=(3, 3), padding=0, activation="relu"))
        net.add(Conv2D(channels=32, kernel_size=(3, 3), padding=0, activation="relu"))
        net.add(MaxPool2D(pool_size=(2, 2)))
        net.add(Dropout(.25))
        net.add(Flatten())
        net.add(Dense(8))

    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()

    # Also known as Glorot
    net.collect_params().initialize(Xavier(magnitude=2.24), ctx=ctx)

    loss = SoftmaxCrossEntropyLoss()

    # kvstore type for multi - gpu and distributed training.
    if len(hosts) == 1:
        kvstore = "device" if num_gpus > 0 else "local"
    else:
        kvstore = "dist_device_sync'" if num_gpus > 0 else "dist_sync"

    trainer = Trainer(net.collect_params(), optimizer="adam", kvstore=kvstore)

    smoothing_constant = .01

    for e in range(epochs):
        moving_loss = 0
        for i, (data, label) in enumerate(train_data):
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)
            with autograd.record():
                output = net(data)
                loss_result = loss(output, label)
            loss_result.backward()
            trainer.step(batch_size)

            curr_loss = nd.mean(loss_result).asscalar()
            moving_loss = (curr_loss if ((i == 0) and (e == 0))
                           else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        validation_accuracy = measure_performance(net, ctx, validation_data)
        train_accuracy = measure_performance(net, ctx, train_data)
        print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" % (e, moving_loss, train_accuracy, validation_accuracy))

    return net


def measure_performance(model, ctx, data_iter):
    acc = mx.metric.Accuracy()
    for _, (data, labels) in enumerate(data_iter):
        data = data.as_in_context(ctx)
        labels = labels.as_in_context(ctx)
        output = model(data)
        predictions = nd.argmax(output, axis=1)
        acc.update(preds=predictions, labels=labels)
    return acc.get()[1]


def save(net, model_dir):
    y = net(mx.sym.var("data"))
    y.save("{}/model.json".format(model_dir))
    net.collect_params().save("{}/model.params".format(model_dir))


def model_fn(model_dir):
    with open("{}/model.json".format(model_dir), "r") as model_file:
        model_json = model_file.read()
    outputs = mx.sym.load_json(model_json)
    inputs = mx.sym.var("data")
    param_dict = gluon.ParameterDict("model_")
    net = gluon.SymbolBlock(outputs, inputs, param_dict)
    # We will serve the model on CPU
    net.load_params("{}/model.params".format(model_dir), ctx=mx.cpu())
    return net


# noinspection PyUnusedLocal
def transform_fn(model, input_data, content_type, accept):
    try:
        if content_type == "audio/wav":
            np_arr = wav2mfcc(input_data)
            mx_arr = mx.nd.array(np_arr)
            logging.info(mx_arr.shape)
            logging.info(mx_arr)
            response = model(mx_arr)
            response = nd.argmax(response, axis=1) \
                .asnumpy() \
                .astype(np.int) \
                .ravel() \
                .tolist()[0]
            return json.dumps(voices[response]), accept
        elif content_type == "application/json":
            json_array = json.loads(input_data, encoding="utf-8")
            np_arrs = [img2arr(file) for file in json_array]
            # noinspection PyUnresolvedReferences
            np_arr = np.concatenate(np_arrs)
            nd_arr = nd.array(np_arr)
            response = model(nd_arr)
            response = nd.argmax(response, axis=1) \
                .asnumpy() \
                .astype(np.int) \
                .ravel() \
                .tolist()
            return json.dumps([voices[idx] for idx in response]), accept
        else:
            raise ValueError("Cannot decode input to the prediction.")
    except Exception as ex:
        logging.error(ex)
        logging.error(traceback.format_exc())


# Handy function to convert wav2mfcc
def wav2mfcc(file_path, max_len=11):
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    wave = wave[::3]
    mfcc = librosa.feature.mfcc(wave, sr=16000)

    # If maximum length exceeds mfcc lengths then pad the remaining ones
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]
    
    return mfcc
