""" Example on how to train on OXFORD from scratch
"""

import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
import os

from detr_tf.networks.detr import get_detr_model
from detr_tf.data.oxford import load_oxford_dataset
from detr_tf.optimizers import setup_optimizers
from detr_tf.training_config import TrainingConfig, training_config_parser
from detr_tf import training


import time


# TODO: re-implement get_detr_model ?
def build_model(config):
    """Build the model with the pretrained weights. In this example
    we do not add new layers since the pretrained model is already trained on coco.
    See examples/finetuning_voc.py to add new layers.
    """
    # Load detr model without weight.
    # Use the tensorflow backbone with the imagenet weights
    detr = get_detr_model(config, include_top=True, weights=None, tf_backbone=True)
    detr.summary()
    return detr


def run_training(config):

    # Load the model with the new layers to finetune
    detr = build_model(config)

    # Load the training and validation dataset
    # TODO: replace these with local paths
    train_dt, oxford_class_names = load_oxford_dataset(config)

    # TODO: setup validation dataset later on ...
    # valid_dt, _ = load_oxford_dataset(config, 1, augmentation=False, img_dir="val2017", ann_fil="annotations/instances_val2017.json")

    # Train the backbone and the transformers
    # Check the training_config file for the other hyperparameters
    config.train_backbone = True
    config.train_transformers = True

    # Setup the optimziers and the trainable variables
    # TODO: check if optimizers are used for the keypoints dense layer
    optimzers = setup_optimizers(detr, config)

    # Run the training for 100 epochs
    # TODO: enable eval
    for epoch_nb in range(10):
        # training.eval(detr, valid_dt, config, coco_class_names, evaluation_step=200)
        training.fit(detr, train_dt, optimzers, config, epoch_nb, oxford_class_names)

    # TODO: save model


if __name__ == "__main__":

    physical_devices = tf.config.list_physical_devices("GPU")
    if len(physical_devices) == 1:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    config = TrainingConfig()
    args = training_config_parser().parse_args()
    config.update_from_args(args)

    # Run training
    run_training(config)
