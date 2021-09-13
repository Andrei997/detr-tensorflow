import tensorflow as tf
import argparse
import os


def training_config_parser():
    """Training config class can be overide using the script arguments"""
    parser = argparse.ArgumentParser()

    # xml_annotation_path, csv_annotation_path, oxford_annotations_path, oxford_images_path
    # Dataset info
    parser.add_argument(
        "--xml_annotation_path", type=str, required=True
    )
    parser.add_argument(
        "--csv_annotation_path", type=str, required=True
    )
    parser.add_argument(
        "--oxford_annotations_path", type=str, required=True
    )
    parser.add_argument(
        "--oxford_images_path", type=str, required=True
    )
    
    parser.add_argument(
        "--background_class",
        type=int,
        required=False,
        default=0,
        help="Default background class",
    )

    # What to train
    parser.add_argument(
        "--train_backbone",
        action="store_true",
        required=False,
        default=False,
        help="Train backbone",
    )
    parser.add_argument(
        "--train_transformers",
        action="store_true",
        required=False,
        default=False,
        help="Train transformers",
    )
    parser.add_argument(
        "--train_nlayers",
        action="store_true",
        required=False,
        default=False,
        help="Train new layers",
    )

    # How to train
    parser.add_argument(
        "--finetuning",
        default=False,
        required=False,
        action="store_true",
        help="Load the model weight before to train",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        default=1,
        help="Batch size to use to train the model",
    )
    parser.add_argument(
        "--gradient_norm_clipping",
        type=float,
        required=False,
        default=0.1,
        help="Gradient norm clipping",
    )
    parser.add_argument(
        "--target_batch",
        type=int,
        required=False,
        default=None,
        help="When running on a single GPU, aggretate the gradient before to apply.",
    )

    # Learning rate
    parser.add_argument(
        "--backbone_lr", type=bool, required=False, default=1e-5, help="Train backbone"
    )
    parser.add_argument(
        "--transformers_lr",
        type=bool,
        required=False,
        default=1e-4,
        help="Train transformers",
    )
    parser.add_argument(
        "--nlayers_lr", type=bool, required=False, default=1e-4, help="Train new layers"
    )

    return parser


class TrainingConfig:
    def __init__(self):
        
        self.nlayers = []

        # Dataset info
        # xml_annotation_path, csv_annotation_path, oxford_annotations_path, oxford_images_path
        (
            self.xml_annotation_path,
            self.csv_annotation_path,
            self.oxford_annotations_path,
            self.oxford_images_path,
        ) = (None, None, None, None)
        self.data = DataConfig(
            self.xml_annotation_path,
            self.csv_annotation_path,
            self.oxford_annotations_path,
            self.oxford_images_path,
        )
        self.background_class = 0
        self.image_size = 512, 512

        # What to train
        self.train_backbone = False
        self.train_transformers = False
        self.train_nlayers = False

        # How to train
        self.finetuning = False
        self.batch_size = 1
        self.gradient_norm_clipping = 0.1
        # Batch aggregate before to backprop
        self.target_batch = 1

        # Learning rate
        # Set as tf.Variable so that the variable can be update during the training while
        # keeping the same graph
        self.backbone_lr = tf.Variable(1e-5)
        self.transformers_lr = tf.Variable(1e-4)
        self.nlayers_lr = tf.Variable(1e-4)

        # Training progress
        self.global_step = 0
        self.log = False

    def update_from_args(self, args):
        """ Update the training config from args
        """
        args = vars(args)
        for key in args:
            if isinstance(getattr(self, key), tf.Variable):
                getattr(self, key).assign(args[key])
            else:
                setattr(self, key, args[key])
        
        # Set the config on the data class
        self.data = DataConfig(
            self.xml_annotation_path,
            self.csv_annotation_path,
            self.oxford_annotations_path,
            self.oxford_images_path,
        )


class DataConfig:
    def __init__(
        self,
        xml_annotation_path,
        csv_annotation_path,
        oxford_annotations_path,
        oxford_images_path,
    ):
        self.xml_annotation_path = xml_annotation_path
        self.csv_annotation_path = csv_annotation_path
        self.oxford_annotations_path = oxford_annotations_path
        self.oxford_images_path = oxford_images_path


if __name__ == "__main__":
    args = training_config_parser()
    config = TrainingConfig()
    config.update_from_args(args)
