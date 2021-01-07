# DETR : End-to-End Object Detection with Transformers (Tensorflow)

Tensorflow implementation of DETR : Object Detection with Transformers, including code for inference, training, and finetuning. DETR is a promising model that brings widely adopted transformers to vision models. We believe that models based on convolution and transformers will soon become the default choice for most practitioners because of the simplicity of the training procedure: NMS and anchors free! Therefore this repository is a step toward making this type of architecture widely available. 


<b>About this implementation:</b> https://arxiv.org/pdf/2005.12872.pdf <br>
<b>Torch implementation: https://github.com/facebookresearch/detr</b>

<img src="images/detr-figure.png"></img>

<b>About this implementation:</b> This repository includes codes to run an inference with the original model's weights (based on the PyTorch weights), to train the model from scratch (multi-GPU training support coming soon) as well as examples to finetune the model on your dataset. Unlike the PyTorch implementation, the training uses fixed image sizes and a standard Adam optimizer with gradient norm clipping.

Additionally, our logging system is based on https://www.wandb.com/ so that you can get a great visualization of your model performance!

- Checkout our logging board here: https://wandb.ai/thibault-neveu/detr-tensorflow-log
- Also we released the following wandb report to help you getting started with the repository and the logs
    - link


<img src="images/wandb_logging.png"></img>



## Datasets

This repository currently support three dataset format : **COCO**, **VOC** and **Tensorflow Object detection csv**. The easiest way to get started is to setup your dataset based on one of theses format. Along with the datasets, we provide code exmaple to finetune your model.

Finally, we provide a jupyter notebook to help you understand how to load a dataset, setup a custom dataset and finetune your model.

<img src="images/datasetsupport.png"></img>

## Tutorials

To get started with the repository you can check the following Jupyter notebooks:

- ✍ How to load a dataset.ipynb
- ✍ DETR Tensorflow - Finetuning tutorial.ipynb
- ✍ DETR Tensorflow - How to setup a custom dataset.ipynb

As well as the logging board on wandb https://wandb.ai/thibault-neveu/detr-tensorflow-log with the following report:

- 🚀 Finetuning DETR on Tensorflow - A step by step guide


## Install

The code has been currently tested with Tensorflow 2.3.0 and python 3.7. The following dependencies are required.


```
wandb
matplotlib
numpy
pycocotools
scikit-image
imageio
pandas
```

```
pip install -r requirements.txt
```


## Evaluation :

Run the following to evaluate the model using the pre-trained weights:


```
python eval.py --datadir /path/to/coco
```

Outputs:

```
       |  all  |  .50  |  .55  |  .60  |  .65  |  .70  |  .75  |  .80  |  .85  |  .90  |  .95  |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+
   box | 36.53 | 55.38 | 53.13 | 50.46 | 47.11 | 43.07 | 38.11 | 32.10 | 25.01 | 16.20 |  4.77 |
  mask |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |  0.00 |
-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+-------+

```

The result is not the same as reported in the paper because the evaluation is run on the <b>original image size</b> and not on the larger images. The actual implementation resizes the image so that the shorter side is at least 800pixels and the longer side at most 1333.


## Finetune on your dataset

To fine-tune the model on a new dataset, we must remove the last layers that predict the box class and positions.

```python
# Load the pretrained model
detr = get_detr_model(config, include_top=False, nb_class=3, weights="detr", num_decoder_layers=6, num_encoder_layers=6)
detr.summary()

# Load your dataset
train_dt, class_names = load_tfcsv_dataset("train", config.batch_size, config, augmentation=True)

# Setup the optimziers and the trainable variables
optimzers = setup_optimizers(detr, config

# Train the model
training.fit(detr, train_dt, optimzers, config, epoch_nb, class_names)
```
The following commands gives an example to finetune the model on a new dataset (VOC) and (The Hard hat dataset) with a real ```batch_size``` of 8 and a virtual ```target_batch``` size (gradient aggregate) of 32. ```--log``` is used for logging the training into wandb. 
```
python finetune_voc.py --datadir /path/to/VOCdevkit/VOC2012 --batch_size 8 --target_batch 32  --log
```
```
python  finetune_hardhat.py --datadir /home/thibault/data/hardhat/ --batch_size 8 --target_batch 32 --log
```





## Training on COCO

(Multi GPU training comming soon)

```
python train_coco.py --datadir /path/to/COCO --batch_size 8  --target_batch 32 --log
```


## Inference

Here is an example of running an inference with the model on your webcam.

```
python webcam_inference.py 
```

<img src="images/webcam_detr.png" width="400"></img>


## Acknowledgement

The pretrained weights of this models are originaly provide from the Facebook repository https://github.com/facebookresearch/detr and made avaiable in tensorflow in this repository: https://github.com/Leonardo-Blanger/detr_tensorflow
