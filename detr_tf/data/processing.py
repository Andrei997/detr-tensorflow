import tensorflow as tf
import numpy as np



def normalized_images(image, config):
    """ Normalized images. torch_resnet is used on finetuning
    since the weights are based on the  original paper training code
    from pytorch. tf_resnet is used when training from scratch with a
    resnet50 traine don tensorflow.
    """
    if config.normalized_method == "torch_resnet":
        channel_avg = np.array([0.485, 0.456, 0.406])
        channel_std = np.array([0.229, 0.224, 0.225])
        image = (image / 255.0 - channel_avg) / channel_std
        return image.astype(np.float32)
    elif config.normalized_method == "tf_resnet":
        mean = [103.939, 116.779, 123.68]
        image = image[..., ::-1]
        image = image - mean
        return image.astype(np.float32)
    else:
        raise Exception("Can't handler this normalized method")


def numpy_fc(idx, fc, outputs_types, **params):
    """
    Call a numpy function on each given ID (`idx`) and load the associated image and labels (bbbox and cls)
    """
    def _np_function(_idx):
        return fc(_idx, **params)
    return tf.numpy_function(_np_function, [idx], outputs_types)


# TODO: add padding to keypoint as well
def pad_labels(images: tf.Tensor, t_class: tf.Tensor, t_bbox: tf.Tensor, t_kpts: tf.Tensor):
    """ Pad the bbox by adding [0, 0, 0, 0] at the end
    and one header to indicate how many bbox are set.
    Do the same with the labels. 
    """
    
    # this is the number of boxes
    nb_bbox = tf.shape(t_bbox)[0]
    # 1
    
    bbox_header = tf.expand_dims(nb_bbox, axis=0)
    # [1]
    
    bbox_header = tf.expand_dims(bbox_header, axis=0)
    # [[1]]
    
    bbox_header = tf.pad(bbox_header, [[0, 0], [0, 3]])
    # [[1, 0, 0, 0]]
    
    cls_header = tf.constant([[0]], dtype=tf.int64)
    bbox_header = tf.cast(bbox_header, tf.float32)
    kpts_header = tf.constant([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.float32)

    # Padd bbox and class
    t_class = tf.pad(t_class, [[0, 100 - 1 - nb_bbox], [0, 0]], mode='CONSTANT', constant_values=0)
    t_bbox = tf.pad(t_bbox, [[0, 100 - 1 - nb_bbox], [0, 0]], mode='CONSTANT', constant_values=0)
    t_kpts = tf.pad(t_kpts, [[0, 100 - 1 - nb_bbox], [0, 0]], mode='CONSTANT', constant_values=0)

    t_class = tf.concat([cls_header, t_class], axis=0)
    t_bbox = tf.concat([bbox_header, t_bbox], axis=0)
    t_kpts = tf.concat([kpts_header, t_kpts], axis=0)

    return images, t_class, t_bbox, t_kpts

