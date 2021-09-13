import numpy as np
import tensorflow as tf
import imageio
import cv2
import xml.etree.ElementTree as ET
import csv
import os

from random import sample, shuffle

# import transformation
from detr_tf.data import processing

OXOFORD_CLASS_NAME = ["PET"]


def get_face_bbox_oxford(xml_path):
    tree = ET.parse(xml_path)
    object_annotation = tree.find("object")
    bndbox_annotation = object_annotation.find("bndbox")
    xmin = float(bndbox_annotation.find("xmin").text)
    ymin = float(bndbox_annotation.find("ymin").text)
    xmax = float(bndbox_annotation.find("xmax").text)
    ymax = float(bndbox_annotation.find("ymax").text)
    return (xmin, ymin, xmax, ymax)


def get_csv_annotations(csv_path):
    csv_annotations = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                frame_id = int(row[1])
                if frame_id not in csv_annotations:
                    csv_annotations[frame_id] = []
                csv_annotations[frame_id].append(
                    {"points": row[3], "id": row[0], "occluded": row[2]}
                )
    return csv_annotations


def get_annotations_CVAT(xml_path, csv_path, oxford_annotations_path):

    tree = ET.parse(xml_path)
    image_annotations = tree.findall("image")

    csv_annotations = get_csv_annotations(csv_path)

    annotations = {}

    for image_annotation in image_annotations:

        image_name = image_annotation.get("name")
        image_width = int(image_annotation.get("width"))
        image_height = int(image_annotation.get("height"))
        image_id = int(image_annotation.get("id"))

        try:
            xml_annot = os.path.join(
                oxford_annotations_path, image_name.replace(".jpg", ".xml")
            )
            face_bbox = get_face_bbox_oxford(xml_annot)
        except:
            continue

        keypoints = []

        try:
            image_keypoints_annotations = csv_annotations[image_id]
        except:
            continue

        if len(image_keypoints_annotations) != 5:
            continue

        for keypoint_annotation in image_keypoints_annotations:

            points = keypoint_annotation["points"]
            points = points.split(",")
            points = [float(point) for point in points]

            visible = 0 if keypoint_annotation["occluded"] == "True" else 1

            keypoint_id = int(keypoint_annotation["id"])

            keypoints.append([*points[:2], visible, keypoint_id])

        new_keypoints = sorted(keypoints, key=lambda x: x[3])

        annotations[image_name] = [
            image_name,
            [image_width, image_height],
            face_bbox,
            new_keypoints,
        ]

    return annotations


def load_image(image_name, annotations=None, images_path=None, config=None):

    annotation = annotations[image_name.decode("utf-8")]
    image_path, image_size, bbox, keypoints = annotation

    full_image_path = os.path.join(images_path, image_path)
    image = imageio.imread(full_image_path)

    x1, y1, x2, y2 = bbox
    width, height = x2 - x1, y2 - y1

    x_center = x1 + (width / 2)
    y_center = y1 + (height / 2)

    x_center = x_center / image_size[0]
    y_center = y_center / image_size[1]
    width = width / image_size[0]
    height = height / image_size[1]

    kpts = []
    for keypoint in keypoints:
        k_x, k_y = keypoint[:2]
        k_x = k_x / image_size[0]
        k_y = k_y / image_size[1]
        kpts.append(k_x)
        kpts.append(k_y)

    image = cv2.resize(image, (512, 512))
    image = tf.image.per_image_standardization(image)
    # image = image.astype(np.float32)

    return (
        image,
        [[0]],
        np.array([[x_center, y_center, width, height]], dtype=np.float32),
        np.array([kpts], dtype=np.float32),
    )


def load_oxford_dataset(
    config,
    xml_annotation_path,
    csv_annotation_path,
    oxford_annotations_path,
    oxford_images_path,
):
    max_id = 0
    class_names = ["N/A"] * (max_id + 2)
    class_names[0] = "pet"
    class_names[-1] = "back"
    config.background_class = max_id + 1

    # load all images from dir
    all_annotations = get_annotations_CVAT(
        xml_annotation_path, csv_annotation_path, oxford_annotations_path
    )

    # create dataset from image dirs
    images_tensor = tf.convert_to_tensor(list(all_annotations.keys()), dtype=tf.string)
    dataset = tf.data.Dataset.from_tensor_slices(images_tensor)

    outputs_types = (tf.float32, tf.int64, tf.float32, tf.float32)

    dataset = dataset.map(
        lambda image_id: processing.numpy_fc(
            image_id,
            load_image,
            outputs_types,
            annotations=all_annotations,
            images_path=oxford_images_path,
            config=config
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )

    dataset = dataset.map(
        processing.pad_labels, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(config.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(32)

    return dataset, class_names
