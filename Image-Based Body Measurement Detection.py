# Import libraries
import os
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import ops as utils_ops
from object_detection.utils.label_map_util import get_label_map_dict

model_url = 'https://universe.roboflow.com/ds/8J5858NJP4?key=qMvFC4AHcw'
configs = config_util.get_configs_from_pipeline_file(os.path.join(model_url, 'pipeline.config'))
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(model_dir, 'checkpoint', 'ckpt-0')).expect_partial()


dataset_dir = 'https://drive.google.com/drive/folders/10fqLYQ9JSOAupXiXd6viKyZQX-u0M39K?usp=drive_link'
dataset = tf.data.TFRecordDataset(os.path.join(dataset_dir, 'test.record'))


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



# Define the approximate conversion factor (e.g., pixels to centimeters)
conversion_factor = 2.54

for raw_record in dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    image = tf.image.decode_jpeg(example.features.feature['image/encoded'].bytes_list.value[0], channels=3)
    image_np = image.numpy()

    input_tensor = tf.convert_to_tensor([np.asarray(image_np)])
    detections = detect_fn(input_tensor)

    # Access bounding box coordinates and classes
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(np.int32)

    # Assuming class 1 corresponds to the person in your dataset
    person_boxes = boxes[classes == 1]

    for box in person_boxes:
        ymin, xmin, ymax, xmax = box
        height = (ymax - ymin) * conversion_factor
        width = (xmax - xmin) * conversion_factor

        # Calculating circumferences, width and length based on bounding box dimensions and giving Simplified assumption
        chest_circumference = height
        waist_circumference = height
        hip_circumference = height
        Shoulder_width =  width
        Sleeve_length = height
        Shirt_length = height
        neck_circumference = width 
        Bicep_circumference = width 
        wrist_circumference = width  


        # Print or store the calculated circumferences
        print("Chest Circumference:", chest_circumference)
        print("Waist Circumference:", waist_circumference)
        print("Hip Circumference:", hip_circumference)
        print("Shoulder Width:", Shoulder_width)
        print("Sleeve length:", Sleeve_length)
        print("Shirt length:", Shirt_length)
        print("Neck Circumference:", neck_circumference)
        print("Neck Circumference:", Bicep_circumference)
        print("Wrist Circumference:", wrist_circumference)
