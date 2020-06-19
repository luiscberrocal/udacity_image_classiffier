import argparse

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import json

parser = argparse.ArgumentParser()

# Add the arguments to the parser
parser.add_argument("-t", "--top_k", required=False,
                    help="Top results", default='5')
parser.add_argument("-c", "--category_names", required=False,
                    default='label_map.json',
                    help="Categories")

parser.add_argument("image")
parser.add_argument("model")


def process_image(image_path, image_size=224):
    image = Image.open(image_path)
    numpy_image = np.asarray(image)
    numpy_image = tf.cast(numpy_image, tf.float32)
    numpy_image = tf.image.resize(numpy_image, (image_size, image_size))
    numpy_image /= image_size
    return numpy_image


def predict(image, model, top_k=5):
    result = model.predict(image)
    probs, classes = tf.nn.top_k(result, top_k)
    return probs.numpy().tolist()[0], classes.numpy().tolist()[0]


if __name__ == '__main__':
    """
        $ python predict.py /path/to/image saved_model
    Options:

    --top_k : Return the top KK most likely classes:

        $ python predict.py /path/to/image saved_model --top_k KK

    --category_names : Path to a JSON file mapping labels to flower names:
        $ python predict.py /path/to/image saved_model --category_names map.json

    The best way to get the command line input into the scripts is with the argparse module in the standard library. 
    You can also find a nice tutorial for argparse here.

    Examples
    For the following examples, we assume we have a file called orchid.jpg in a folder named/test_images/ that contains
    the image of a flower. We also assume that we have a Keras model saved in a file named my_model.h5.

    Basic usage:

        $ python predict.py ./test_images/orchid.jpg my_model.h5

    Options:

    Return the top 3 most likely classes:

        $ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3

    Use a label_map.json file to map labels to flower names:

        $ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json

    """
    args = vars(parser.parse_args())

    print(args)
    top_k = int(args['top_k'])
    labels_file = args['category_names']
    with open(labels_file, 'r') as json_file:
        class_names = json.load(json_file)

    keras_model = tf.keras.models.load_model(args['model'], custom_objects={'KerasLayer': hub.KerasLayer})
    print(keras_model.summary())
    numpy_image = process_image(args['image'])
    probs, classes = predict(np.expand_dims(numpy_image, axis=0), keras_model, top_k=top_k)

    i = 0
    for prob in probs:
        class_num = str(classes[i] + 1)
        print(prob, class_num, class_names[class_num])
        i += 1

