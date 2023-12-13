import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

def loadArray(x):
  max_dim = 512
  img = tf.image.convert_image_dtype(x, tf.float32)
  shape = tf.cast(tf.shape(img)[1:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim
  new_shape = tf.cast(shape * scale, tf.int32)
  img = tf.image.resize(img, new_shape)
  return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)


def vgg_layers(layer_names):
  """ Creates a VGG model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on ImageNet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model


def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

class StyleContentModel(tf.keras.models.Model):
  def __init__(self, style_layers, content_layers):
    super(StyleContentModel, self).__init__()
    self.vgg = vgg_layers(style_layers + content_layers)
    self.style_layers = style_layers
    self.content_layers = content_layers
    self.num_style_layers = len(style_layers)
    self.vgg.trainable = False

  def call(self, inputs):
    "Expects float input in [0,1]"
    inputs = inputs*255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                      outputs[self.num_style_layers:])

    style_outputs = [gram_matrix(style_output)
                     for style_output in style_outputs]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    style_dict = {style_name: value
                  for style_name, value
                  in zip(self.style_layers, style_outputs)}

    return {'content': content_dict, 'style': style_dict}


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)



def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))



class MHStyler:
    def __init__(self, style_path, style_weight=1e-2, content_weight=1e4, total_variation_weight=30, content_layers=None, style_layers=None):
        self.style_path = style_path
        self.style_image = load_img(style_path)
        self.style_weight = style_weight
        self.total_variation_weight = total_variation_weight
        self.content_weight = content_weight
        if not content_layers:
            self.content_layers = ['block5_conv2']
        else:
            self.content_layers = content_layers
        
        if not style_layers:
            self.style_layers = ['block1_conv1',
                                 'block2_conv1',
                                 'block3_conv1', 
                                 'block4_conv1', 
                                 'block5_conv1']
        else:
            self.style_layers = style_layers
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.extractor = StyleContentModel(self.style_layers, self.content_layers)
        self.style_targets = self.extractor(self.style_image)['style']

    def style_content_loss(self, outputs, style_weight=1e-2, content_weight=1e4, num_content_layers=1, num_style_layers=1):
        style_outputs = outputs['style']
        content_outputs = outputs['content']
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-self.style_targets[name])**2) 
                            for name in style_outputs.keys()])
        style_loss *= style_weight / num_style_layers

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-self.content_targets[name])**2) 
                                for name in content_outputs.keys()])
        content_loss *= content_weight / num_content_layers
        loss = style_loss + content_loss
        return loss

    def train_step(self, image, opt, style_weight=1e-2, content_weight=1e4, total_variation_weight=30, num_content_layers=1, num_style_layers=1):
        with tf.GradientTape() as tape:
            outputs = self.extractor(image)
            loss = self.style_content_loss(outputs)
            loss += total_variation_weight*tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        opt.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    def __call__(self, content, n=1):
        if isinstance(content, str):
            content_image = load_img(content)
        else:
            content_image = loadArray(content)
        self.content_targets = self.extractor(content_image)['content']
        image = tf.Variable(content_image)
        opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)
        for i in range(n):
            self.train_step(image, opt, self.style_weight, self.content_weight, self.total_variation_weight, self.num_content_layers, self.num_style_layers)
        return np.array(image * 255., dtype=np.uint8)


if __name__ == '__main__':
    mh_styler = MHStyler('img_udacity_0.png')
    img = plt.imread('img_beamng_0.png').copy()
    image_arr = mh_styler(img[np.newaxis, ...], 5)
    plt.imshow(image_arr)
    plt.show()
    img = plt.imread('img_beamng_1.png').copy()
    image_arr = mh_styler(img[np.newaxis, ...], 3)
    plt.imshow(image_arr)
    plt.show()

