import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image
from mh_ds import loadDataset
import pandas as pd
from tqdm import tqdm
# import logging as log

# log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ds_names = ['beamng']  #  , 'dclgan', 'saevae', 'magenta']
df_index = ['BeamNG']
ds = [loadDataset(ds_name) for ds_name in ds_names]
func_map = lambda x: {'total': (np.concatenate((x[0], x[4]), axis=0), np.concatenate((x[1], x[5]), axis=0)), 'test': (x[4], x[5])}
ds = map(func_map, ds)
ds = {ds_name: ds_ for ds_name, ds_ in zip(ds_names, ds)}
X = ds['beamng']['total'][0]
y = ds['beamng']['total'][1]
print(f'Dataset loaded: {X.shape}, {y.shape}')
shape_original = X.shape[1:-1]

model_path = 'models/mh_autumn_udacity_1'
model_name = os.path.basename(model_path)
model = tf.keras.models.load_model(model_path)
print(f'Model loaded: {model_path}')

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

content_path = 'img_beamng_0.png'
style_path = 'img_udacity_0.png'

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

N = 500
XX = loadArray(X[:N])
yy = y[:N]

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

content_image = load_img(content_path)
style_image = load_img(style_path)

# plt.subplot(1, 2, 1)
# imshow(content_image, 'Content Image')

# plt.subplot(1, 2, 2)
# imshow(style_image, 'Style Image')

# plt.show()

x = tf.keras.applications.vgg19.preprocess_input(content_image*255)
x = tf.image.resize(x, (224, 224))

vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')

content_layers = ['block5_conv2'] 

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
  """ Creates a VGG model that returns a list of intermediate output values."""
  # Load our model. Load pretrained VGG, trained on ImageNet data
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  
  outputs = [vgg.get_layer(name).output for name in layer_names]

  model = tf.keras.Model([vgg.input], outputs)
  return model

style_extractor = vgg_layers(style_layers)
style_outputs = style_extractor(style_image*255)

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

extractor = StyleContentModel(style_layers, content_layers)

results = extractor(tf.constant(content_image))

style_targets = extractor(style_image)['style']
content_targets = extractor(content_image)['content']

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

opt = tf.keras.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

style_weight=1e-2
content_weight=3e4

def high_pass_x_y(image):
  x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
  y_var = image[:, 1:, :, :] - image[:, :-1, :, :]

  return x_var, y_var

def total_variation_loss(image):
  x_deltas, y_deltas = high_pass_x_y(image)
  return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))



def style_content_loss(outputs, style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n([tf.reduce_mean((style_outputs[name]-style_targets[name])**2) 
                           for name in style_outputs.keys()])
    style_loss *= style_weight / num_style_layers

    content_loss = tf.add_n([tf.reduce_mean((content_outputs[name]-content_targets[name])**2) 
                             for name in content_outputs.keys()])
    content_loss *= content_weight / num_content_layers
    loss = style_loss + content_loss
    return loss

total_variation_weight=30

# @tf.function()
def train_step(image, style_weight, content_weight):
  opt = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.99, epsilon=1e-1)
  with tf.GradientTape() as tape:
    outputs = extractor(image)
    loss = style_content_loss(outputs, style_weight, content_weight)
    loss += total_variation_weight*tf.image.total_variation(image)

  grad = tape.gradient(loss, image)
  opt.apply_gradients([(grad, image)])
  image.assign(clip_0_1(image))

# image = tf.Variable(XX)

# print("Before train step: type", type(image))
# print("Before train step: shape ", image.shape)

# print("Train step: type", type(image))
# print("Train step: shape ", image.shape)
# image_arr = np.array(image * 255., dtype=np.uint8).squeeze()
# plt.imshow(image_arr)
# plt.show()

##  Raw test...
loss_raw = model.evaluate(X, y, steps=len(X)//128)
print('Raw test loss:', loss_raw)

##  Test with style transfer...
# train_step(XX)
def _evaluate(x, y, style_weight, content_weight):
    train_step(x, style_weight, content_weight)
    return x, model.evaluate(tf.image.resize(x, shape_original), y)

def evaluate(x, y, style_weight, content_weight, bs=128):
    n = len(x) // bs
    losses = [(*_evaluate(tf.Variable(x[i*bs:(i+1)*bs]), y[i*bs:(i+1)*bs], style_weight, content_weight),) for i in range(n)]
    x = tf.concat([x for x, _ in losses], 0)
    losses = [l for _, l in losses]
    return x, np.mean(losses, axis=0)

# style_weights = [1e-3, 1e-2, 1e-1]
# content_weights = [1e3, 1e4, 1e5]
style_weights = [1e-2]
content_weights = [1e4]

weights = [(sw, cw) for sw in style_weights for cw in content_weights]
no_of_steps = 3  ##  3

N = 1000  ##  3000
XX = loadArray(X[:N])
yy = y[:N]
x = XX

df_dict = {'Step': [], 'Style Weight': [], 'Content Weight': [], 'Loss': []}
output_dir = 'ds_beamng_style'
os.makedirs(output_dir, exist_ok=True)
labels_dict = {'img': [], 'steer': []}

for sw, cw in weights:
  for i in range(no_of_steps):
      print(f'Step {i+1}/{no_of_steps}, style weight: {sw}, content weight: {cw}')
      x, loss = evaluate(x, yy, sw, cw)
      x = np.array(x)
      print(f'Loss at step {i+1}: {loss}')
      df_dict['Step'].append(i+1)
      df_dict['Style Weight'].append(sw)
      df_dict['Content Weight'].append(cw)
      df_dict['Loss'].append(loss)
      df = pd.DataFrame(df_dict)
      df = df.sort_values(by=['Loss'])  
      df.to_csv(f'style_offline_eval_{model_name}.csv', index=False)
      if i == 2:
        for k, img in enumerate(tqdm(x)):
          plt.imsave(os.path.join(output_dir, f'img_{k}.png'), img)
          labels_dict['img'].append(f'img_{k}.png')
          labels_dict['steer'].append(yy[k])
        df_label = pd.DataFrame(labels_dict)
        df_label.to_csv(os.path.join(output_dir, 'labels.csv'), index=False)





    