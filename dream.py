from keras import backend as K, models
from keras.models import Model
from scipy.misc import imsave
from PIL import Image
import numpy as np
from os.path import basename
from datetime import datetime

model_orig = models.load_model('models/CatDog_0.941.h5')
model = Model(inputs=model_orig.input, outputs=model_orig.layers[49].output)

layer_dict = {l.name:l for l in model.layers}
filter_index = 0
layer_name = 'conv2d_6'


def deproc_img(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    # x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def deproc2(x):
    x /= 2.
    x += 0.5
    x *= 255
    # x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def deproc3(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def dream(filename = None, iterations = 200, weight = 0.01):
    activation = K.variable(0.)

    for i in range(1, len(model.layers)):
        if 'conv2d' in model.layers[i].name:
            curr_out = model.layers[i].output
            activation += K.sum(K.square(curr_out))
            activation /= K.prod(K.cast(K.shape(curr_out), 'float32'))

    # for i in range(1, len(model.layers)):
    #     if 'conv2d' in model.layers[i].name:
    #         curr_out = model.layers[i].output
    #         # Sum the squares of the output of the current layer, divide by scaling factor (apparently called the L2?)
    #         activation += K.sum(K.square(curr_out))
    #         activation /= K.prod(K.cast(K.shape(curr_out), 'float32'))

    # activation = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(activation, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) # K.maximum(K.mean(K.abs(grads)), K.epsilon())#

    iterate = K.function([model.input], [activation, grads])
    # Create a pipeline that takes an input tensor, puts it through the graph, and returns the loss and the gradient
    # In the above, all values are symbolic

    if filename is None:
        in_img_data = (np.random.random((1, 500, 500, 3)) * 20 + 128)/255
    else:
        basewidth = 500
        img = Image.open(filename)
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.ANTIALIAS)
        in_img_data = np.expand_dims(np.asarray(img, dtype='float32'), 0)/255

    print('min: {}, max: {}, std: {}'.format(in_img_data.min(), in_img_data.max(), in_img_data.std()))

    for i in range(iterations):
        loss_val, grads_val = iterate([in_img_data])
        if i % 10 == 0:
            print('loss at {}: {}'.format(i, loss_val))
        # if loss_val >= 10:
        #     print(loss_val)
        #     break
        in_img_data += weight * np.cbrt(np.cbrt(grads_val / (abs(grads_val)).max()))

    img = in_img_data[0]
    img = deproc3(img)
    name = 'dream-{}.png'.format(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    imsave(name, img)


def load_img(path):
    temp_img_ = Image.open(path)
    temp_img_.load()
    temp_img_ = np.asarray(temp_img_)
    temp_img_ = np.expand_dims(temp_img_, 0)
    return temp_img_


def dreamf(filename, layer_dict=layer_dict, layer_name=layer_name, filter_index=filter_index):
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])
    # Create a pipeline that takes an input tensor, puts it through the graph, and returns the loss and the gradient
    # In the above, all values are symbolic
    #
    # in_img_data = K.cast(np.copy(load_img(filename)), 'float32')
    basewidth = 300
    img = Image.open(filename)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    in_img_data = np.expand_dims(np.asarray(img, dtype='float32'), 0)
    for i in range(20):
        loss_val, grads_val = iterate([in_img_data])
        in_img_data += grads_val
    fn = basename(filename)
    img = in_img_data[0]
    img = deproc_img(img)
    name = 'dream/{}-{}_filter_{}.png'.format(fn, layer_name, filter_index)
    imsave(name, img)


def dreamr(filename, layer_dict=layer_dict, layer_name=layer_name, filter_index=filter_index):
    layer_output = layer_dict[layer_name].output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])
    # Create a pipeline that takes an input tensor, puts it through the graph, and returns the loss and the gradient
    # In the above, all values are symbolic
    #
    # in_img_data = K.cast(np.copy(load_img(filename)), 'float32')
    basewidth = 300
    img = Image.open(filename)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), Image.ANTIALIAS)
    in_img_data = np.expand_dims(np.asarray(img, dtype='float32'), 0)
    for i in range(20):
        loss_val, grads_val = iterate([in_img_data])
        in_img_data += grads_val
    fn = basename(filename)
    img = in_img_data[0]
    img = deproc_img(img)
    name = 'dream/{}-{}_filter_{}.png'.format(fn, layer_name, filter_index)
    imsave(name, img)