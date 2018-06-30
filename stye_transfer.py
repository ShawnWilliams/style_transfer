from __future__ import print_function

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time

import numpy as np
import tensorflow as tf

import vgg_model
import utils

# parameters
STYLE = 'starry_night'
CONTENT = 'hua'
STYLE_IMAGE = 'styles/' + STYLE + '.jpg'
CONTENT_IMAGE = 'content/' + CONTENT + '.jpg'
# IMAGE_HEIGHT = 250
# IMAGE_WIDTH = 333
IMAGE_HEIGHT = 250
IMAGE_WIDTH = 400
OUT_DIR = 'flower_pattern'
CHECKPOINT_DIR = 'flower_starryNight_checkpoints'
NOISE_RATIO = 0.6  # percentage of weight of the noise for mixing with the content image

CONTENT_WEIGHT = 0.03
STYLE_WEIGHT = 1

# Layers used for style features.
STYLE_LAYERS = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
W = [0.5, 1.0, 1.5, 3.0, 4.0]  # more weights to deeper layers.

# Layer used for content features.
CONTENT_LAYER = 'conv4_2'

ITERS = 300
LR = 2.0

SAVE_EVERY = 20

# MEAN_PIXELS is defined according to description on this github: https://gist.github.com/ksimonyan/211839e770f7b538e2d8
MEAN_PIXELS = np.array([123.68, 116.779, 103.939]).reshape((1, 1, 1, 3))


# VGG-19 parameters file
VGG_DOWNLOAD_LINK = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'
VGG_MODEL = 'imagenet-vgg-verydeep-19.mat'
EXPECTED_BYTES = 534904783


def _create_content_loss(p, f):
    """ Calculate the loss between the feature representation of the
    content image and the generated image.

    Inputs:
        p, f are just P, F in the paper
        use coeffient 4 * p.szie instead of 0.5 as defined in the paper,
        it can accelerate the converge speed.
    Output:
        the content loss

    """
    return tf.reduce_sum((f - p) ** 2) / (4.0 * p.size)


def _gram_matrix(F, N, M):
    """ Create and return the gram matrix for tensor F
        Hint: you'll first have to reshape F
    """
    # N is the third dimension of the feature map, and M is the product of the first two dimensions of the feature map.
    F = tf.reshape(F, (M, N))
    return tf.matmul(tf.transpose(F), F)


def _single_style_loss(a, g):
    """ Calculate the style loss at a certain layer

    Inputs:
        a is the feature representation of the real image
        g is the feature representation of the generated image

    Output:
        the style loss at a certain layer (which is E_l in the paper)

    """
    N = a.shape[3]  # number of filters
    M = a.shape[1] * a.shape[2]  # height times width of the feature map
    A = _gram_matrix(a, N, M)
    G = _gram_matrix(g, N, M)
    return tf.reduce_sum((G - A) ** 2) / ((2 * N * M) ** 2)


def _create_style_loss(A, model):
    """ Return the total style loss

    Inputs:
        a is the feature representation of the real image
        g is the feature representation of the generated image

    Output:
        total style loss

    """
    n_layers = len(STYLE_LAYERS)
    E = [_single_style_loss(A[i], model[STYLE_LAYERS[i]]) for i in range(n_layers)]

    return sum([W[i] * E[i] for i in range(n_layers)])


def _create_losses(model, input_image, content_image, style_image):
    with tf.variable_scope('loss') as scope:
        with tf.Session() as sess:
            sess.run(input_image.assign(content_image))  # assign content image to the input variable
            p = sess.run(model[CONTENT_LAYER])
        content_loss = _create_content_loss(p, model[CONTENT_LAYER])

        with tf.Session() as sess:
            sess.run(input_image.assign(style_image))
            A = sess.run([model[layer_name] for layer_name in STYLE_LAYERS])
        style_loss = _create_style_loss(A, model)


        total_loss = CONTENT_WEIGHT * content_loss + STYLE_WEIGHT * style_loss

    return content_loss, style_loss, total_loss


def _create_summary(model):
    """ Create summary ops necessary
    """

    with tf.name_scope('summaries'):
        tf.summary.scalar('content loss', model['content_loss'])
        tf.summary.scalar('style loss', model['style_loss'])
        tf.summary.scalar('total loss', model['total_loss'])
        tf.summary.histogram('histogram content loss', model['content_loss'])
        tf.summary.histogram('histogram style loss', model['style_loss'])
        tf.summary.histogram('histogram total loss', model['total_loss'])
        return tf.summary.merge_all()


def train(model, generated_image, initial_image):
    skip_step = 1
    with tf.Session() as sess:
        saver = tf.train.Saver()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('graphs', sess.graph)

        sess.run(generated_image.assign(initial_image))
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(CHECKPOINT_DIR + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = model['global_step'].eval()

        start_time = time.time()
        for index in range(initial_step, ITERS):
            if index >= 5 and index < 20:
                skip_step = 10
            elif index >= 20:
                skip_step = 20

            sess.run(model['optimizer'])
            if (index + 1) % skip_step == 0:

                gen_image, total_loss, summary = sess.run([generated_image, model['total_loss'], model['summary_op']])
                gen_image = gen_image + MEAN_PIXELS
                writer.add_summary(summary, global_step=index)
                print('Step {}\n   Sum: {:5.1f}'.format(index + 1, np.sum(gen_image)))
                print('   Loss: {:5.1f}'.format(total_loss))
                print('   Time: {}'.format(time.time() - start_time))
                start_time = time.time()

                filename = (OUT_DIR + '/%d.png') % (index)
                utils.save_image(filename, gen_image)

                if (index + 1) % SAVE_EVERY == 0:
                    saver.save(sess, CHECKPOINT_DIR + '/style_transfer', index)


def main():
    with tf.variable_scope('input') as scope:
        input_image = tf.Variable(np.zeros([1, IMAGE_HEIGHT, IMAGE_WIDTH, 3]), dtype=tf.float32)

    utils.download(VGG_DOWNLOAD_LINK, VGG_MODEL, EXPECTED_BYTES)
    utils.make_dir(CHECKPOINT_DIR)
    # utils.make_dir('outputs')
    utils.make_dir(OUT_DIR)
    model = vgg_model.load_vgg(VGG_MODEL, input_image)
    model['global_step'] = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    # because PIL is column major so need to change place of width and height
    content_image = utils.get_resized_image(CONTENT_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    content_image = content_image - MEAN_PIXELS
    style_image = utils.get_resized_image(STYLE_IMAGE, IMAGE_HEIGHT, IMAGE_WIDTH)
    style_image = style_image - MEAN_PIXELS

    model['content_loss'], model['style_loss'], model['total_loss'] = _create_losses(model,
                                                                                     input_image, content_image,
                                                                                     style_image)

    model['optimizer'] = tf.train.AdamOptimizer(LR).minimize(model['total_loss'], global_step=model['global_step'])

    model['summary_op'] = _create_summary(model)

    initial_image = utils.generate_noise_image(content_image, IMAGE_HEIGHT, IMAGE_WIDTH, NOISE_RATIO)
    train(model, input_image, initial_image)


if __name__ == '__main__':
    main()
