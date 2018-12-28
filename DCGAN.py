import keras
from keras import layers
from keras.backend import tensorflow_backend as K
import numpy as np
import os
import PIL.Image as Image
from keras.preprocessing import image
latent_dim = 100
height = 32
width = 32
channels = 3
col = 5
row = 5
image_size = 32
save_dir = './WGAN_car2'


def wasserstein(y_true, y_pred):
    return K.mean(y_true * y_pred)


def Conv_Down(x, kernel_size, channel, name='Conv_Down?_'):
    # x = layers.BatchNormalization(axis=-1, name=name + 'BN')(x)
    x = layers.Conv2D(channel, kernel_size, strides=2, padding='same', name=name+'Conv1',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x)
    x = layers.LeakyReLU(0.2, name=name+'LeakyReLU1')(x)
    # x = layers.Dropout(0.2)(x)
    # x = layers.Conv2D(channel, kernel_size, padding='same', name=name + 'Conv2',
    #                   kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x)
    # x = layers.LeakyReLU(0.2, name=name + 'LeakyReLU2')(x)
    return x


def Conv_Up(x, kernel_size, channel, name='Conv_Down?_'):
    x = layers.Conv2DTranspose(channel, kernel_size, strides=2, padding='same', name=name+'ConvT',
                      kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x)
    x = layers.BatchNormalization(axis=-1, name=name+'BN')(x)
    # x = layers.LeakyReLU(0.2, name=name + 'LeakyReLU1')(x)
    x = layers.ReLU(name=name + 'ReLU')(x)
    return x


def G_net():
    generator_input = keras.Input(shape=(latent_dim,), name='G_input')
    x = layers.Dense(512 * 4 * 4, name='G_First_Dense',
                     kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(generator_input)
    x = layers.Reshape((4, 4, 512))(x)
    x = layers.BatchNormalization(name='G_BN0')(x)
    x = layers.ReLU(name='G_ReLU')(x)
    x = Conv_Up(x, kernel_size=5, channel=256, name='Conv_Up1_')
    # x = layers.BatchNormalization(name='G_BN1')(x)
    x = Conv_Up(x, kernel_size=5, channel=128, name='Conv_Up2_')
    # x = layers.BatchNormalization(name='G_BN2')(x)
    # x = Conv_Up(x, kernel_size=3, channel=128, name='Conv_Up3_')
    x = layers.Conv2DTranspose(channels, 5, activation='tanh', strides=2, padding='same', name='G_output',
                               kernel_initializer=keras.initializers.RandomNormal(stddev=0.02))(x)    # 输出层
    generator = keras.models.Model(generator_input, x, name='G')
    generator.summary()
    return generator


def D_net():
    discriminator_input = layers.Input(shape=(height, width, channels), name='D_input')
    # x = layers.Conv2D(64, 3, padding='same', name='D_First_Conv')(discriminator_input)
    # # x = layers.BatchNormalization(name='D_First_BN')(x)
    # x = layers.LeakyReLU()(x)
    x = Conv_Down(discriminator_input, kernel_size=5, channel=128, name='Conv_Down1_')
    x = layers.Dropout(0.3)(x)
    x = Conv_Down(x, kernel_size=5, channel=256, name='Conv_Down2_')
    x = layers.Dropout(0.3)(x)
    x = Conv_Down(x, kernel_size=5, channel=512, name='Conv_Down3_')
    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1, activation='sigmoid', name='D_output')(x)
    discriminator = keras.models.Model(discriminator_input, x, name='D')
    discriminator.summary()
    # discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0001, clipvalue=1.0, decay=1e-8)
    discriminator_optimizer = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(optimizer=discriminator_optimizer,
                          loss='binary_crossentropy')
    return discriminator


# G和D的网络结构
G = G_net()
D = D_net()
# G.load_weights(save_dir+'/G_3.5w.h5', by_name=True)
# D.load_weights(save_dir+'/D_3.5w.h5', by_name=True)
# G, D, WGAN = WGAN_net()
# G和D的训练方法
D.trainable = False
WGAN_input = keras.Input(shape=(latent_dim,))
WGAN_output = D(G(WGAN_input))
WGAN = keras.models.Model(WGAN_input, WGAN_output)
WGAN.summary()
# WGAN_optimizer = keras.optimizers.RMSprop(lr=0.0002, clipvalue=1.0, decay=1e-8)
WGAN_optimizer = keras.optimizers.Adam(lr=0.0004, beta_1=0.5)
WGAN.compile(optimizer=WGAN_optimizer, loss='binary_crossentropy')
# WGAN.load_weights(save_dir+'/WGAN_3.5w.h5', by_name=True)


# 定义图像拼接函数
def image_compose(ori_images, save_path):
    imgs = []
    for i in range(col * row):
        imgs.append(image.array_to_img((ori_images[i]+1) * 127.5, scale=False))
    to_image = Image.new('RGB', (col * image_size, row * image_size))   #创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(row):
        for x in range(col):
            from_image = imgs[y*col+x].resize((image_size, image_size), Image.ANTIALIAS)
            to_image.paste(from_image, (x * image_size, y * image_size))
    return to_image.save(save_path)  # 保存新图


# DCGAN的训练
(x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 1]   # car
x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 127.5 - 1
iterations = 40000
batch_size = 64
start = 0
for step in range(iterations):
    # Samples random points in the latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # Decodes them to fake images
    G_images = G.predict(random_latent_vectors)
    # Combines them with real images
    stop = start + batch_size
    real_images = x_train[start: stop]
    combined_images = np.concatenate([G_images, real_images])
    # Assembles labels, discriminating real from fake images
    labels = np.concatenate([0.90*np.ones((batch_size, 1)),
                             np.zeros((batch_size, 1))])
    # 为标签添加随机噪音————这是一个重要的技巧！
    labels += 0.05 * np.random.random(labels.shape)+0.05
    for i in range(1):
        d_loss = D.train_on_batch(combined_images, labels)

    # Samples random points in the Assembles latent space
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))
    # Assembles latent space labels that say “these are all real images” (it’s a lie!)
    misleading_targets = np.zeros((batch_size, 1))
    for i in range(1):
        a_loss = WGAN.train_on_batch(random_latent_vectors, misleading_targets)
    # a_loss = WGAN.train_on_batch(random_latent_vectors, misleading_targets)
    start += batch_size
    if start > len(x_train) - batch_size:
        start = 0
    # Occasionally saves and plots (every 100 steps)
    # print('step:', step, '/', iterations, '------', round(100 * step / iterations, 3), '%',
    #       '-- discriminator loss:', d_loss, '-- adversarial loss:', a_loss)
    if step % 125 == 0:
        if d_loss < 0: break
        print('step:', step, '/', iterations, '------', round(100 * step / iterations, 3), '%',
              '-- discriminator loss:', d_loss, '-- adversarial loss:', a_loss)
        if step < 5000 or step % 625 == 0:
            image_compose(G_images, os.path.join(save_dir, 'generated_pic' + str(step) + '.png'))
    if step % 5000 == 0:
        G.save(save_dir + '/G_' + str(step / 10000) + 'w.h5')
        D.save(save_dir + '/D_' + str(step / 10000) + 'w.h5')
        WGAN.save(save_dir + '/WGAN_' + str(step / 10000) + 'w.h5')
    if step % 10000 == 0:
        image_compose(real_images, os.path.join(save_dir, 'real_pic' + str(step) + '.png'))
G.save('./G_END.h5')
D.save('./D_END.h5')
WGAN.save('./WGAN_END.h5')
