import tensorflow as tf

class Pix2PixModel:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.set_generator(self.input_shape, self.output_shape[2])
        self.set_discriminator(self.input_shape, self.output_shape)
        self.set_gan()

    #The generator for the pix2pix network is an skip connection autoencoder
    #Here we define the encoder unit for the encoder network
    def build_encoder_block(self, layer_in, n_filters, batchnorm=True):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        # Generator
        block = tf.keras.layers.Conv2D(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)

        if batchnorm:
            block = tf.keras.layers.BatchNormalization()(block, training=True)

        block = tf.keras.layers.LeakyReLU(alpha=0.2)(block)

        return block

    #Here we define the decoder block for the network
    def build_decoder_block(self, layer_in, skip_in, n_filters, dropout=True):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        block = tf.keras.layers.Conv2DTranspose(n_filters, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(layer_in)
        block = tf.keras.layers.BatchNormalization()(block, training=True)

        if dropout:
            block = tf.keras.layers.Dropout(0.5)(block, training=True)

        print(skip_in.shape)
        block = tf.keras.layers.concatenate([block, skip_in])

        block = tf.keras.layers.ReLU()(block)

        return block

    #Set generator to the model
    def set_generator(self, image_shape=(256, 256, 3), output_size=1):
        init = tf.keras.initializers.RandomNormal(stddev=0.02)

        in_image = tf.keras.Input(shape=image_shape, name="input_generator")

        encoder1 = self.build_encoder_block(in_image, 64, batchnorm=False)
        encoder2 = self.build_encoder_block(encoder1, 128)
        encoder3 = self.build_encoder_block(encoder2, 256)
        encoder4 = self.build_encoder_block(encoder3, 512)
        encoder5 = self.build_encoder_block(encoder4, 512)
        encoder6 = self.build_encoder_block(encoder5, 512)
        encoder7 = self.build_encoder_block(encoder6, 512)

        b = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init,
                                   activation='relu')(encoder7)

        decoder1 = self.build_decoder_block(b, encoder7, 512)
        decoder2 = self.build_decoder_block(decoder1, encoder6, 512)
        decoder3 = self.build_decoder_block(decoder2, encoder5, 512)
        decoder4 = self.build_decoder_block(decoder3, encoder4, 512, dropout=False)
        decoder5 = self.build_decoder_block(decoder4, encoder3, 256, dropout=False)
        decoder6 = self.build_decoder_block(decoder5, encoder2, 128, dropout=False)
        decoder7 = self.build_decoder_block(decoder6, encoder1, 64, dropout=False)

        out_image = tf.keras.layers.Conv2DTranspose(output_size, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init, activation='tanh')(decoder7)

        self.g_model = tf.keras.Model(in_image, out_image)

    #Set discriminator to model
    def set_discriminator(self, image_shape, output_shape):
        # Discriminator values
        init = tf.keras.initializers.RandomNormal(stddev=0.02)
        # source image input
        in_src_image = tf.keras.Input(shape=image_shape)
        # target image input
        in_target_image = tf.keras.Input(shape=output_shape)
        # merged images
        merged = tf.keras.layers.concatenate([in_src_image, in_target_image])

        discriminator = tf.keras.layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(
            merged)
        discriminator = tf.keras.layers.LeakyReLU(alpha=0.2)(discriminator)

        discriminator = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(
            discriminator)
        discriminator = tf.keras.layers.BatchNormalization()(discriminator)
        discriminator = tf.keras.layers.LeakyReLU(alpha=0.2)(discriminator)

        discriminator = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(
            discriminator)
        discriminator = tf.keras.layers.BatchNormalization()(discriminator)
        discriminator = tf.keras.layers.LeakyReLU(alpha=0.2)(discriminator)

        discriminator = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same', kernel_initializer=init)(
            discriminator)
        discriminator = tf.keras.layers.BatchNormalization()(discriminator)
        discriminator = tf.keras.layers.LeakyReLU(alpha=0.2)(discriminator)

        discriminator = tf.keras.layers.Conv2D(512, (4, 4), padding='same', kernel_initializer=init)(discriminator)
        discriminator = tf.keras.layers.BatchNormalization()(discriminator)
        discriminator = tf.keras.layers.LeakyReLU(alpha=0.2)(discriminator)

        discriminator = tf.keras.layers.Conv2D(1, (4, 4), padding='same', kernel_initializer=init)(discriminator)
        patch_out = tf.keras.activations.sigmoid(discriminator)

        self.d_model = tf.keras.Model([in_src_image, in_target_image], patch_out)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        self.d_model.compile(loss='binary_crossentropy', optimizer=optimizer, loss_weights=[0.5])

    #Set final GAN to the model
    def set_gan(self):
        for layer in self.d_model.layers:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        # instantiate input
        in_src = tf.keras.Input(shape=self.input_shape)
        # model generator output
        gen_out = self.g_model(in_src)

        print(gen_out.shape)
        # discriminator output
        dis_out = self.d_model([in_src, gen_out])

        self.gan_model = tf.keras.Model(in_src, [dis_out, gen_out])

        opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.gan_model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1, 100])
        self.gan_model.summary()

    #Define function to perform the adversarial training
    def train(self, generator, n_ephocs=100, n_batch=1):
        n_patch = self.d_model.output_shape[1]

        steps = int(len(generator.files) / n_batch)
        for j in range(n_ephocs):
            for i in range(steps):
                [x_realA, x_realB], y = generator.generate_real_samples(n_batch, n_patch)
                x_fake, y_fake = generator.generate_fake_samples(self.g_model, x_realA, n_patch)

                d_loss_1 = self.d_model.train_on_batch([x_realA, x_realB], y)
                d_loss_2 = self.d_model.train_on_batch([x_realA, x_fake], y_fake)

                g_loss, _, _ = self.gan_model.train_on_batch(x_realA, [y, x_realB])

                print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i + 1, d_loss_1, d_loss_2, g_loss))