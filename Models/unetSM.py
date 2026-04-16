from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

class Unet(tf.keras.Model):
	def __init__(self, inh, inw, ind, outd, ff=1, wdr=0.00001):
		super(Unet, self).__init__()

		WEIGHT_DECAY_RATE = wdr
		REGULARIZER = tf.keras.regularizers.l2
		self.PADDING = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

		self.allName = ['en1in', 'mp2', 'en1conv1', 'en1conv2', 'en2conv1', 'en2conv2', 'en3conv1', 'en3conv2', 'en4conv1', 'en4conv2', 'bottle_conv1', 'bottle_conv2', 'de4dc', 'de4conv1', 'de4conv2', 'de3dc', 'de3conv1', 'de3conv2', 'de2dc', 'de2conv1', 'de2conv2', 'de1dc', 'de1conv1', 'de1conv2', 'de1conv3']

		self.nameLayWei = ['en1conv1', 'en1conv2', 'en2conv1', 'en2conv2', 'en3conv1', 'en3conv2', 'en4conv1', 'en4conv2', 'bottle_conv1', 'bottle_conv2', 'de4dc', 'de4conv1', 'de4conv2', 'de3dc', 'de3conv1', 'de3conv2', 'de2dc', 'de2conv1', 'de2conv2', 'de1dc', 'de1conv1', 'de1conv2', 'de1conv3']

		self.en1in = tf.keras.layers.InputLayer(input_shape=(inh, inw, ind), name='en1in')
		self.mp2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2, name='mp2')
		self.deConcat = tf.keras.layers.Concatenate()

		self.en1conv1 = tf.keras.layers.Conv2D(filters=int(64/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='en1conv1')
		self.en1conv2 = tf.keras.layers.Conv2D(filters=int(64/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='en1conv2')

		self.en2conv1 = tf.keras.layers.Conv2D(filters=int(128/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='en2conv1')
		self.en2conv2 = tf.keras.layers.Conv2D(filters=int(128/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='en2conv2')

		self.en3conv1 = tf.keras.layers.Conv2D(filters=int(256/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='en3conv1')
		self.en3conv2 = tf.keras.layers.Conv2D(filters=int(256/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='en3conv2')

		self.en4conv1 = tf.keras.layers.Conv2D(filters=int(512/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='en4conv1')
		self.en4conv2 = tf.keras.layers.Conv2D(filters=int(512/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='en4conv2')

		self.bottle_conv1 = tf.keras.layers.Conv2D(filters=int(1024/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='bottle_conv1')
		self.bottle_conv2 = tf.keras.layers.Conv2D(filters=int(1024/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='bottle_conv2')

		self.de4dc = tf.keras.layers.Conv2DTranspose(filters=int(512/ff), kernel_size=2, strides=2, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de4dc')
		self.de4conv1 = tf.keras.layers.Conv2D(filters=int(512/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de4conv1')
		self.de4conv2 = tf.keras.layers.Conv2D(filters=int(512/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de4conv2')

		self.de3dc = tf.keras.layers.Conv2DTranspose(filters=int(256/ff), kernel_size=2, strides=2, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de3dc')
		self.de3conv1 = tf.keras.layers.Conv2D(filters=int(256/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de3conv1')
		self.de3conv2 = tf.keras.layers.Conv2D(filters=int(256/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de3conv2')

		self.de2dc = tf.keras.layers.Conv2DTranspose(filters=int(128/ff), kernel_size=2, strides=2, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de2dc')
		self.de2conv1 = tf.keras.layers.Conv2D(filters=int(128/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de2conv1')
		self.de2conv2 = tf.keras.layers.Conv2D(filters=int(128/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de2conv2')

		self.de1dc = tf.keras.layers.Conv2DTranspose(filters=int(64/ff), kernel_size=2, strides=2, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de1dc')
		self.de1conv1 = tf.keras.layers.Conv2D(filters=int(64/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de1conv1')
		self.de1conv2 = tf.keras.layers.Conv2D(filters=int(64/ff), kernel_size=3, strides=1, activation='relu', kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de1conv2')

		self.de1conv3 = tf.keras.layers.Conv2D(filters=outd, kernel_size=1, strides=1, kernel_regularizer=REGULARIZER(WEIGHT_DECAY_RATE), name='de1conv3')

	def pad(self, lay):
		layPad = tf.pad(lay, self.PADDING, "REFLECT")

		return layPad

	def call(self, tsIn, dr=0):
		y = self.en1in(tsIn)
		y = self.pad(y)
		y = self.en1conv1(y)
		y = self.pad(y)
		yConcat1 = self.en1conv2(y)
		y = self.mp2(yConcat1)
		y = self.pad(y)
		y = tf.keras.layers.Dropout(dr)(y)
		y = self.en2conv1(y)
		y = self.pad(y)
		yConcat2 = self.en2conv2(y)
		y = self.mp2(yConcat2)
		y = self.pad(y)
		y = tf.keras.layers.Dropout(dr)(y)
		y = self.en3conv1(y)
		y = self.pad(y)
		yConcat3 = self.en3conv2(y)
		y = self.mp2(yConcat3)
		y = self.pad(y)
		y = tf.keras.layers.Dropout(dr)(y)
		y = self.en4conv1(y)
		y = self.pad(y)
		yConcat4 = self.en4conv2(y)
		y = self.mp2(yConcat4)
		y = self.pad(y)
		y = self.bottle_conv1(y)
		y = self.pad(y)
		y = self.bottle_conv2(y)
		y = self.de4dc(y)
		y = self.deConcat([yConcat4, y])
		y = self.pad(y)
		y = tf.keras.layers.Dropout(dr)(y)
		y = self.de4conv1(y)
		y = self.pad(y)
		y = self.de4conv2(y)
		y = self.de3dc(y)
		y = self.deConcat([yConcat3, y])
		y = self.pad(y)
		y = tf.keras.layers.Dropout(dr)(y)
		y = self.de3conv1(y)
		y = self.pad(y)
		y = self.de3conv2(y)
		y = self.de2dc(y)
		y = self.deConcat([yConcat2, y])
		y = self.pad(y)
		y = tf.keras.layers.Dropout(dr)(y)
		y = self.de2conv1(y)
		y = self.pad(y)
		y = self.de2conv2(y)
		y = self.de1dc(y)
		y = self.deConcat([yConcat1, y])
		y = self.pad(y)
		y = self.de1conv1(y)
		y = self.pad(y)
		y = self.de1conv2(y)
		y = self.de1conv3(y)
		tsOut = tf.nn.softmax(y, 3)

		return tsOut

	def getWeiLayName(self):
		
		return self.nameLayWei
