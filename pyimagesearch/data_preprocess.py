# import the necessary packages
import tensorflow as tf

# define the module level autotune
AUTO = tf.data.AUTOTUNE
def load_image(imageFile):
	# read and decode an image file from the path
	image = tf.io.read_file(imageFile)
	image = tf.io.decode_jpeg(image, channels=3)

	# calculate the midpoint of the width and split the
	# combined image into input mask and real image 
	width = tf.shape(image)[1]
	splitPoint = width // 2
	inputMask = image[:, splitPoint:, :]
	realImage = image[:, :splitPoint, :]

	# convert both images to float32 tensors and
	# convert pixels to the range of -1 and 1
	inputMask = tf.cast(inputMask, tf.float32)/127.5 - 1
	realImage = tf.cast(realImage, tf.float32)/127.5 - 1

	# return the input mask and real label image
	return (inputMask, realImage)