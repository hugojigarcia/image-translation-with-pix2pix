# USAGE
# python train_pix2pix.py 

# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf
tf.random.set_seed(42)

# import the necessary packages
from pyimagesearch import config
from pyimagesearch.Pix2PixTraining import Pix2PixTraining
from pyimagesearch.Pix2PixGAN import Pix2Pix
from pyimagesearch.data_preprocess import load_dataset
from pyimagesearch.train_monitor import get_train_monitor
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.utils import get_file
import pathlib
import os

# download the cityscape training dataset 
print("[INFO] downloading the dataset...")
pathToZip = get_file(
	fname=f"{config.DATASET}.tar.gz",
	origin=config.DATASET_URL,
	extract=True
)
pathToZip  = pathlib.Path(pathToZip)
path = pathToZip.parent/config.DATASET

# build the training dataset
print("[INFO] building the train dataset...")
trainDs = load_dataset(path=path, train=True,
	batchSize=config.TRAIN_BATCH_SIZE, height=config.IMAGE_HEIGHT,
	width=config.IMAGE_WIDTH)

# build the test dataset
print("[INFO] building the test dataset...")
testDs = load_dataset(path=path, train=False,
	batchSize=config.INFER_BATCH_SIZE, height=config.IMAGE_HEIGHT,
	width=config.IMAGE_WIDTH)

# initialize the generator and discriminator network
print("[INFO] initializing the generator and discriminator...")
pix2pixObject = Pix2Pix(imageHeight=config.IMAGE_HEIGHT,
	imageWidth=config.IMAGE_WIDTH)
generator = pix2pixObject.generator()
discriminator = pix2pixObject.discriminator()

# build the pix2pix training model and compile it
pix2pixModel = Pix2PixTraining(
	generator=generator,
	discriminator=discriminator)
pix2pixModel.compile(
	dOptimizer=Adam(learning_rate=config.LEARNING_RATE),
	gOptimizer=Adam(learning_rate=config.LEARNING_RATE),
	bceLoss=BinaryCrossentropy(from_logits=True),
	maeLoss=MeanAbsoluteError(),
)

# check whether output model directory exists
# if it doesn't, then create it
if not os.path.exists(config.BASE_OUTPUT_PATH):
	os.makedirs(config.BASE_OUTPUT_PATH)

# check whether output image directory exists, if it doesn't, then
# create it
if not os.path.exists(config.BASE_IMAGES_PATH):
	os.makedirs(config.BASE_IMAGES_PATH)

# train the pix2pix model
print("[INFO] training the pix2pix model...")
callbacks = [get_train_monitor(testDs, epochInterval=10,
	imagePath=config.BASE_IMAGES_PATH,
	batchSize=config.INFER_BATCH_SIZE)]
pix2pixModel.fit(trainDs, epochs=config.EPOCHS, callbacks=callbacks,
	steps_per_epoch=config.STEPS_PER_EPOCH)

# set the path for the generator
genPath = config.GENERATOR_MODEL

# save the pix2pix generator
print(f"[INFO] saving pix2pix generator to {genPath}...")
pix2pixModel.generator.save(genPath)