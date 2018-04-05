from mnist import MNIST
import random

mndata = MNIST('train_data\\mnist')

images, labels = mndata.load_training()
# or
#images, labels = mndata.load_testing()

index = random.randrange(0, len(images))  # choose an index ;-)
#print(len(images[index]))
print(mndata.display(images[index]))