# nn1

**Deep convolutional neural network implemented in Keras with TensorFlow backend**

This network was trained on the Kaggle cat-dog classification competition dataset. Dream was implemented loosely based off of https://github.com/keras-team/keras/blob/master/examples/deep_dream.py, except the convnet was manually architected and trained. The architecture is also inspired by VGGNet16, but with fewer and less complex convolutional layers.

The highest validation accuracy reached so far is 0.960 after 7500 iterations. I guess we'll have to see how much further it can get...

**Edit:** 0.972 @ 10060
