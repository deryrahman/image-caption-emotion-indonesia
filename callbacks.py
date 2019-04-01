from keras.callbacks import Callback


class ModelCheckpoint(Callback):

    def __init__(self, architecture, filepath):
        self.architecture = architecture
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        print('save model on epoch-{}'.format(epoch))
        self.architecture.save(self.filepath)
