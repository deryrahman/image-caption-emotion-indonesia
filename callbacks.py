from keras.callbacks import ModelCheckpoint as ModelCheckpointBase


class ModelCheckpoint(ModelCheckpointBase):

    def __init__(self, architecture, **kwargs):
        super(ModelCheckpoint, self).__init__(save_weights_only=False, **kwargs)
        self.architecture = architecture

    def on_epoch_end(self, epoch, logs=None):
        super(ModelCheckpointBase, self).set_model(self.architecture)
        super(ModelCheckpoint, self).on_epoch_end(epoch, logs)