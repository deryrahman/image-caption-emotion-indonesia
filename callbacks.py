from keras.callbacks import Callback
import numpy as np


class ModelCheckpoint(Callback):

    def __init__(self, architecture, filepath, monitor='loss', verbose=1):
        super(ModelCheckpoint, self).__init__()
        self.architecture = architecture
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.monitor_op = np.less
        self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        filepath = self.filepath.format(epoch=epoch + 1)
        if current is not None:
            if self.monitor_op(current, self.best):
                if self.verbose > 0:
                    print(
                        '\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                        ' saving model to %s' % (epoch + 1, self.monitor,
                                                 self.best, current, filepath))
                self.best = current
                self.architecture.save(filepath)
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: %s did not improve from %0.5f' %
                          (epoch + 1, self.monitor, self.best))
