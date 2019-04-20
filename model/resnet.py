from keras.applications.resnet_v2 import ResNet152V2
from keras.models import Model


class EncoderResNet152():

    def __init__(self, weights='imagenet'):
        self.weights = weights
        self.model = None
        self._build()

    def _build(self):
        # from pretrained model
        image_model = ResNet152V2(include_top=True, weights=self.weights)
        transfer_layer = image_model.get_layer('avg_pool')
        self.model = Model(
            inputs=image_model.input, outputs=transfer_layer.output)

    def save(self, path, overwrite):
        pass

    def load(self, path):
        pass
