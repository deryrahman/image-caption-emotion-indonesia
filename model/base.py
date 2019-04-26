import numpy as np
import re


class RichModel():

    def _build(self):
        raise NotImplementedError()

    def save(self, path, overwrite):
        raise NotImplementedError()

    def load(self, path):
        raise NotImplementedError()

    def predict(self, **kwargs):
        raise NotImplementedError()

    def _get_weight_values(self, layer_name, weight_name=None):
        """Helper for get weight values based on layer name and weight name

        Arguments:
            layer_name {str} -- layer name

        Keyword Arguments:
            weight_name {str} -- weight name, if None it means all weight from specific layer 
            will be returned (default: {None})

        Returns:
            np array -- np array with rank 4. (layer_numbers, weight_numbers, dim_weight, dim_weight)
        """
        layer_name_pattern = re.compile(layer_name)
        weights_name_pattern = re.compile(weight_name if weight_name else '')

        weight_values = []

        if weight_name is None:
            weight_values = [
                l.get_weights()
                for l in self.model.layers
                if layer_name_pattern.search(l.name)
            ]
        else:
            for l in self.model.layers:
                if layer_name_pattern.search(l.name) is None:
                    continue

                layer_weights = zip(l.weights, l.get_weights())
                weights = [
                    val for w, val in layer_weights
                    if weights_name_pattern.search(w.name)
                ]

                weight_values.append(weights)

        return np.array(weight_values)

    def _set_weight_values(self, layer_name, weight_values, weight_name=None):
        """Helper to set weights based on layer name and weight name

        Arguments:
            layer_name {str} -- targeted layer
            weight_values {np.array} -- weights value to be inserted

        Keyword Arguments:
            weight_name {str} -- targeted weight (default: {None})
        """
        layer_name_pattern = re.compile(layer_name)
        weights_name_pattern = re.compile(weight_name if weight_name else '')

        target = self._get_weight_values(
            layer_name=layer_name, weight_name=weight_name)
        assert target.shape == weight_values.shape

        layer_target_i = 0
        for i, l in enumerate(self.model.layers):
            if layer_name_pattern.search(l.name) is None:
                continue

            weights = []
            if weight_name is None:
                weights = weight_values[layer_target_i]
            else:
                layer_weights = zip(l.weights, l.get_weights())
                weight_target_i = 0

                for w, val in layer_weights:
                    if weights_name_pattern.search(w.name):
                        weights.append(
                            weight_values[layer_target_i][weight_target_i])
                        weight_target_i += 1
                    else:
                        weights.append(val)

            self.model.layers[i].set_weights(weights)
            layer_target_i += 1
