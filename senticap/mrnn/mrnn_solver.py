# import numpy as np
import theano
import theano.tensor as T

ADADELTA = "adadelta"
RMSPROP = "rmsprop"

ff = T.constant(1e-8, name="fudge_factor", dtype=theano.config.floatX)


def get_sgd_weight_updates(method, grads, learnable_params, hist_grad,
                           delta_grad, **kwargs):
    #do weight updates using adagrad
    comp_grads = grads
    if method == ADADELTA:
        rho = T.constant(kwargs["rho"], name="rho", dtype=theano.config.floatX)
        grad_sq_new = [
            rho * g_sq + (1 - rho) * (g**2)
            for g_sq, g in zip(hist_grad, comp_grads)
        ]

        deltas = [
            -(T.sqrt(d_sq + ff) / T.sqrt(g_sq + ff)) * grad
            for d_sq, g_sq, grad in zip(delta_grad, grad_sq_new, comp_grads)
        ]
        deltas_sq_new = [
            rho * d_sq + (1 - rho) * (d**2)
            for d_sq, d in zip(delta_grad, deltas)
        ]

        weight_updates = [(p, p + d) for p, d in zip(learnable_params, deltas)]
        weight_updates += list(zip(hist_grad, grad_sq_new))
        weight_updates += list(zip(delta_grad, deltas_sq_new))
    elif method == RMSPROP:
        decay = T.constant(kwargs["decay"],
                           name="decay",
                           dtype=theano.config.floatX)
        learning_rate = T.constant(kwargs["learning_rate"],
                                   name="learning_rate",
                                   dtype=theano.config.floatX)

        step_cache = [
            h_g * decay + (1.0 - decay) * g**2
            for h_g, g in zip(hist_grad, comp_grads)
        ]
        dx = [(learning_rate * g) / T.sqrt(sc + ff)
              for sc, g in zip(step_cache, comp_grads)]
        weight_updates = [(p, p - d) for p, d in zip(learnable_params, dx)]
        weight_updates += list(zip(hist_grad, step_cache))

    return weight_updates
