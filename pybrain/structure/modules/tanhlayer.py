__author__ = 'Tom Schaul, tom@idsia.ch'

from scipy import tanh

from neuronlayer import NeuronLayer


class TanhLayer(NeuronLayer):
    """ A layer implementing the tanh squashing function. """

    def _forwardImplementation(self, inbuf, outbuf):
        outbuf[:] = tanh(inbuf)
        #speed up calculation by using continued fraction expansion
        #2.0 / (1.0 + math.exp(-2.0 * x)) - 1.0

    def _backwardImplementation(self, outerr, inerr, outbuf, inbuf):
        inerr[:] = (1 - outbuf**2) * outerr
