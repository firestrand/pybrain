""" plot the re-read results produced by femexperiments. """

__author__ = 'Tom Schaul, tom@idsia.ch'

import pylab    
from scipy import array, zeros, size, log, exp

from nesexperiments import pickleReadDict


if __name__ == '__main__':
    
    res = pickleReadDict('../temp/fem/dim15results-b')
    for k, val in res.items():
        allmuevals = filter(lambda x: max(x) > -1e-10, val[2])
        print k, len(val[2]), len(allmuevals)
        if len(allmuevals):
            maxlen = max(map(len, allmuevals))
            merged = zeros(maxlen)
            avgover = zeros(maxlen)
            for me in allmuevals:
                tmp = array(me)
                merged[:size(tmp)] -= tmp
                avgover[:size(tmp)] += 1
            merged /= avgover
            merged.clip(max = 1e10, min = 1e-10)
            x = array(range(maxlen))*25
            pylab.semilogy()
            pylab.plot(x, merged, label = k)
        
    pylab.ylabel('fitness')
    pylab.xlabel('number of evaluations')
    pylab.legend()
    pylab.show()
    
   