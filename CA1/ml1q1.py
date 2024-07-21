import numpy
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
dataarray = [1,1.1,1.2,1.5,0.9,0.7,0.5,1.005]
xaxis = numpy.arange(-2,2,0.01)
variance =numpy.var(dataarray)
average = numpy.mean(dataarray)
print(variance)
print(average)
plt.plot(xaxis, norm.pdf(xaxis,average,numpy.sqrt(variance)))
plt.show()
value = stats.norm.pdf(-1,average,numpy.sqrt(variance))
print(value)
random = numpy.random.normal(average, numpy.sqrt(variance),1000)
averagenew = numpy.mean(random)
print(averagenew)
