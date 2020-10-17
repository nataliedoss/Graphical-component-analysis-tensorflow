# Graphical Component Analysis via Deep Energy Estimator Networks
This is a TensorFlow implementation of the algorithm described in Section 4.5 of <a href="https://github.com/nataliedoss/Thesis/blob/master/main.pdf" download>Chapter 4: Graphical component analysis for latent signal detection</a>. As in the original variant, <a href="https://github.com/nataliedoss/Graphical-component-analysis-R" download>Graphical component analysis (GCA)</a>, we allow for latent signal detection when the hidden source components have dependence modeled by a pairwise graphical model structure. But in this version, the energy function of the latent source components is represented using a neural network (multilayer perceptrons). In addition, the unitary de-mixing matrix is represented using the FFT-style approximation of Michael Mathieu and Yann LeCun's [Fast Approximation of Rotations and Hessians matrices](https://arxiv.org/abs/1404.7195). This matrix is then optimized using the method of [Tunable Efficient Unitary Neural Networks (EUNN) and their application to RNNs](https://arxiv.org/pdf/1612.05231.pdf). The neural network parameters are estimated using stochastic gradient descent; the implementation is in TensorFlow. A demonstration of the algorithm is provided in a Jupyter notebook, [Graphical component analysis via deep energy estimator networks](https://github.com/nataliedoss/Graphical-component-analysis-tensorflow/blob/master/gca.ipynb).


## External dependencies

[Numpy](http://numpy.org/)

[Math](https://docs.python.org/3.0/library/math.html)

[TensorFlow](https://www.tensorflow.org/)
