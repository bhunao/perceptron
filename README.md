# Perceptron

This is a Python implementation of a Perceptron, a simple neural network that can classify `data` into two classes. It uses numpy for numerical operations and is trained on randomly generated data points.

The Perceptron class has these parameters:

- `n_inputs`: number of input features
- `bias`: bias term in the linear function
- `threshold`: threshold value used by the step function
- `learning_rate`: learning rate used in weight update rule
- `verbose`: boolean flag to enable verbose output

The class has two methods: predict and fit. The train function trains the Perceptron and returns the loss (fraction of misclassified points).

---
### usage:
```python
>>> x,y = random_points(200)
>>> p = Perceptron()
>>> loss = p.train(x, y)
>>> loss
0.07
```
