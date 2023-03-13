import numpy as np

from random import random
from typing import List
from dataclasses import dataclass, field


def random_points(n_points):
    points = [(2 * random() - 1, 2 * random() - 1) for _ in range(n_points)]
    targets = [int(point[0] > point[1]) for point in points]
    return np.array(points), np.array(targets)

def step(value, threshold=0.5) -> int:
    return 1 if value > threshold else 0

@dataclass
class Perceptron:
    n_inputs: int = 2
    bias: float = 0.1
    threshold: float = 0.1
    learning_rate: float = 0.1
    verbose:bool = False
    weights: np.ndarray = field(init=False)

    def __post_init__(self):
        self.weights = np.random.rand(1, self.n_inputs) 

    def predict(self, input: List[float], target: int) -> int:
        weighted_sum = np.dot(self.weights, input) + self.bias
        prediction = step(weighted_sum, self.threshold)
        if self.verbose:
            print(f"{weighted_sum=} | {prediction==target=}")
        return prediction

    def fit(self, input, target) -> int:
        prediction = self.predict(input, target)
        for i, w in enumerate(self.weights):
            self.weights[i] = w +  self.learning_rate * (target - int(prediction)) * input
            if self.verbose:
                print(f"old weight: {w} | new weight: {self.weights[i]}")
        
        return prediction

    def train(self, inputs, targets) -> float:
        misses = 0
        for input, target in zip(inputs, targets):
            predicted = self.fit(input, target)
            prediction_is_true = target == predicted
            misses += 0 if prediction_is_true else 1
        if self.verbose:
            print(f"from {len(inputs)} set of inputs, the model missed {misses}.")
        return misses / len(inputs)


def main():
    x, y = random_points(200)
    p = Perceptron(verbose=False)
    loss = p.train(x, y)
    print(f"your loss is: {loss}")

if __name__ == "__main__":
    main()
