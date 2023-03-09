import numpy as np
from typing import List
from dataclasses import dataclass


def step(value, threshold=0.5) -> bool:
    return 1 if value > threshold else 0

@dataclass
class Perceptron:
    n_inputs: int = 2
    bias: float = 0.5
    threshold: float = 0.1
    learning_rate: float = 0.1
    weights: List[int] = np.random.rand(1, n_inputs)
    verbose:bool = False


    def predict(self, inputs: List[float], target: bool) -> bool:
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        prediction = step(weighted_sum, self.threshold)
        if self.verbose:
            print(f"{weighted_sum=} | {prediction==target=}")
        return prediction

    def fit(self, input, target) -> bool:
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
            misses += int(predicted)
        if self.verbose:
            print(f"from {len(inputs)} set of inputs, the model missed {misses}.")
        return misses / len(inputs)


def main():
    x = np.array ([ [1, 1],
                    [2, 2],
                    [3, 3],
                    [4, 4],
                    [5, 5] ])
    y = np.array([  [0],
                    [0],
                    [0],
                    [0],
                    [1] ])

    p = Perceptron(verbose=True)
    p.predict(x[0], y[0])
    p.fit(x[0], y[0])
    loss = p.train(x, y)

if __name__ == "__main__":
    main()
