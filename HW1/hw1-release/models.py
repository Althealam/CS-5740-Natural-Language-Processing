from collections import Counter
import math
import random
from typing import List
from data import Example
from features import FeatureExtractor

class Classifier:
    def predict(self, ex_words: list[str]) -> int:
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: list[list[str]]) -> list[int]:
        return [self.predict(ex_words) for ex_words in all_ex_words]

class TrivialClassifier(Classifier):
    def predict(self, ex_words: list[str]) -> int:
        return 1


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    else:
        exp_z = math.exp(z)
        return exp_z / (1.0 + exp_z)


class LogisticRegressionClassifier(Classifier):
    def __init__(self, feat_extractor: FeatureExtractor):
        # Keep the feature extractor and initialize weights/bias to 0.
        self.feat_extractor = feat_extractor
        self.weights: dict[str, float] = {feat: 0.0 for feat in feat_extractor.vocabulary}  # maps feature name -> weight
        self.bias: float = 0.0  # model bias term

    def predict(self, sent: list[str]) -> int:
        """ This function will predict the label (return 0/1) for a given input. 
            It should convert the sentence into a feature dictionary, predict the probability of the two labels, 
            and return the label prediction
        """
        features = self.feat_extractor.extract_features(sent)
        z = self.bias + sum(self.weights.get(feat, 0.0) * val for feat, val in features.items())
        prob = sigmoid(z)
        return 1 if prob >= 0.5 else 0

    def train(self, train_exs: List[Example], learning_rate=0.01, epochs=10, val_exs=None):
        """
            This is the main function to train the logistic regression model. 
        """
        train_losses = []
        train_accs = []
        val_accs = []

        for epoch in range(epochs):
            random.shuffle(train_exs)
            total_loss = 0.0

            for ex in train_exs:
                # 1. Compute prediction: y_hat = sigmoid(w·x + b)
                features = self.feat_extractor.extract_features(ex.get_combined_words())
                z = self.bias + sum(self.weights.get(feat, 0.0) * val for feat, val in features.items())
                y_hat = sigmoid(z)

                # 2. Compute loss (cross-entropy)
                loss = cross_entropy_loss(y_hat, ex.label)
                total_loss += loss

                # 3. Compute the gradients of the weights and bias
                # For cross-entropy loss with sigmoid: dL/dw_k = (y_hat - y) * f_k, dL/db = (y_hat - y)
                grad = y_hat - ex.label

                # 4. Update the weight and bias using SGD
                for feat, val in features.items():
                    if feat in self.weights:
                        self.weights[feat] -= learning_rate * grad * val
                self.bias -= learning_rate * grad

            avg_loss = total_loss / len(train_exs)
            train_acc = evaluate(self, train_exs)
            train_losses.append(avg_loss)
            train_accs.append(train_acc)

            print(f"Epoch {epoch + 1}/{epochs} - Avg Loss: {avg_loss:.4f} - Train Acc: {train_acc:.4f}", end="")
            if val_exs:
                val_acc = evaluate(self, val_exs)
                val_accs.append(val_acc)
                print(f" - Val Acc: {val_acc:.4f}", end="")
            print()

        return train_losses, train_accs, val_accs


def cross_entropy_loss(y_hat: float, y: int, eps: float = 1e-10) -> float:
    return -(y * math.log(y_hat + eps) + (1 - y) * math.log(1 - y_hat + eps))


def evaluate(classifier, examples: List[Example]) -> float:
    """
    Evaluates a classifier on a set of examples.
    
    :param classifier: trained classifier
    :param examples: list of examples to evaluate on
    :return: accuracy
    """
    correct = 0
    predictions = classifier.predict_all([ex.get_combined_words() for ex in examples])
    
    for pred, ex in zip(predictions, examples):
        if pred == ex.label:
            correct += 1
    
    accuracy = correct / len(examples) if examples else 0.0
    return accuracy


def print_evaluation_metrics(classifier, examples: List[Example], dataset_name: str):
    """
    Prints detailed evaluation metrics for entailment.
    """
    predictions = classifier.predict_all([ex.get_combined_words() for ex in examples])
    
    true_pos = sum(1 for pred, ex in zip(predictions, examples) if pred == 1 and ex.label == 1)
    true_neg = sum(1 for pred, ex in zip(predictions, examples) if pred == 0 and ex.label == 0)
    false_pos = sum(1 for pred, ex in zip(predictions, examples) if pred == 1 and ex.label == 0)
    false_neg = sum(1 for pred, ex in zip(predictions, examples) if pred == 0 and ex.label == 1)
    
    accuracy = (true_pos + true_neg) / len(examples) if examples else 0.0
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    True Pos: {true_pos}, False Pos: {false_pos}")
    print(f"    False Neg: {false_neg}, True Neg: {true_neg}")
