# Handwritten Digit Recognition Using Deep Learning on the MNIST Dataset

A Python-based deep learning project implementing a Sequential Artificial Neural Network (ANN) with dense fully-connected layers and dropout regularization to classify handwritten digits from the classic MNIST dataset.

---

## 📌 Project Overview
- **Problem Statement:** Handwritten digits have vast human variation in stroke, style, and thickness. This project provides a non-linear classification solution to map raw pixel matrices into accurate numerical classifications (digits 0-9).
- **Solution:** A fully-connected Multilayer Perceptron (MLP) built with Keras and TensorFlow that handles spatial unrolling and learns non-linear decision boundaries.
- **Full Documentation:** For an exhaustive 15-20 page structural explanation and mathematical breakdowns, see the attached [Technical Report PDF](./Deep%20Learning%20MNIST%20Project.docx).

---

## 🧠 Model Architecture

The model uses a feedforward **Sequential Artificial Neural Network (ANN)** consisting of the following specific layer stack:

```python
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
