# **Deep Learning Projects**

## 1. **Digit Recognition with MNIST Dataset**

### Overview
This project demonstrates how to build a simple neural network to recognize handwritten digits using the MNIST dataset. The model is built with TensorFlow/Keras.

### Project Structure
```
Digit_Recognition_MNIST/
├── digit_recognition_mnist.ipynb    # Jupyter notebook with ANN implementation
├── mnist_data.csv                   # MNIST dataset (or use Keras dataset)
└── README.md                        # Project documentation
```

### Dataset
The MNIST dataset contains 70,000 grayscale images of handwritten digits (0-9), split into training and test sets.

### Model
- **Input:** 784 pixels (28x28 images flattened)
- **Hidden Layers:** 2 Dense layers with ReLU activation
- **Output Layer:** Softmax activation for multi-class classification
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam

### How to Run
1. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas matplotlib
   ```
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook digit_recognition_mnist.ipynb
   ```
3. Run all cells to train and evaluate the model.

---

## 2. **Heart Disease Prediction with Artificial Neural Network (ANN)**

### Overview
This project demonstrates how to build an ANN for predicting heart disease risk using medical data. The model is built with TensorFlow/Keras.

### Project Structure
```
Heart_Disease_Prediction_with_ANN/
├── Heart_Disease_Prediction_with_ANN.ipynb     # Jupyter notebook with ANN implementation
├── better_heart_disease_data.csv               # Enhanced synthetic dataset
└── README.md                                    # Project documentation
```

### Dataset
The synthetic dataset includes 13 medical features and a target column (`has_disease`) indicating the presence of heart disease.

### Model
- **Input:** 13 medical features
- **Hidden Layers:** 2 Dense layers with ReLU activation
- **Output Layer:** Sigmoid activation for binary classification
- **Loss Function:** Binary Crossentropy
- **Optimizer:** Adam

### How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn tensorflow
   ```
2. Open the Jupyter notebook:
   ```bash
   jupyter notebook Heart_Disease_Prediction_with_ANN.ipynb
   ```
3. Run all cells to preprocess data, train the model, and evaluate performance.

