# 🐾 Dogs vs Cats Classifier

This project implements a **Convolutional Neural Network (CNN)** to classify images of cats and dogs using the **Kaggle Cats vs Dogs dataset**.  
It demonstrates data preprocessing, model training, and evaluation using TensorFlow and Keras.

---

## 📦 Dataset

The dataset comes from [Microsoft’s Kaggle Cats vs Dogs dataset](https://www.kaggle.com/c/dogs-vs-cats/data).

- Total images: ~25,000  
- Classes: 2 (Cats 🐱, Dogs 🐶)  
- Split: 80% Training / 20% Validation  
- All images are resized to **128×126** before being fed into the CNN.

---

## 🧠 Model Architecture

The network used in this project is a small CNN built from scratch with Keras:

```python
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(128, 126, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])
