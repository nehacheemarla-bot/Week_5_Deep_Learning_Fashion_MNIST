# Week 5 - Deep Learning Application

## Project Overview

This project was completed as part of my Virtual Data Science with Python internship.

For Week 5, I worked on a deep learning image classification problem using the Fashion-MNIST dataset. I built a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify clothing images into ten different categories.

The main aim was to understand how a CNN can learn image features and how its performance can be evaluated using training curves, classification metrics and a confusion matrix.

## Dataset

The Fashion-MNIST dataset contains grayscale images of clothing and accessories.

- Training images: 60,000
- Test images: 10,000
- Image size: 28 × 28 pixels
- Number of classes: 10
- Pixel values were normalized from 0–255 to 0–1
- Each training class contains 6,000 images

### Classes

1. T-shirt/top
2. Trouser
3. Pullover
4. Dress
5. Coat
6. Sandal
7. Shirt
8. Sneaker
9. Bag
10. Ankle boot

## Methodology

The following steps were followed:

1. Loaded the Fashion-MNIST dataset using TensorFlow/Keras.
2. Inspected the image dimensions, labels and class distribution.
3. Normalized pixel values to the range 0–1.
4. Added a channel dimension for CNN input.
5. Built a CNN with two convolutional and pooling blocks.
6. Added a dense layer and dropout for classification and regularization.
7. Trained the model for 10 epochs with a batch size of 64.
8. Used 10% of the training data for validation.
9. Evaluated the trained model on 10,000 unseen test images.
10. Analysed class-wise precision, recall and F1-score.
11. Created a confusion matrix to study classification errors.
12. Inspected incorrect predictions and prediction confidence.
13. Saved the trained model in Keras format.

## CNN Architecture

The main architecture was:

```text
Input (28 × 28 × 1)
        ↓
Conv2D (32 filters, 3 × 3, ReLU)
        ↓
MaxPooling2D
        ↓
Conv2D (64 filters, 3 × 3, ReLU)
        ↓
MaxPooling2D
        ↓
Flatten
        ↓
Dense (128, ReLU)
        ↓
Dropout (0.3)
        ↓
Dense (10, Softmax)
```

The model contained **225,034 trainable parameters**.

## Results

The model achieved the following results from the actual training run:

| Metric | Result |
|---|---:|
| Training Accuracy | 93.04% |
| Best Validation Accuracy | 91.93% |
| Test Accuracy | 90.51% |
| Test Loss | 0.2652 |
| Correct Test Predictions | 9,051 / 10,000 |
| Incorrect Test Predictions | 949 / 10,000 |
| Macro F1-score | 0.90 |

The best validation accuracy of **91.93%** was reached at Epoch 9.

## Classification Analysis

The model performed particularly well on classes such as:

- Trouser
- Sandal
- Sneaker
- Bag
- Ankle boot

The **Shirt** class was the most difficult category.

For Shirt:

- Precision: 0.79
- Recall: 0.64
- F1-score: 0.71

The confusion matrix showed that Shirt images were often predicted as T-shirt/top, Pullover or Coat. These categories have visually similar shapes, which makes them harder to distinguish in small 28 × 28 grayscale images.

## Overfitting Observation

Training accuracy continued to increase throughout the ten epochs, while validation accuracy reached its highest point at Epoch 9 and then decreased slightly at Epoch 10.

This indicates some mild overfitting near the end of training. Dropout was included in the model to help reduce overfitting.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## Repository Files

| File | Description |
|---|---|
| `Week_5_Deep_Learning_Fashion_MNIST.ipynb` | Complete Colab notebook with code and outputs |
| `Week_5_Deep_Learning_Fashion_MNIST_Report_HUMAN_MADE_FINAL.docx` | Detailed project report |
| `fashion_mnist_cnn.keras` | Saved trained CNN model |
| `README.md` | Project documentation |

## Limitations

The model was trained and evaluated only on Fashion-MNIST. The images are small grayscale images, so the results should not be directly generalized to real-world clothing images.

The project also focused on one main CNN architecture rather than comparing several deep learning architectures.

## Possible Improvements

Some possible improvements for future work are:

- Use suitable image augmentation.
- Apply early stopping.
- Tune the learning rate and dropout rate.
- Experiment with different CNN architectures.
- Compare the CNN with transfer-learning approaches.
- Focus on improving recognition of visually similar classes such as Shirt, Coat and Pullover.

## Conclusion

This project provided practical experience with a complete deep learning workflow, from image preprocessing and CNN design to model training and evaluation.

The final CNN achieved **90.51% test accuracy** on 10,000 unseen Fashion-MNIST images. The confusion matrix and class-level metrics also showed that the model performs differently across categories, with visually similar clothing classes being more difficult to classify.

The trained model was saved as `fashion_mnist_cnn.keras` for future use.
