# Computer Vision - Plant Image CNN Classification
This project applied a custom Convolutional Neural Network (CNN) built in Keras and TensorFlow to classify plant images onto almond, cherry, and maize categories. The pipeline integrates OpenCV image preprocessing, data augmentation, and soft-max classification to process visual inputs and generate predictive class labels and confidence scores.

## Description
Plant image classification faces challenges due to high variability in lighting, leaf angles, and background noise. Training deep learning architectures on small datasets also requires significant preprocessing and tranformations in order to reduce overfitting and enable gradient flow.

This project implements an end-to-end computer vision workflow:
1. Data wrangling & preprocessing: Load multi-format image datasets (.jpg, .jpeg, and .png files), standardise RGB colour channels using OpenCV, and resize inputs to 224 x 224 pixels.
2. Object detection: Use dynamic data augmentation, integrating horizontal flips, spatial rotations, height/width zooms, and brightness adjustments within Keras Sequential layer so to increase visual variance during training.
3. Object classification: Construct a multi-stage feature extraction network using stacked Conv2D and MaxPooling2D layers with internal normalisation. Train using categorical crossentrpy loss and the Adam optimiser over 50 epochs and evaluate training performance using validation holds.
4. Model evaluation: Evaluate predictive ability on an unclassified testing set using probability outputs.

## Interpretation
### Object Classification
CNN architecture: 1,422,563 trainable parameters

Keras Sequential modelling yielded a total of 1,422,563 parameters suitable for training.

Training loss: Decreased from 1.262 (Epoch 1) to 0.120 (Epoch 50)
Training accuracy: Increased from 0.542 (Epoch 1) to 0.958 (Epoch 50)

Across 50 epochs, training loss dropped whilst training accuracy improved. Validation accuracy stabilised at 0.833 (loss = 0.320).

### Model Evaluation
Softmax confidence: Between 94% and 100%

Running inference against the 15 test sample generated class predictions with high softmax confidence.
Comparing predicted outputs to real labels revealed 6/15 correct predictions (40% test accuracy). The model demonstrated strong performance on cherry plant samples but overconfident misclassification between almond and maize plants. High confidence on incorrect predictions indicates model overfitting caused by the small sample size.

## Next Steps
1. Implement pre-trained transfer learning backbones to utlise pre-learned botanical visual features and reduce overconfidence.
2. Incorporate confusion matrix visualisation and out-of-fold cross-validation to isolate specific leaf characteristics leading to class misclassification.
