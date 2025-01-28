# Retinal-OCT-Classification

This project is a deep learning-based image classification system that detects diseases from medical images. The model is built and trained using **TensorFlow** and **Keras**, and it classifies images into one of eight categories.

## Dataset

The dataset used in this project contains images organized into eight classes:

- **AMD** (Age-related Macular Degeneration)
- **CNV** (Choroidal Neovascularization)
- **CSR** (Central Serous Retinopathy)
- **DME** (Diabetic Macular Edema)
- **DR** (Diabetic Retinopathy)
- **DRUSEN** (Drusen in the retina)
- **MH** (Macular Hole)
- **NORMAL** (Healthy retina)

## Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Numpy
- Matplotlib

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Mayank3000/Retinal-OCT-Classification-.git

##Model Architecture
The model is a Convolutional Neural Network (CNN) with the following layers:

1.Three convolutional layers with ReLU activation and max pooling.
2.A fully connected (dense) layer with 256 units and dropout for regularization.
3.A final softmax layer with 8 units corresponding to the number of classes.
##Data Augmentation
The training images are augmented using:

-Random flips
-Random rotations
-Random zoom
-Random contrast changes
-Normalization

##Training the Model
Training and validation datasets are loaded using image_dataset_from_directory.
The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as a metric.
##Save and Load the Model
The trained model is saved in HDF5 format:
model.save("disease_detection_model.h5")
To load the model for inference:
loaded_model = tf.keras.models.load_model("disease_detection_model.h5")
##Testing the Model
To test the model with a new image:
Preprocess the image to resize it to 224x224 and normalize pixel values.
Use model.predict() to obtain predictions.
Example:
from tensorflow.keras.preprocessing import image
img = image.load_img("data/test/DME/dme_test_1004.jpg", target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
predictions = loaded_model.predict(img_array)
print(f"Predicted Class: {predicted_class}")
##Evaluation
Evaluate the model on the validation dataset:
test_loss, test_accuracy = model.evaluate(val_dataset)
print(f"Validation Accuracy: {test_accuracy:.2f}")
##Results
The model achieved:
-Training Accuracy: ~95.8%
-Validation Accuracy: ~90.0%
