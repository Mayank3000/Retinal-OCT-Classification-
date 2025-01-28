# Retinal-OCT-Classification-
This project is a deep learning-based image classification system that detects diseases from medical images. The model is built and trained using TensorFlow and Keras. It classifies images into one of eight categories.

## Dataset
The dataset used in this project contains images organized into eight classes:
- **AMD**
- **CNV**
- **CSR**
- **DME**
- **DR**
- **DRUSEN**
- **MH**
- **NORMAL**

### Dataset Structure
The dataset is organized as follows:
data/ ├── train/ # Training dataset │ ├── AMD/ │ ├── CNV/ │ ├── ... │ └── NORMAL/ ├── val/ # Validation dataset │ ├── AMD/ │ ├── CNV/ │ ├── ... │ └── NORMAL/ └── test/ # Test dataset
## Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Numpy
- Matplotlib

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Mayank3000/Retinal-OCT-Classification-.git

Model Architecture
The model is a Convolutional Neural Network (CNN) with the following layers:

Three convolutional layers with ReLU activation and max pooling.
A fully connected (dense) layer with 256 units and dropout for regularization.
A final softmax layer with 8 units corresponding to the number of classes.
Data Augmentation
The training images are augmented using:

Random flips
Random rotations
Random zoom
Random contrast changes
Normalization
Images are normalized to the range [0, 1] by rescaling pixel values.

Training the Model
Training and validation datasets are loaded using image_dataset_from_directory.
The model is compiled with the Adam optimizer, categorical cross-entropy loss, and accuracy as a metric.
Training is performed for 10 epochs with real-time validation.

Run the following command to train the model:

bash
Copy
Edit
python train.py
Training Output
Sample training output:

python
Copy
Edit
Epoch 1/10
...
Epoch 10/10
Validation Accuracy: 0.90
Save and Load the Model
The trained model is saved in HDF5 format:

python
Copy
Edit
model.save("disease_detection_model.h5")
To load the model for inference:

python
Copy
Edit
loaded_model = tf.keras.models.load_model("disease_detection_model.h5")
Testing the Model
To test the model with a new image:

Preprocess the image to resize it to 224x224 and normalize pixel values.
Use model.predict() to obtain predictions.
Example:

python
Copy
Edit
from tensorflow.keras.preprocessing import image
img = image.load_img("data/test/DME/dme_test_1004.jpg", target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
predictions = loaded_model.predict(img_array)
print(f"Predicted Class: {predicted_class}")
Evaluation
Evaluate the model on the validation dataset:

python
Copy
Edit
test_loss, test_accuracy = model.evaluate(val_dataset)
print(f"Validation Accuracy: {test_accuracy:.2f}")
Results
The model achieved:

Training Accuracy: ~95.8%
Validation Accuracy: ~90.0%
