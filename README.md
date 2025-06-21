# Google Derm-foundation-model-covid-19-detection
Paper Title ( IEEE)
Transfer Learning-Based COVID-19 Detection Using Chest X-Ray Images with a Pretrained Neural Network and Custom Classifier

# Study Objection 
The objective of this study is to develop and evaluate a machine learning-based COVID-19 detection system using chest X-ray images. By leveraging a pretrained model for feature extraction and a custom neural network classifier, the system aims to accurately classify chest X-ray images as either "Normal" or "COVID." The study focuses on optimizing the workflow, including data preprocessing, feature extraction, model training, and visualization, to ensure reliability and interpretability of the results. This research seeks to contribute to the growing field of AI-driven diagnostic tools for early and efficient detection of COVID-19.


# Raw dataset 
Dataset Credit (Public Dataset)
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
*COVID-19 CHEST X-RAY DATABASE

A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal and other lung infection dataset is released in stages. In the first release we have released 219 COVID-19, 1341 normal and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection) and 1345 Viral Pneumonia images. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients.

# Setup Instructions

## Step 1: Create and Activate Virtual Environment
1. Open your terminal and navigate to the project directory.
2. Run the following commands to create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

## Step 2: Install Required Packages
1. Ensure you are in the virtual environment.
2. Run the following command to install the required packages:
   ```powershell
   pip install -r requirements.txt
   ```

## Step 3: Download Dataset
1. Visit the following link to download the dataset:
   [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
2. Extract the dataset and ensure only the `Covid` and `Normal` folders are included.
3. Place these folders inside the `Dataset` directory in the project.

## Step 4: Download Pretrained Models
1. Download the pretrained models from the following links:
   - [Google Derm Foundation Model](https://huggingface.co/google/derm-foundation)
   - [VGG19 Model](https://huggingface.co/keras-io/VGG19/tree/main)
   - [ResNet50 Model](https://huggingface.co/microsoft/resnet-50)
2. Extract the downloaded files and place them in their respective directories:
   - `GoogleDermModel`
   - `VGG19`
   - `Resnet50`

## Step 5: Run Classification Script
1. Execute the classification script to train and evaluate the models:
   ```powershell
   python classification.py
   ```

## Step 6: Run Cross-Validation Script
1. After completing the classification step, run the cross-validation script:
   ```powershell
   python cross_validation.py
   ```

# Additional Setup Instructions

## Step 7: Delete Temporary Files
1. After cloning the project, delete all `read.txt` files from the directories to clean up temporary files.

