# Ovarian Subtype Cancer Classification System

This project implements a deep learning-based classification system for identifying different subtypes of ovarian cancer. The system utilizes a ResNet model trained on a dataset obtained from Kaggle. It is deployed as a web application using Streamlit for ease of testing and evaluation.

## Features

- Classifies ovarian cancer subtypes into five categories: EC, CC, LGSC, MC, HGSC.
- Utilizes a pre-trained ResNet model for image classification.
- Provides a user-friendly interface for uploading images and viewing classification results.
- Enables easy testing and evaluation of the classification system.

## Dataset

The dataset used for training the model was obtained from Kaggle. It contains images of ovarian tissue samples labeled with one of the five ovarian cancer subtypes: EC (Endometrioid Carcinoma), CC (Clear Cell Carcinoma), LGSC (Low-Grade Serous Carcinoma), MC (Mucinous Carcinoma), and HGSC (High-Grade Serous Carcinoma).
link: https://www.kaggle.com/datasets/sunilthite/ovarian-cancer-classification-dataset/data

## Model Architecture

The classification system employs a ResNet model architecture for image classification. ResNet (Residual Neural Network) is a deep learning architecture known for its effectiveness in handling deep networks and addressing the vanishing gradient problem.

## Deployment

The classification system is deployed as a web application using Streamlit. Streamlit provides an intuitive and interactive interface for users to upload images, trigger classification, and view the results. The deployment process ensures accessibility and ease of use for testing and evaluation purposes.

## Usage

To use the classification system:

1. Clone the repository to your local machine.
2. Install the required dependencies specified in the `requirements.txt` file.
3. Run the Streamlit application using the command `streamlit run mainapp.py`.
4. Upload an image containing ovarian tissue sample.
5. View the classification result displayed on the web application.

## Contributors

- [David Sam] 

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Acknowledgments

Special thanks to Kaggle for providing the dataset used in this project, and to the Streamlit development team for creating an excellent framework for building web applications in Python.
