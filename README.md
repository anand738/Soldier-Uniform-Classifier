# Soldier Uniform Classifier

A deep learning-based application for classifying soldier uniforms using computer vision techniques. This project leverages convolutional neural networks (CNNs) to identify and categorize military uniforms from images, aiding in surveillance and defense applications.

## Features

- Image classification of soldier uniforms
- Utilizes CNNs for feature extraction and classification
- Includes a Streamlit web application for user interaction
- Jupyter notebooks for model training and data preprocessing

## Project Structure

```

Soldier-uniform-classifier/
├── app.py                 # Streamlit web application
├── model.ipynb            # Jupyter notebook for model training
├── rename.ipynb           # Jupyter notebook for data preprocessing
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.6 or higher
- pip package manager

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/anand738/Soldier-uniform-classifier.git
   cd Soldier-uniform-classifier
   ```

2. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```
   
## Usage

### Training the Model

1. **Prepare your dataset:**

   Ensure your dataset is organized and labeled appropriately.

2. **Preprocess the data:**

   Use the `rename.ipynb` notebook to preprocess and organize your dataset.

3. **Train the model:**

   Open and run the `model.ipynb` notebook to train your CNN model on the prepared dataset.

### Running the Web Application

1. **Start the Streamlit app:**

   ```bash
   Streamlit run app.py
   ```

2. **Access the application:**

   Navigate to `http://localhost:8501/` in your web browser to interact with the uniform classifier.
