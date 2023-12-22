# ML-Sentiment
Create to Deploy ML Sentiment Models Using Flask.

This guide provides step-by-step instructions on how to run this models in your own local environment.

## Prerequisites

Make sure you have the following installed:

- Python (version 3.10.12) 
- Pip (Python package installer)
- Anaconda or Miniconda (for managing Python environments) : [Anaconda Docs](https://docs.anaconda.com/free/anaconda/install/)

## Step 1: Clone the Repository

Clone this repository to your local machine using the following command:

```bash
git clone <repository_url>
```

## Step 2: Create and Active conda virtual env
```bash
cd your_project_directory
conda create --name my_env python=3.10.12
```
- On Windows :
```bash
conda activate my_env
```
- On MacOs or Linux :
```bash
source activate my_env
```

## Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

## Step 4: Starting a Flask API with a Trained Model
```bash
python app.py
```

## Step 5: Access the Model/Application
For web-based applications or APIs, access them through the specified endpoints or URLs. 
For example: [http://localhost:5000](http://localhost:5000)
