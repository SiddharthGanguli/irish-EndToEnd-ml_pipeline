# Iris Classification End-to-End ML Pipeline

This repository demonstrates a complete **end-to-end machine learning workflow** for classifying the Iris dataset using **Logistic Regression**, integrated with modern **MLOps practices** including:

- Data preprocessing
- Train/test splitting
- Model training and evaluation
- Model versioning and experiment tracking
- API deployment using FastAPI
- Dockerized deployment for reproducibility

The deployed API is accessible at: [Iris Classification API](https://irish-endtoend-ml-pipeline.onrender.com/)

Here is the fully furnished app: [Irish Prediction App](https://irish-pred.netlify.app/)


---

## Table of Contents

- [Project Overview](#project-overview)  
- [Data](#data)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
- [API Endpoints](#api-endpoints)  
- [Docker Deployment](#docker-deployment)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Overview

This project focuses on building a **scalable ML pipeline** with an emphasis on production readiness. The main goals include:

- **Reproducibility:** Use of Docker to encapsulate environment and dependencies.  
- **Version Control:** Data and model versioning with DVC and MLflow.  
- **API Deployment:** Serving predictions via FastAPI.  
- **Automation & MLOps:** Clear separation of training and inference, enabling continuous integration and deployment.

---

## Data

- **Dataset:** Iris dataset, containing 150 samples with 4 features:  
  - `sepal_length`  
  - `sepal_width`  
  - `petal_length`  
  - `petal_width`  
- **Target:** Species of Iris (`setosa`, `versicolor`, `virginica`) encoded as `0`, `1`, `2`.

---

## Project Structure
```
irish/
├── src/
│ ├── app/ # FastAPI application
│ │ ├── app.py
│ │ ├── schema.py
│ │ └── irishmodel/ # Saved model
│ ├── model/ # ML scripts: preprocessing, training, split
│ └── main.py
├── data/ # Raw & processed data
├── requirements.txt
├── Dockerfile
└── README.md
```
---

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/irish-ml-pipeline.git
cd irish-ml-pipeline
```
Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Usage
1️. Training

Run the training script to preprocess data, split, and train the model:
```bash
python src/model/main.py
```
2. Running the API Locally
   ```bash
   uvicorn src.app.app:app --reload

   ```
3. Docker Deployment
 ```bash
   docker build -t iris-api .
```
Run the container:
```bash
   docker run -p 8000:8000 iris-api
```

License

This project is licensed under the MIT License.

Made with ❤️ by Siddhartha Ganguli.
Thank You
