# Spaceship Titanic classifier

---


In this project, we aim to utilize data science techniques to predict which passengers of the Spaceship 
Titanic were transported to an alternate dimension after a collision with a spacetime anomaly.
Spaceship Titanic is dataset about anomaly because almost half of the passengers were transported to an alternate dimension.
[More about Spaceship Titanic.](https://www.kaggle.com/competitions/spaceship-titanic)
- directed by `Daniil Lysenko 972202`

## About the project

---
`LiidClassifierModel` class represents the model used for predicting the transportation status of passengers. 
It is based on the [CatBoostClassifier](https://catboost.ai/), a gradient boosting decision tree library.

### Features
- **Imputation of Missing Values**: Various strategies are employed to handle missing data in the dataset, including mode imputation and encoding missing values as a specific out-of-range value.
- **Feature Engineering**: New features are created and existing ones are modified to enhance predictive power, such as extracting information from the passenger's cabin and one-hot encoding categorical features.
- **Model Training**: The model is trained using the [CatBoostClassifier](https://catboost.ai/) with hyperparameters tuned with [Optuna](https://optuna.org/).




## Usage

---

### API

1. Clone this repository to your local machine
```bash
git clone https://github.com/L11D/Spaceship_Titanic.git
cd Spaceship_Titanic
```

2. Install poetry
```bash
pip install poetry
```
3. Install requirement packages
```bash
POETRY_VIRTUALENVS_CREATE=false poetry install --no-interaction --no-ansi
```

4. Run api
```bash
python api.py
```
5. Train model. Response must contain dataset = `*.csv`
```bash
http://172.0.0.1:5000/train
```

6. Predict. Response must contain dataset = `*.csv`
```bash
http://172.0.0.1:5000/predict
```

### Docker

1. Clone this repository to your local machine
```bash
git clone https://github.com/L11D/Spaceship_Titanic.git
cd Spaceship_Titanic
```

2. Install poetry
```bash
docker-compose up -d
```
3. Train model. Response must contain dataset = `*.csv`
```bash
http://172.0.0.1:5000/train
```

4. Predict. Response must contain dataset = `*.csv`
```bash
http://172.0.0.1:5000/predict
```
---

### CLI

1. Clone this repository to your local machine
```bash
git clone https://github.com/L11D/Spaceship_Titanic.git
cd Spaceship_Titanic
```

2. Install poetry
```bash
pip install poetry
```

3. Install requirement packages
```bash
POETRY_VIRTUALENVS_CREATE=false poetry install --no-interaction --no-ansi
```

4. Train model
```bash
python main.py train /path/to/your/train/dataset.csv
```

5. Predict
```bash
python main.py predict /path/to/your/test/dataset.csv
```
6. Predictions will be saved to `./data/results.csv`

### Docker

1. Clone this repository to your local machine
```bash
git clone https://github.com/L11D/Spaceship_Titanic.git
cd Spaceship_Titanic
```
2. Create docker image
```bash
docker build -t spaceship_titanic:latest .
```
3. Train model
```bash
docker run -v /path/to/your/dataset/folder:/SpaceshipTitanic/data spaceship_titanic train data/your_train_dataset.csv
```
4. Predict
```bash
docker run -v /path/to/your/dataset/folder:/SpaceshipTitanic/data spaceship_titanic predict data/your_test_dataset.csv
```
5. Predictions will be saved to `path/to/your/dataset/folder/results.csv`