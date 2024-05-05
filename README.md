# Machine Learning Model Training Application

This is a Python application that trains a Random Forest Classifier model using the `sklearn` library. The trained model is then saved using the `pickle` library.

## Getting Started

To get started with this project, you'll need to have Python 3.6 or later installed on your machine. You can download it from [python.org](https://www.python.org/downloads/).

Once you have Python installed, you can clone this repository and install the required packages using pip:

```bash
git clone https://github.com/yourusername/machine-learning-app.git
cd machine-learning-app
pip install -r requirements.txt
```

## Usage

To use the application, you'll need to provide a CSV file containing the training data. The file should have a target variable column and other feature columns.

You can run the application by executing the `train.py` script:

```bash
python train.py
```

The application will train a Random Forest Classifier model and save it to the `pickle_path/model.sav` file. It will also print the accuracy, precision, recall, and F1 score of the model.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.