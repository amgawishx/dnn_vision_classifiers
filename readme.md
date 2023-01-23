## Project Description
This is an educational project to compare different neural network vision classifiers and their results for the college course (Machine Learing EC559). It is designed with strong modularity in mind so that anyone can quickly tweak a few parameters and get it running.

Main dependencies: 
- tensorflow
- kaggle
- numpy
- matplotlib
- sklearn
- Pillow
## Usage
To begin using this project there are two options:
- Provide your own data, in `config.py` change the parameter`ABS_DATA_PATH` to match the path of your data
- Download a dataset via _Kaggle_, this requires to have an account on the platform and an API key, configure the file `.kaggle/kaggle.json` to match your username and key, then in `config.py` modify the `KAGGLE_DATASET` variable to download the dataset you desire.

The next step is to simply run `main.py` file and setback because running these models is going to take quite a lot of time.

## Structural Description
- `main.py`: the entry point to the project, doesn't define anything new, just calls the different functions and methods in the other files and the libraries, models are compiled and trained in this file.
- `config.py`: doesn't contain any code at all, just global variables that define the entire parameters of the projects, the file act as settings to setup different configurations for the project.
- `models`: this folder contain the neural network models written using _TensorFlow_ functional API, add any new models you desire to this folder.
- `utility`: contain auxiliary scripts for data acquisition, preprocessing, post-processing and logging options.
- `data`: optionally contains the data used to run the train the models, this is where the data will be downloaded if the _Kaggle_ option used.
## Configuration Parameters
- `ABS_DATA_PATH`: absolute path to your training and testing data.
- `TRAIN_DATA_PATH`: the path to training data relative to `ABS_DATA_PATH`
- `TEST_DATA_PATH`: the path to test data relative to `ABS_DATA_PATH`
- `BATCH_SIZE`: the size of your batches.
- `VALIDATION_SPLIT`: defines how much of your training data is used for validation
- `IMAGE_SIZE`: the desired rescaling resolution for the input images done in preprocessing.
- `DROPOUT_RATE`: the dropout rate for any dropout layer across all models.
- `NO_LAYERS`: defines how deep is any neural network across all models.
- `OPTIMIZER`: specifies the GD algorithm to be used in compilation and training for all models.
- `LEARNING_RATE`: specifies the desired initial learning rate for training.
- `LOSS`: specifies the loss function used in training.
- `METRICS`: specifies which metrics _TensorFlow_ should output.
- `INPUT_SHAPE`: specifies whether the input images are colored or greyscale.
- `TARGET_ACCURACY`: specifies the accuracy which training will terminate upon reaching.
- `DECAY_RATE`: specifies the decay rate for the learning rate used in the adaptive training.
- `EPOCHS`: number of training epochs.
- `STEPS`: number of training steps per epoch.
- `VAL_STEPS`: number of validation steps per epoch.
- `NO_CLASSES`: the number of output classes for the data.
- `KAGGLE_DATASET`:  the name of the _Kaggle_ dataset to download.
- `LOG_FILE`: the name of the file where the runtime and results of the project will be logged.
- `VIT_*`: specifies parameters related to the ViTransformer models.
## Copyrights
If you find this project useful, feel free to use it however you like, just cite this repo when doing so.
