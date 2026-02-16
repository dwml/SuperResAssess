# SuperResAssess
## Project Title
Assessing the performance of deep-learned super-resolution of brain magnetic resonance images in small datasets.

## Project Description
This study will investigate different methods to assess deep-learned super-resolution of brain magnetic resonance images in small datasets.

## Environment install
This project uses poetry to manage dependencies. In case poetry is not installed you will need to install it with for example ```pip install poetry```. Then you can install the environment using ```poetry install```.

## Usage
To see all possibilities of the assess command, use ```poetry run assess --help```. To replicate our experiment, use ```poetry run assess download``` to download HCP data. Note that the AWS CLI needs to be installed and the HCP credentials need to be setup to use it (please refer to HCP website). Then you can preprocess the images using ```poetry run assess preprocess```. Then the experiment needs to be setup using ```poetry run assess setup```. And finally you can run the experiment using ```poetry run assess run```.

## Figures
For recreation of all figures, please refer to the scripts folder. Here you'll find Jupyter notebook scripts to recreate the images. In order to do so you need to have executed the previous poetry run commands up to and including the preprocess commands. Additionally, you'll need to copy the log.zip file from our [OSF project]{https://doi.org/10.17605/OSF.IO/J6QXD} and extract it in the logs folder.

## Implementation Overview
For the comparison of the assessment methods, we create an abstract SuperResAssess base class. All assessment classes should inherit from this base class and implement the assess-model method.