# SuperResAssess
## Project Title
Assessing the performance of deep-learned super-resolution of brain magnetic resonance images in small datasets.

## Project Description
This study will investigate different methods to assess deep-learned super-resolution of brain magnetic resonance images in small datasets.

## Environment install
This project uses poetry to manage dependencies. In case poetry is not installed you will need to install it with for example ```pip install poetry```. Than you can install the environment using ```poetry install```.

## Implementation Overview
For the comparison of the assessment methods, we create an abstract SuperResAssess base class. All assessment classes should inherit from this base class and implement the assess model method.