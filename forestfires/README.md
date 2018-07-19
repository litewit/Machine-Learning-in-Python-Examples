# Machine Learning in Python Examples for Forest Fires Data Set

## Forest fires example

This classification has been done on the dataset of forest fires taken from [UIC Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

In `classification.py` we have used simple classification



In `classification-tpot.py` we have user python [tpot](https://epistasislab.github.io/tpot/) library.

## Automated Machine Learning tool 
TPOT is a Python Automated Machine Learning tool that optimizes machine learning pipelines using genetic programming.

TPOT is built on top of several existing Python libraries, including:
- NumPy
- SciPy
- scikit-learn
- DEAP
- update_checker
- tqdm
- stopit
- pandas

NumPy, SciPy, scikit-learn, and pandas can be installed with `pip` via the command:

`pip3 install numpy scipy scikit-learn pandas`

DEAP, update_checker, tqdm and stopit can be installed with `pip` via the command:

`pip3 install deap update_checker tqdm stopit`

Finally to install TPOT itself, run the following command:

`pip3 install tpot`

For more see the [tpot docs](https://epistasislab.github.io/tpot/installing/)
