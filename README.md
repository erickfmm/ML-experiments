# Machine Learning Experiments
A simple and experimental project for a bunch of machine learning codes and utils

# Download data
Data will downloaded programatically when using download() function of loaders. It will prompt to the user the username and key of kaggle.json.

# Execution

## Execute:  Option 1 - Using Docker 
Install Docker

1. run build.bat (Works with sh too)
2. run run.bat (Works with sh too)

## Execute: Option 2 - Using local enviroment
Install python, pip and run 

1. `python -m venv venv`
2. Activate venv enviroment, example: `.\venv\Scripts\Activate.ps1`
3. run `pip install -r requirements.txt`
4. run `python call_test.py`
4. or run `python call_test_gui.py` if you want a gui

## Execute: Option 3 - Install as a package
You can install as a package (it doesn't include test/ folder) using pip:

`pip install git@https://github.com/erickfmm/ML-experiments.git`

Then just import the module you want, for example (see test files): 

```python
import mlexperiments.unsupervised.clustering.cluster_sklearn as clustering_sk

labels1 = [int(i/1000) for i in range(4000)]
np.random.seed(844)
clust1 = np.random.normal(5, 2, (1000, 2))
clust2 = np.random.normal(15, 3, (1000, 2))
clust3 = np.random.multivariate_normal([17, 3], [[1, 0], [0, 1]], 1000)
clust4 = np.random.multivariate_normal([2, 16], [[1, 0], [0, 1]], 1000)
simple_dataset = np.concatenate((clust1, clust2, clust3, clust4))

assignments = clustering_sk.dbscan(dataset1)
```

# TODO (ideas):

- [ ] More loaders
- [ ] Documentation (pydoc or similar)
- [ ] Documentation - UML
- [x] Better menu (call_test), maybe using some ncurses or similar
- [ ] Documentation inside test/ files
- [ ] More tests (more ML models and experiments)


# Screenshots

By now I only have 2 interfaces. The main and one experiment:

Main interface:

![selector of test file](https://github.com//erickfmm/ML-experiments/blob/master/docs/selector_of_test.png?raw=true)

Segmentator of Butterflies:
![Segmentation of Butterflies](https://github.com//erickfmm/ML-experiments/blob/master/docs/segmentator_of_butterflies.png?raw=true)