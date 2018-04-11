## Task11

First, excute command:
```python
make clean && make install
```
to clean unneeded files and install all the packages needed. 

#### Knn
Two python files: knnA.py for task A and knnB.py for task B
You can provide arguments to specify which test to run.

###### Usage
If you want to classify a test dataset in task A, use the following command (All inputs should be string and dataset should be csv file):

```python
python3 knnA.py 'prediction' training_data_filepath test_data_filepath
```

If you want to do cross-validation of a dataet in task A, use the following command:

```python
python3 knnA.py 'validation' training_data_filepath test_data_filepath
```

If you want to classify a test dataset in task B using real number dataset, use the following command:

```python
python3 knnB.py 'real' 'prediction' training_data_filepath test_data_filepath
```

If you want to do cross-validation of a dataet in task B using real number dataset, use the following command:

```python
python3 knnB.py 'real' validation' training_data_filepath test_data_filepath
```

Similarly, if you want to test binary dataset, just change the 'real' to 'binary' in the above command, and run.

#### DecisionTree
There are two files, one for task A and one for task B. 
There are three system arguments, action (could only be 'classify' or 'crossvalidation'), training data filepath, test data filepath.

###### Usage

If you want to classify a test dataset in task A, use the following command (All inputs should be string and dataset should be csv file):

```python
python3 decision_tree_A.py 'classify' training_data_filepath test_data_filepath
```

If you want to do cross-validation of a dataet in task A, use the following command:

```python
python3 decision_tree_A.py 'crossvalidation' training_data_filepath None
```

Notice that for cross-validation, the third system argument should be None.

If you want to classify a test dataset in task B, use the following command:

```python
python3 decision_tree_B.py 'classify' training_data_filepath test_data_filepath
```

If you want to do cross-validation of a dataet in task B, use the following command:

```python
python3 decision_tree_B.py 'crossvalidation' training_data_filepath None
```
