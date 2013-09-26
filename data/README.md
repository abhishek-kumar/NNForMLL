DATASETS
--------

1. Dataset format
2. Sample datasets

1. Dataset format
------------------

Each dataset contains two comma separated (CSV) files - a train file and a test file.
Each CSV file contains `m` rows and `p + k` columns where `m` is the number of
instances, `p` is the number of features, and `k` is the number of labels.

Each row in the CSV file corresponds to one instance. The first `p` columns are
the feature values, and the last `k` columns are the label values.

The feature values are in the range `[-1, +1]` and the labels are binary, `{0, 1}`.

Example dataset with 2 instances, 3 features and 2 labels:

    0.1,0.3,-0.3,0,1
    -0.9,-0.5,0.4,1,0


2. Sample Datasets
------------------

When you run make, three sample datasets will be downloaded to this directory: `Scene`, `Yeast` and `Emotions`.
The datasets have been obtained from [Mulan](http://mulan.sourceforge.net/datasets.html).

| Dataset Name     | # Instances (m)      |  # Features (p)   | # Labels (k)   |
|------------------|:--------------------:|:-----------------:|---------------:|
| Scene|2407|294|6
| Yeast|2417|103|14
| Emotions|593|72|6
| NN-test|4|3|2

Note: In order to convert from arff format, open the file in [Weka](http://www.cs.waikato.ac.nz/ml/weka/) and save as
CSV. Then, open in a text editor and delete the header and remove all quotes and
spaces.
