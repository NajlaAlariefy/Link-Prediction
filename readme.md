# Link Prediction Kaggle Competition
#### *Predicting whether an edge is real or imaginary*
6 Sept, 2018

## Summary
This part of the project contains the data and feature engineering pipelines. Model building & selection were done using the Weka Workbench version 3.8.2.

Ensembles of Decision Trees: Random Forest and AdaBoostM1 (with Decision Stump) were used. In both cases, the model was evaluated using 10-fold cross validation, and for competition submissions, trained on the entire training set.

Preprocessing in Weka is straightforward. First, the class has to be situated in the last column of the csv file. The numeric to nominal transform is then used, and the above models can be evaluated with ease under the "Classify" tab. 

The pipeline is summarised in **Data & Feature Engineering.ipynb**.


## Feature List
Following are the list of features used for the final submission. The code could be found in *feature_list.py*.

| Feature | Description |
|---------|-----------|
| Common Friends        |  Common friends between **u** & **v**.         |
| Total Friends        | Total friends of **u** & **v**.          |
| Preferential Attachment        |           |
| Jaccard's Coefficient       |  Ratio of common friends to total friends.         |
| Adar      |   [Reference](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.449.1743&rep=rep1&type=pdf)  |
| Transitive Friends          |  Does **u** follow followers of **v**?         |
| Opposite Friends        | Does **v** follow **u**?          |
| Dice Coefficient Directed           |  [Reference](https://opus.lib.uts.edu.au/bitstream/10453/43302/1/final.pdf)         |
| Dice Coefficient Undirected          |  [Reference](https://opus.lib.uts.edu.au/bitstream/10453/43302/1/final.pdf)         |
| Friends Closeness          |      |
| Jaccard Outneighbours           |           |
| Jaccard Inneighbours          |           |
| Degree Source          |   In & out degrees of source node **u**.        |
| Degree Source In          |  In degree of source node **u**.          |
| Degree Source Out          | Out degree of source node **u**.           |
| Degree Target        | In & out degrees of target node **v**.           |
| Degree Target In |In degree of target node **v**.  |
| Degree Target Out |Out degree of target node **v**.  |


## Files

* **Data & Feature Engineering.ipynb** — Contains data and feature engineering pipelines.
* **preprocessing.py** — Contains the process of creating a training dataset from the provided Graph.   
* **feature_list.py** — Contains all features used for the final submission of the competition.
* **feature.engineering.py** — This file generates features, given a dataset (i.e. train or test).

## Data
#### Dataset descriptions (excerpt from Kaggle)

- **train.txt** - the training graph adjacency lists (tab delimited; line per source node followed by sink neighbours; no header line) <t style='color:crimson'> Note: Not contained in repo due to size.</t>  
- **test-public.txt** - the test edge IDs (tab delimited; edge ID, source node, sink node; with header line)  
- **sample.csv** - a sample submission file in the correct format  

**Note:** Sink nodes are refered to as target/sink interchangeably.



## Authors

* **William Rudd** - *Team lead* - [github](https://github.com/billrudd)
* **Sadegh Modarres** -  [github](https://github.com/HadiModarres)
* **Najla Alariefy** -  [github](https://github.com/najlaalariefy)
