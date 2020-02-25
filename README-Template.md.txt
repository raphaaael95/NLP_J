# skip-gram with negative-sampling implementation from scratch

The goal of this exercise is to implement skip-gram with negative-sampling from scratch.
To do so, the python 3 code, called "skipGram.py", provided contains several functions which allow one to compute the cosine similarity between 2 words.
The algorithm is trained on a huge dataset containing English sentences, one per line.

## Getting Started

These instructions will get you running on your local machine for development and testing purposes.

### Prerequisites

In order to run the python 3 code you need to create a directory which contains:
* The python 3 code called: "skipGram.py"
* The file called: "test_script.sh"

Then, you need to install the fixed dataset to train your model. This dataset is a text file which contains English sentences, one per line.
To install this dataset, you can run the following command line:

```
bash test_script.sh
```

## Running the tests

To run the tests you can use your bash. Once you are in the directory where the python code "skiaGram.py", "test_script.sh" and the dataset are, you can write the following command line:

```
bash test_script.sh
```

This command line will execute your code.


## The code

Let's describe the solution we implemented. 
First, we give an overview of the final code structure chosen, then, we describe our thought process. Finally, we will explain the design choices we made.

### Structure of the code

The code "skipGram.py" is built using preliminary functions and a class:
* The function: text2sentences(path)
* The function: loadPairs(path)
* The class: SkipGram

The class SkipGram uses different methods:
* The constructor method "__init__ ": this method initialize our SkipGram class. 
* The "sample" method: this method returns the id of the negative words used for the negative sampling algorithm
* The "sigmoid" method: this method takes a vector as input and outputs a probability
* The "train" method
* The "trainWord" method
* The "save" method: it saves our model
* The "similarity" method: it computes the similarity between 2 words


### Thought process



### Design choices




## Authors

* **Ariel Modaï** 
* **Raphaël Attali**
* **Niels Nicolas**
* **Michaël Allouche**


## Acknowledgments
