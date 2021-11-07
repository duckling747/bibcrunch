# BibCrunch
BibCrunch is a project with the goal of finding reliable measures of text similarity. One such found
is to use the, apparently now classic, tf-idf method, combined with cosine similarity for the
resulting vectors. This repository hosts the source code for a Flask website, that demostrates
similarity measures achieved in this way. The similarities have been precalculated for a few
hundred random Project Gutenberg books. The site itself can also calculate this measure
dynamically, with the user sending in two text files. 

## Running it
Running the project should be familiar to anyone with experience in Flask and Python. For example, one
simple way to do it in Debian 11 is: `FLASK_APP=bibcrunch.py flask run`. Just make sure you have installed
the requirements, found in the requirements.txt file.

## Heroku
The project is deployed in Heroku at [https://bibcrunch.herokuapp.com/](https://bibcrunch.herokuapp.com/).
Please note, that sending in too many requests too quickly can easily result in the app crashing, because
the limitations for memory on the Heroku free plan are so strict. 

## Further results
During the project, other text similarity measures were also found. One such measure was achieved by
training a siamese neural network, by using the already gathered cosine similarity data, mentioned
above. You can view the script that was used to achieve this [here](https://github.com/duckling747/bibcrunch_train_script).
