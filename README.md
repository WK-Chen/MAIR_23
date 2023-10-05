# MAIR Group 23

## Team
Wenkai Chen, Stanescu Raluca, Frederieke Blom, Niek Kemp, Dimitra Tsolka

## Part 1a
Run `python part1a/data_process.py` to turn dialog_acts.dat to csv file and create a deduplication version dialog_acts_dedup.csv

### Baseline 1
See baseline1 result by code `python part1a/baseline1.py`

### Baseline 2
See baseline1 result by code `python part1a/baseline2.py`

### Classifier 1
Train and evaluate classifier by running `python part1a/classifier1.py [DATA_PATH]`. 
e.g. `python part1a/classifier1.py data/dialog_acts.csv`

Users can also input sentences to test the classifier after the evaluation.

### Classifier 2
Train classifier2 by running `python part1a/classifier2.py [ACTION] [DATA_PATH]`.
e.g. `python part1a/classifier2.py train data/dialog_acts.csv`

We can also download trained BERT model for evaluation.
First, we run `mkdir models/` to create the folder. 
Then we download the model [here](https://drive.google.com/file/d/1XBmQHv-fevgoihTdokZQwY_IAfh3ViJ1/view?usp=sharing).
Last unzip it under "model/" folder.

Evaluate classifier2 by running `python part1a/classifier2.py [ACTION] [DATA_PATH] [MODEL_PATH]`.
e.g. `python part1a/classifier2.py evaluate data/dialog_acts.csv models/trained_bert`

Users can also input sentences to test the classifier after the evaluation.

## Part 1b
Diagram_1b.pdf shows the state transition is the dialog system.

Start the dialog system by running `python part1b/dialog_system.py`.

## Part 1c
Diagram_1c.pdf shows the state transition is the dialog system.

Run `python part1c/create_restaurant_info_v2.py` to create a new .csv file with extra properties

Start the dialog system by running `python part1c/dialog_system_v2.py`.

[//]: # (Four features)

[//]: # (1. Use one of the baselines for dialog act recognition instead of the machine learning classifier)

[//]: # (2. Ask user about correctness of match for Levenshtein results)

[//]: # (3. OUTPUT IN ALL CAPS OR NOT &#40;switch on/off&#41;)

[//]: # (4. Introduce a delay before showing system responses &#40;switch on/off&#41;)
