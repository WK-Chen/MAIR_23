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
Download trained BERT model by running the script ``
Run `python part1a/classifier2.py [ACTION] [DATA_PATH] [MODEL_PATH]` to see the results for classifier2. Users can also input 
sentences to test the classifier. For example, we use `python part1a/classifier2.py train data/dialog_acts.csv bert-base-uncased`
to train the model. We use `python part1a/classifier2.py evaluate data/dialog_acts.csv models/bert` 
to evaluate the trained model

## Part 1b

## Part 1c
See the diagram for 1c in `FinalDiagram1c.pdf`

Run `create_restaurant_info_v2.py` to create a new .csv file with extra properties

Run `dialog_system_v2.py` for the dialog system with switches and reasoning part

Four features
1. Use one of the baselines for dialog act recognition instead of the machine learning classifier
2. Ask user about correctness of match for Levenshtein results
3. OUTPUT IN ALL CAPS OR NOT (switch on/off)
4. Introduce a delay before showing system responses (switch on/off)
