# ngram

1. Run `train_ngram.py´ to build the n_gram linguistic model. Current setting is to build 4_gram to 8_gram. An input file and class file is needed.
2. Run `inference.py input_file output_file` to do the inference.
3. If a ground truth is provided, run `compare.py predict_file ground_truth_file` to generate the confusion matrix and accuracy.
