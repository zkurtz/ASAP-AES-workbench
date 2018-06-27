import pdb
import pandas as pd

from asap_essay_scoring import metrics
from asap_essay_scoring.utils import data_path as dp

training_data_file = dp("training_set_rel3.tsv")
test_data_file = training_data_file # predicting on the training set should be a best-case scenario for performance
output_file = dp("length_benchmark.csv")


# Most of the following is a near-verbatim copy of Ben Hamner's benchmark
# https://raw.githubusercontent.com/benhamner/ASAP-AES/master/Benchmarks/length_benchmark.py

import re
from sklearn.ensemble import RandomForestRegressor


def add_essay_training(data, essay_set, essay, score):
    if essay_set not in data:
        data[essay_set] = {"essay": [], "score": []}
    data[essay_set]["essay"].append(essay)
    data[essay_set]["score"].append(score)


def add_essay_test(data, essay_set, essay, essay_id):
    if essay_set not in data:
        data[essay_set] = {"essay": [], "essay_id": []}
    data[essay_set]["essay"].append(essay)
    data[essay_set]["essay_id"].append(essay_id)


def read_training_data(training_file):
    with open(training_file, encoding="latin-1") as f:
        f.readline()
        training_data = {}
        for row in f:
            row = row.strip().split("\t")
            essay_set = row[1]
            essay = row[2]
            domain1_score = int(row[6])
            # Suppress domain2:
            #if essay_set == "2":
            #    essay_set = "2_1"
            add_essay_training(training_data, essay_set, essay, domain1_score)

            if essay_set == "2_1":
                essay_set = "2_2"
                domain2_score = int(row[9])
                add_essay_training(training_data, essay_set, essay, domain2_score)

    return training_data


def read_test_data(test_file):
    with open(test_file, encoding="latin-1") as f:
        f.readline()
        test_data = {}
        for row in f:
            row = row.strip().split("\t")
            essay_id = row[0]
            essay_set = row[1]
            essay = row[2]
            # # Suppress treatment of domain2
            # domain1_predictionid = int(row[3])
            # if essay_set == "2":
            #     domain2_predictionid = int(row[4])
            #     add_essay_test(test_data, "2_1", essay, domain1_predictionid)
            #     add_essay_test(test_data, "2_2", essay, domain2_predictionid)
            # else:
            #     add_essay_test(test_data, essay_set, essay, domain1_predictionid)
            add_essay_test(test_data, essay_set, essay, essay_id)
        return test_data


def get_character_count(essay):
    return len(essay)


def get_word_count(essay):
    return len(re.findall(r"\s", essay)) + 1


def extract_features(essays, feature_functions):
    return [[f(es) for f in feature_functions] for es in essays]


def main():
    print("Reading Training Data")
    training = read_training_data(training_data_file)
    print("Reading Validation Data")
    test = read_test_data(test_data_file)

    feature_functions = [get_character_count, get_word_count]

    essay_sets = sorted(training.keys())
    predictions = {}
    feats = {}

    for es_set in essay_sets:
        print("Making Predictions for Essay Set %s" % es_set)
        features = extract_features(training[es_set]["essay"],
                                    feature_functions)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(features, training[es_set]["score"])
        features = extract_features(test[es_set]["essay"], feature_functions)
        predicted_scores = rf.predict(features)
        for pred_id, pred_score in zip(test[es_set]["essay_id"],
                                       predicted_scores):
            predictions[pred_id] = [es_set, round(pred_score)]

        for pred_id, f in zip(test[es_set]["essay_id"], features):
            feats[pred_id] = f

    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")
    f.write("essay_id,essay_set,pred,nchar,nword\n")
    for key in sorted(predictions.keys()):
        f.write("%d,%d,%d,%d,%d\n" % (int(key), int(predictions[key][0]), predictions[key][1],
                                      feats[key][0], feats[key][1]))
    f.close()


if __name__ == "__main__":
    main()
    df = pd.read_csv(output_file)
    import csv
    dd = pd.read_csv(training_data_file, sep='\t', encoding='latin-1', quoting=csv.QUOTE_NONE)
    zz = dd[['essay_id', 'domain1_score']].merge(df, on='essay_id', how='outer')
    zz.rename(columns = {'domain1_score': 'truth'}, inplace = True)
    metrics.evaluate(zz)
    zz.to_csv(dp("hamner_features.csv"), index = False)
    pdb.set_trace()