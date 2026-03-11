import pandas as pd
import numpy as np

data = pd.read_csv(r"..\Datasets\ml-bugs.csv")

# multiclass data entropy =
#   probability of group 1 * (
#       -((probability of outcome 1 in group 1) * log_2(probability of outcome 1 in group 1) + (
#       (probability of outcome 2 in group 1) * log_2(probability of outcome 2 in group 1))) +
#   probability of group 2 * (
#       -((probability of outcome 1 in group 2) * log_2(probability of outcome 1 in group 2) + (
#       (probability of outcome 2 in group 2) * log_2(probability of outcome 2 in group 2))) +
#   ...
#   probability of group n * (
#       -((probability of outcome 1 in group n) * log_2(probability of outcome 1 in group n) + (
#       (probability of outcome 2 in group n) * log_2(probability of outcome 2 in group n)))
#
# information gain = parent entropy - average(child entropies)

# TODO: store information gain values for each factor, return highest value
def InformationGain(dataset, outcome_column, num_factors = 2, num_limit = [0]):

    outcomes = dataset[outcome_column].unique()
    factors = dataset.columns[dataset.columns != outcome_column]
    num_rows = len(dataset)

    entropy_outcomes = 0

    for outcome in outcomes:
        prob = len(dataset[dataset[outcome_column] == outcome]) / len(dataset)
        entropy_outcomes += np.negative(prob * np.log2(prob))

    for i in range(num_factors):
        print(f"Information gain from {factors[i]}:")
        if dataset[factors[i]].dtype == "str":
            variations = dataset[factors[i]].unique()
            for t in variations:

                print(f"---{t}:")
                info_gain = entropy_outcomes

                split_set_1 = dataset[dataset[factors[i]] == t]
                set_1_prob = len(split_set_1) / len(dataset)

                split_set_2 = dataset[dataset[factors[i]] != t]
                set_2_prob = len(split_set_2) / len(dataset)

                post_split_entropy = 0

                for s in outcomes:
                    prob_1 = len(split_set_1[split_set_1[outcome_column] == s]) / len(split_set_1)
                    prob_1_0 = len(split_set_1[split_set_1[outcome_column] != s]) / len(split_set_1)

                    prob_2 = len(split_set_2[split_set_2[outcome_column] == s]) / len(split_set_2)
                    prob_2_0 = len(split_set_2[split_set_2[outcome_column] != s]) / len(split_set_2)

                    ent_1 = set_1_prob * (-((prob_1 * np.log2(prob_1)) + (prob_1_0 * np.log2(prob_1_0))))
                    ent_2 = set_2_prob * (-((prob_2 * np.log2(prob_2)) + (prob_2_0 * np.log2(prob_2_0))))

                    post_split_entropy += (ent_1 + ent_2) / 2

                info_gain -= (post_split_entropy)
                print(f"------{info_gain}")

        elif dataset[factors[i]].dtype in ["int", "float"]:

            for n in num_limit:

                print(f"---{factors[i]} < {n}:")
                info_gain = entropy_outcomes

                split_set_1 = dataset[dataset[factors[i]] >= n]
                set_1_prob = len(split_set_1) / len(dataset)

                split_set_2 = dataset[dataset[factors[i]] < n]
                set_2_prob = len(split_set_2) / len(dataset)

                post_split_entropy = 0

                for s in outcomes:
                    prob_1 = len(split_set_1[split_set_1[outcome_column] == s]) / len(split_set_1)
                    prob_1_0 = len(split_set_1[split_set_1[outcome_column] != s]) / len(split_set_1)

                    prob_2 = len(split_set_2[split_set_2[outcome_column] == s]) / len(split_set_2)
                    prob_2_0 = len(split_set_2[split_set_2[outcome_column] != s]) / len(split_set_2)

                    ent_1 = set_1_prob * (-((prob_1 * np.log2(prob_1)) + (prob_1_0 * np.log2(prob_1_0))))
                    ent_2 = set_2_prob * (-((prob_2 * np.log2(prob_2)) + (prob_2_0 * np.log2(prob_2_0))))

                    post_split_entropy += (ent_1 + ent_2) / 2

                info_gain -= (post_split_entropy)
                print(f"------{info_gain}")

InformationGain(dataset=data, outcome_column="Species", num_limit=[17, 20])

