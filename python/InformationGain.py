import pandas as pd
import numpy as np

data = pd.read_csv("..\\Datasets\\ml-bugs.csv")

def MCEntropy(dataset, outcome_column, num_factors = 2, num_limit = 0):

    outcomes = dataset[outcome_column].unique()
    factors = dataset.columns[dataset.columns != outcome_column]

    entropy_outcomes = 0

    for outcome in outcomes:
        prob = len(dataset[dataset[outcome_column] == outcome]) / len(dataset)
        entropy_outcomes += np.negative(prob * np.log2(prob))

    print(outcome_column, "entropy:", entropy_outcomes)

    entropies = {}

    for i in range(num_factors):
        factor_entropy = 0
        print(factors[i], " entropy: ", sep="")
        if dataset[factors[i]].dtype == object:
            variations = dataset[factors[i]].unique()
            for t in variations:
                prob = len(dataset[dataset[factors[i]] == t]) / len(dataset)
                factor_entropy += np.negative(prob * np.log2(prob))
        elif dataset[factors[i]].dtype in [int, float]:
            prob_1 = len(dataset[dataset[factors[i]] >= num_limit]) / len(dataset)
            prob_2 = len(dataset[dataset[factors[i]] < num_limit]) / len(dataset)
            factor_entropy += np.negative(prob_1 * np.log2(prob_1)) - (prob_2 * np.log2(prob_2))
        print(factor_entropy)
        entropies[factors[i]] = factor_entropy

    for i in range(num_factors):
        factor_entropy = 0
        print(factors[i], "information gain from:")
        if dataset[factors[i]].dtype == object:
            variations = dataset[factors[i]].unique()
            for t in variations:
                print(f"---{t}:")
                info_gain = entropy_outcomes
                total_results = len(dataset[dataset[factors[i]] == t])
                post_split_entropy = 0
                for s in outcomes:
                    prob = len(dataset[dataset[outcome_column] == s]) / total_results
                    post_split_entropy += np.negative(prob * np.log2(prob))
                info_gain -= post_split_entropy
                print(f"------{info_gain}")



MCEntropy(dataset=data, outcome_column="Species", num_limit=17)

prob_lo = len(data[data["Species"] == "Lobug"]) / len(data)
prob_mo = len(data[data["Species"] == "Mobug"]) / len(data)
entropy_spec = np.negative(prob_lo * np.log2(prob_lo)) - (prob_mo * np.log2(prob_mo))

print("Species entropy:",entropy_spec)

print(len(data[(data["Color"] == "Brown") & (data["Species"] == "Lobug")]))

'''
data_species = []
species = data["Species"].unique()
for name in species:
    data_species.append(data[data["Species"] == name])

data_colors = []
colors = data["Color"].unique()
for color in colors:
    data_colors.append(data[data["Color"] == color])

for i in data_species:
    total = len(i)

    data_colors = []
    colors = i["Color"].unique()
    for color in colors:
        data_colors.append(i[i["Color"] == color])

    print("Colors\n", data_colors, "\n")

    low_len = []
    high_len = []
    avg_length = sum(i["Length (mm)"]) / len(i)
    print("Average Length:", avg_length, "\n")
    comp_len = 17
    for row in i["Length (mm)"]:
        if row >= comp_len:
            high_len.append(i[i["Length (mm)"] == row])
        else:
            low_len.append(i[i["Length (mm)"] == row])

    print("Low Lengths\n", low_len, "\nHigh Lengths\n", high_len, "\n")


species = data["Species"].unique()
color = data["Color"].unique()

total = len(data)
prob_species = {}
prob_color = {}

for name in species:
    prob_species[name] = (len(data[data["Species"] == name]) / total)

for name in color:
    prob_color[name] = (len(data[data["Color"] == name]) / total)

print(prob_species)
print(prob_color)

entropy_species = 0
entropy_color = 0

for species in prob_species:
    entropy_species += np.negative(prob_species[species]) * np.log2(prob_species[species])

for color in prob_color:
    entropy_color += np.negative(prob_color[color]) * np.log2(prob_color[color])

print(entropy_species)
print(entropy_color)

print(data[data["Species"] == "Mobug"])
print(data[data["Species"] == "Lobug"])
'''

