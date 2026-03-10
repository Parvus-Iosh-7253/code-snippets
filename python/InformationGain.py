import pandas as pd
import numpy as np

data = pd.read_csv(r"..\..\Datasets\ml-bugs.csv")

def two_group_ent(first, tot):
    return -(first/tot*np.log2(first/tot) +
             (tot-first)/tot*np.log2((tot-first)/tot))

def InformationGain(dataset, outcome_column, num_factors = 2, num_limit = 0):

    outcomes = dataset[outcome_column].unique()
    factors = dataset.columns[dataset.columns != outcome_column]
    num_rows = len(dataset)

    entropy_outcomes = 0

    for outcome in outcomes:
        prob = len(dataset[dataset[outcome_column] == outcome]) / len(dataset)
        print(outcome, "probability:", prob)
        print(outcome, "entropy:", np.negative(prob * np.log2(prob)))
        entropy_outcomes += np.negative(prob * np.log2(prob))

    print(outcome_column, "entropy:", entropy_outcomes)

    for i in range(num_factors):
        print(factors[i], "information gain from:")
        if dataset[factors[i]].dtype == "str":
            variations = dataset[factors[i]].unique()
            for t in variations:
                print(f"-{t}:")
                info_gain = entropy_outcomes
                split_set = dataset[dataset[factors[i]] == t]
                post_split_entropy = 0
                for s in outcomes:
                    prob = len(split_set[split_set[outcome_column] == s]) / num_rows
                    post_split_entropy += np.negative(prob * np.log2(prob))
                info_gain -= (post_split_entropy) / 2
                print(f"------{info_gain}")
        elif dataset[factors[i]].dtype in ["int", "float"]:
            print(f"---{factors[i]} < {num_limit} mm:")
            info_gain = entropy_outcomes
            hi_split = dataset[dataset[factors[i]] >= num_limit]
            lo_split = dataset[dataset[factors[i]] < num_limit]
            for s in outcomes:
                post_split_entropy = 0
                prob_hi = len(hi_split) / num_rows
                hi_entropy = -prob_hi * np.log2(len(hi_split[hi_split["Species"] == s])/len(hi_split))
                print(len(hi_split), prob_hi, len(hi_split[hi_split["Species"] == s]), hi_entropy)
                prob_lo = len(lo_split) / num_rows
                lo_entropy = -prob_lo * np.log2(len(lo_split[lo_split["Species"] == s])/len(lo_split))
                print(len(lo_split), prob_lo, len(lo_split[lo_split["Species"] == s]), lo_entropy)
                post_split_entropy += hi_entropy + lo_entropy
                print(f"{s}: {post_split_entropy}")
            info_gain -= (post_split_entropy) / 2
            print(f"------{info_gain}")




# InformationGain(dataset=data, outcome_column="Species", num_limit=17)
prob_mo = len(data[data["Species"] == "Mobug"])/len(data)
prob_lo = (len(data) - len(data[data["Species"] == "Mobug"]))/len(data)
ent_og = -(prob_mo * np.log2(prob_mo)) + -(prob_lo * np.log2(prob_lo))

more_17 = data[data["Length (mm)"] >= 17]
prob_more_17 = len(data[data["Length (mm)"] >= 17])/len(data)
prob_m17_mb = len(more_17[more_17["Species"] == "Mobug"])/len(more_17)

less_17 = data[data["Length (mm)"] < 17]
prob_less_17 = len(data[data["Length (mm)"] < 17])/len(data)
prob_l17_mb = len(less_17[less_17["Species"] == "Mobug"])/len(less_17)


tot_ent = two_group_ent(len(data[data["Species"] == "Mobug"]), len(data))
g17_ent = prob_more_17 * two_group_ent(len(more_17[more_17["Species"] == "Mobug"]),len(more_17)) + prob_less_17 * two_group_ent(len(less_17[less_17["Species"] == "Mobug"]),len(less_17))

answer = tot_ent - g17_ent
print(answer)



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

