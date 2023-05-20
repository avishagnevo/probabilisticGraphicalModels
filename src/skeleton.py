from pgmpy.factors import factor_product
from pgmpy.inference import VariableElimination
from pgmpy.readwrite import UAIReader
import numpy as np
import random
import time
import math
import itertools

"""A helper function. You are free to use."""


# def numberOfScopesPerVariable(scopes):
def maximalVariableInScopes(scopes):
    # Initialize a dictionary to store the counts
    counts = {}
    # Iterate over each scope
    for scope in scopes:
        # Iterate over each variable in the scope
        for variable in scope:
            # Increment the count for the variable
            if variable in counts:
                counts[variable] += 1
            else:
                counts[variable] = 1
    # Sort the counts dictionary based on the frequency of variables in the scopes
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    # mine
    maximal_variable = max(sorted_counts, key=lambda x: x[1])[0]
    # mine
    # return sorted_counts
    return maximal_variable


"""
You need to implement this function. It receives as input a junction tree object,
and a threshold value. It should return a set of variables such that in the 
junction tree that results from jt by removing these variables (from all bags), 
the size of the largest bag is threshold.
The heuristic used to remove RVs from the junction tree is: repeatedly remove the RV that appears in
the largest number of bags in the junction tree.
"""


def getCutset(jt, threshold):
    cutset = []
    bags = list(jt.nodes)
    maximal_bag = max(len(bags[i]) for i in range(len(bags)))
    while maximal_bag > threshold:
        # appending the most appearing variable
        cutset.append(maximalVariableInScopes(bags))
        # updating the bags
        bags = [tuple(var for var in bag if var not in cutset) for bag in bags]
        # updating maximal bag size
        maximal_bag = max(len(bags[i]) for i in range(len(bags)))
    return cutset


"""
You are provided with this function. It receives as input a junction tree object, the MarkovNetwork model,
and an evidence dictionary. It computes the partition function with this evidence.
"""


def computePartitionFunctionWithEvidence(jt, model, evidence):
    reducedFactors = []
    for factor in jt.factors:
        evidence_vars = []
        for var in factor.variables:
            if var in evidence:
                evidence_vars.append(var)
        if evidence_vars:
            reduce_vars = [(var, evidence[var]) for var in evidence_vars]
            new_factor = factor.reduce(reduce_vars, inplace=False)
            reducedFactors.append(new_factor)
        else:
            reducedFactors.append(factor.copy())

    totalfactor = factor_product(*[reducedFactors[i] for i in range(len(reducedFactors))])
    var_to_marg = (
            set(model.nodes()) - set(evidence.keys())
    )
    marg_prod = totalfactor.marginalize(var_to_marg, inplace=False)
    return marg_prod.values


def GenerateSample(Q):
    # generates an assignment x to X from the distribution Q
    keys = list(Q.keys())
    probabilities = list(Q.values())
    return random.choices(keys, probabilities)[0]


def generateAssignments(variables, model):
    # model.states returns a dictionary mapping each node to its list of possible states
    states = [model.states[var] for var in variables]
    # iterates over the cartesian product of possible variables states
    # creates a list of dictionaries representing possible assignments
    assignments = [{variable: value for variable, value in zip(variables, assignment)}
                   for assignment in itertools.product(*states)]
    return assignments


def generateQ(model, variables, assignments, kind):
    if kind == 'QRB':
        Q = {}
        inference = VariableElimination(model)
        potentials_table = inference.query(variables=variables)
        Z = model.get_partition_function()
        for assignment in assignments:
            # initializes p to be the whole potentials table
            p = potentials_table.values
            # gets the probability that (X1=x1, ..., Xn=xn) from the potentials table
            for value in assignment:
                p = p[value]
            # maps each assignment to its probability (with normalization)
            Q[assignment] = p / Z
    else:
        # uniform over all possible assignments
        Q = {key: 1 / np.power(2,len(variables)) for key in assignments}
    return Q


def SampleCompute(Q, extended_forms, N, model, jt, Z):
    # step 6:
    for _ in range(N):
        # step 7 + step 8:
        x = GenerateSample(Q)
        # step 9: computes partition function given evidence x
        part_x = computePartitionFunctionWithEvidence(jt, model, extended_forms[x])
        # step 10:
        t_x = part_x / Q[x]
        # step 11:
        Z += t_x
    # step 12:
    return Z / N


def ComputePartitionFunction(uai_file_path, w, N):
    # Step 1: Initialize the partition function estimate Z to 0
    Z = 0

    # Step 2: Read the Markov network from the UAI file
    model = UAIReader(uai_file_path).get_model()

    # Step 3: Construct the junction tree
    jt = model.to_junction_tree()

    # Step 4: Heuristically remove variables from the junction tree until the largest cluster has at most w variables
    X = getCutset(jt, w)

    # creates a dictionary : {[x1_1,..,xn_1] : {X1 = x1_1,..,Xn = xn_1},..,[x1_m,..,xn_m] : {X1 = x1_m,..,Xn = xn_m}}
    assignments_dict = generateAssignments(X, model)
    assignment_simplified_form = [assignment.values() for assignment in assignments_dict]
    num_assigments = len(assignment_simplified_form)
    assignment_forms = {assignment_simplified_form[i]: assignments_dict[i] for i in
                        range(num_assigments)}

    return Z, model, assignment_forms, X, jt


"""This function implements the experiments where the sampling distribution is Q^{RB}"""


def ExperimentsDistributionQRB(uai_file_path, w, N):
    Z, model, assignment_forms, X, jt = ComputePartitionFunction(uai_file_path, w, N)
    assignments_simplify = assignment_forms.keys()
    Q = generateQ(model, X, assignments_simplify, kind='QRB')
    return SampleCompute(Q, assignment_forms, N, model, jt, Z)


"""This function implements the experiments where the sampling distribution Q is uniform"""


def ExperimentsDistributionQUniform(uai_file_path, w, N):
    Z, model, assignment_forms, X, jt = ComputePartitionFunction(uai_file_path, w, N)
    assignments_simplify = assignment_forms.keys()
    Q = generateQ(model, X, assignments_simplify, kind='uniform')
    return SampleCompute(Q, assignment_forms, N, model, jt, Z)


def experiments(func, path, w, N, seeds):
    # prints results by w,N for 10 different seeds
    print(f'w={w} and N={N}')
    errors = []
    times = []
    model = UAIReader(path).get_model()
    Z = model.get_partition_function()

    for seed in seeds:
        random.seed(seed)
        start = time.time()
        Z_approximated = func(path, w, N)
        end = time.time()
        times.append(end - start)
        errors.append(math.fabs(math.log(Z) - math.log(Z_approximated)) / math.log(Z))

    mu_time = round(float(np.mean(times)), 4)
    sigma_time = round(float(np.std(times)), 4)
    print(f'{mu_time} +- {sigma_time}')
    #print(f'time is {mu_time} +- {sigma_time}')

    mu_error = round(float(np.mean(errors)), 4)
    sigma_error = round(float(np.std(errors)), 4)
    print(f'{mu_error} +- {sigma_error}')
    #print(f'error is {mu_error} +- {sigma_error}')


    return


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print("grid4x4 Experiments:")
    path = 'grid4x4.uai'
    Ns = [50, 100, 1000, 5000]
    ws = [1, 2, 3, 4, 5]
    seeds = list(range(1, 100, 10))
    for w, N in itertools.product(ws, Ns):
        print("Uniform Experiments")
        experiments(ExperimentsDistributionQUniform, path, w, N, seeds)
        print("QRB Experiments")
        experiments(ExperimentsDistributionQRB, path, w, N, seeds)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
