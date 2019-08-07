import numpy as np


def montecarlo_clustering(model, data, executions, num_clusters, max_iterations):
    all_assignments = []
    models = []
    #print("holi")
    for execution in range(executions):
        #print("execution", execution)
        m = model(data)
        m.cluster(num_clusters, max_iterations)
        #print(m.assign)
        models.append(m)
        all_assignments.append(m.assign)
    all_assignments = np.transpose(all_assignments)
    final_assignment = []
    for i_data in range(len(all_assignments)):
        final_assignment.append(-1)
        higher_sum = -1
        for i_cluster in range(num_clusters):
            actual_sum = np.sum(np.isin(all_assignments[i_data],i_cluster))
            if actual_sum > higher_sum:
                final_assignment[i_data] = i_cluster
                higher_sum = actual_sum
    return (final_assignment, all_assignments, models)