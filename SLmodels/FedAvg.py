import copy

def FedAvg(w, num_samples):
    total_samples = sum(num_samples)
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        if 'num_batches_tracked' in k:
            continue
        w_avg[k] *= num_samples[0] / total_samples
        for i in range(1, len(w)):
            w_avg[k] += (w[i][k] * num_samples[i] / total_samples)
    return w_avg
