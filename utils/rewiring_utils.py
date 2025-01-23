import torch

def get_common_rewiring_candidate(a, c, adj_matrix_all):
    adj_cs = adj_matrix_all[c]
    adj_cs = adj_cs.masked_fill(adj_cs == 1, 2)
    adj_a = adj_matrix_all[a]
    adj_as = adj_a.repeat(adj_cs.size(0), 1)
    diff_matrix = adj_cs - adj_a # if there are more 1s due to (2-1), that means there are more common nghs
    num_ones = (diff_matrix == 1).sum(dim=1)
    max_index = torch.argmax(num_ones)
    return c[max_index]
 