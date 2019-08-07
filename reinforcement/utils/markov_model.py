

def get_matrix_gamma(n_states, p):
    m = []
    for i_row in range(n_states):
        m.append([])
        for i_col in range(n_states):
            if i_row == i_col:
                m[i_row].append(1-p)
            else:
                m[i_row].append(p/float(n_states-1))
    return m

def next_state(m, actual_state, rand):
    row = m[actual_state]
    percentage_sum = 0
    for i_col in range(len(row)):
        percentage_sum += row[i_col]
        if percentage_sum >= rand:
            return i_col