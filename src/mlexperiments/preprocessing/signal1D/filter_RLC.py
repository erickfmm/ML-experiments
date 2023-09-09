import math


def LC_lowpass(input: list, sample_rate: int, frequency: float, Q: float):
    O : float = float(2.0 * math.pi * frequency) / float(sample_rate)
    C : float = Q / O
    L : float = 1.0 / Q / O
    V : float = 0.0
    I : float = 0.0
    T : float = 0.0
    output : list = list()
    for element in input:
        T = float(I-V) / float(C)
        I += float(element * O - V) / L
        V += T
        output.append(float(V)/float(O))
    return output


def LC_highpass(input: list, sample_rate: int, frequency: float, Q: float):
    O : float = float(2.0 * math.pi * frequency) / float(sample_rate)
    C : float = Q / O
    L : float = 1.0 / Q / O
    V : float = 0.0
    I : float = 0.0
    T : float = 0.0
    output : list = list()
    for element in input:
        T = float(element * O) - V
        V += float(I + T) / C
        I += T / L
        output.append(- float(V)/float(O))
    return output
