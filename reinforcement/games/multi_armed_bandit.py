# -*- coding: utf-8 -*-

import random

class Game:
    def __init__(self, n = 3,\
                 rand_funcs=[random.random, random.random, random.random],\
                 rate_success=[0.1, 0.3, 0.2]):
        if len(rand_funcs) != n or len(rate_success) != n:
            raise ValueError("Lenght are not good")
            self.n = n
            self.rand_funcs = rand_funcs
            self.rate_success = rate_success
    
    def play_binary(self, i):
        """
        i starts from 0 to len(n)-1
        """
        result = self.rand_funcs[i]()
        if result <= self.rate_success[i]:
            return True
        else:
            return False
        