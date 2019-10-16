# -*- coding: utf-8 -*-


class Tictac:
    def __init__(self):
        self.state = [[0,0,0],[0,0,0],[0,0,0]]
        self.last_played = 0
        self.winned = 0
    
    def score(self):
        #row
        player1 = 0
        player2 = 0
        ns1 = []
        ns2 = []
        #rows
        for row in self.state:
            n_1 = 0
            n_2 = 0
            for d in row:
                n_1 = n_1+1 if d==1 else n_1
                n_2 = n_2+1 if d==2 else n_2
            ns1.append(n_1)
            ns2.append(n_2)
            if n_1 == 3:
                player1 += 1
            if n_2== 3:
                player2 += 1
        #columns
        for icol in range(len(self.state[0])):
            n_1 = 0
            n_2 = 0
            for row in self.state:
                d = row[icol]
                n_1 = n_1+1 if d==1 else n_1
                n_2 = n_2+1 if d==2 else n_2
            ns1.append(n_1)
            ns2.append(n_2)
            if n_1 == 3:
                player1 += 1
            if n_2 == 3:
                player2 += 1
        #diagonals
        diags_indexes = ([[0,0],[1,1],[2,2]], [ [0,2],[1,1],[2,0] ] )
        for diag_index in diags_indexes:
            n_1 = 0
            n_2 = 0
            for pair_index in diag_index:
                d = self.state[pair_index[0]][pair_index[1]]
                n_1 = n_1+1 if d==1 else n_1
                n_2 = n_2+1 if d==2 else n_2
            ns1.append(n_1)
            ns2.append(n_2)
            if n_1 == 3:
                player1 += 1
            if n_2 == 3:
                player2 += 1
        return player1, player2, max(ns1), max(ns2)
    
    def test_win(self):
        p1, p2, _, _ = self.score()
        if p1 > 0:
            return 1
        if p2 > 0:
            return 2
        return 0
            
    
    def play1(self, pos_x:int, pos_y: int, verbose:bool =False):
        w = self.test_win()
        if w != 0:
            if verbose: print("Ended")
            return w
        if self.state[pos_x][pos_y] == 0:
            if self.last_played in [0,2]:
                self.state[pos_x][pos_y] = 1
                self.last_played = 1
            elif self.last_played == 1:
                if verbose: print("Error now plays 2")
                return -2
        else:
            if verbose: print("Error already used position")
            return -1
        if verbose:
            for row in self.state:
                print(row)
        return self.test_win()
        
    def play2(self, pos_x:int, pos_y: int, verbose:bool =False):
        w = self.test_win()
        if w != 0:
            if verbose: print("Ended")
            return w
        if self.state[pos_x][pos_y] == 0:
            if self.last_played == 1:
                self.state[pos_x][pos_y] = 2
                self.last_played = 2
            elif self.last_played == 2:
                if verbose: print("Error now plays 1")
                return -2
        else:
            if verbose: print("Error already used position")
            return -1
        if verbose:
            for row in self.state:
                print(row)
        return self.test_win()
    
    def auto_play(self, pos_x:int, pos_y: int, verbose:bool =False):
        w = self.test_win()
        if w != 0:
            print("Ended")
            return w
        if self.state[pos_x][pos_y] == 0:
            if self.last_played in [0,2]:
                self.play1(pos_x, pos_y)
            elif self.last_played == 1:
                self.play2(pos_x, pos_y)
        else:
            print("Error already used position")
            return -1
        if verbose:
            for row in self.state:
                print(row)
        return self.test_win()