# -*- coding: utf-8 -*-


import random

class BlackJack:
    def __init__(self, bet, seed=None, ratio_win=2, blackjack_multiplier=1, verbose=False):
        self._rnd = random.Random(seed)
        self.cards_player = [self._rnd.randint(1,12), self._rnd.randint(1,12)]
        self.cards_croupier = [self._rnd.randint(1,12), self._rnd.randint(1,12)]
        self.stopped = 0
        self.bet = bet
        self.ratio_win = ratio_win
        self.blackjack_multiplier = blackjack_multiplier
        if not self.test_stopped(verbose) and verbose:
            print(self.cards_player, self.cards_croupier)
            print(self.score())
    
    def score(self):
        return BlackJack.score_of_cards(self.cards_player), BlackJack.score_of_cards(self.cards_croupier)
    
    @staticmethod
    def score_of_cards(cards):
        ones_idxs = []# [1 if x==1 else 0 for x in cards]
        for i in range(len(cards)):
            if cards[i] == 1:
                ones_idxs.append(i)
        points = sum(cards)
        if points > 21:
            return -1
        for one_idx in ones_idxs:
            if points+10 > 21:
                return points
            else:
                points += 10
        if points == 21 and len(cards) == 2 and 1 in cards and 10 in cards:#blackjack
            return 100
        return points
        
    def test_win(self):
        player, croupier = self.score()
        if player == 100:#blackjack
            return self.bet * self.ratio_win * self.blackjack_multiplier
        elif (croupier < 0) or (player > 0 and player > croupier):#win
            return self.bet * self.ratio_win
        else:
            return 0
    
    def test_stopped(self, verbose = False):
        player, croupier = self.score()
        if player < 0 or croupier < 0 or player == 100 or self.stopped == 2:
            if verbose:
                print(self.cards_player, self.cards_croupier)
                print(self.score())
                print("Finish: ", self.test_win())
            return True
        return False
    
    def play_croupier(self, verbose=False):
        player, croupier = self.score()
        if croupier <= 16:
            self.cards_croupier.append(self._rnd.randint(1,12))
        else:
            self.stopped += 1
        if self.test_stopped(verbose):
            return self.test_win()
        elif verbose:
            print(self.cards_player, self.cards_croupier)
            print(self.score())
    
    def play(self, strategy: int, verbose=False):#0 to stop, 1 to take and 2 to double, no separate
        if self.test_stopped(verbose):
            return self.test_win()
        if strategy == 0: #stop
            self.stopped += 1#player stop
        elif strategy == 1 or strategy == 2:#take or double
            self.cards_player.append(self._rnd.randint(1,12))
        elif strategy == 2:
            self.bet = self.bet * 2
        if not self.test_stopped(verbose):
            if verbose:
                print(self.cards_player, self.cards_croupier)
                print(self.score())
            return self.play_croupier(verbose)
        else:
            return self.test_win()