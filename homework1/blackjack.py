from gym.envs.toy_text.blackjack import BlackjackEnv, cmp, usable_ace, sum_hand, is_bust, score, is_natural
import numpy as np

class BlackjackEnvCount(BlackjackEnv):
    
    def __init__(self, natural=False):
        super().__init__(natural)
    
    def step(self, action):
        assert self.action_space.contains(action)
        if action:  
            self.player.append(draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.0
            else:
                done = False
                reward = 0.0
        else:  
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if (
                self.natural
                and is_natural(self.player)
                and reward == 1.0
            ):
                reward = 1.5
        return self._get_obs(), reward, done, {}
    
    def reset(self):
        self.dealer = [draw_card(), draw_card()]
        self.player = draw_hand()
        return self._get_obs()
    
counter_dict = { 1: -2, 2: 1, 3: 2, 4: 2,  5: 3, 6: 2, 7: 1, 8: 0, 9: -1, 10: -2 }

def draw_card():
    global deck
    global counter 
    card = np.random.choice(deck)
    counter += counter_dict[card]
    deck.remove(card)
    if len(deck) < 15:
        deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
        counter = 0
    return card

def draw_hand():
    return [draw_card(), draw_card()]

deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4
counter = 0

