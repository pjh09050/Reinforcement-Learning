import numpy as np
import random
import copy

class LR_world():
    def __init__(self):
        self.x = []

    def step(self, a):
        if a == 0:
            if self.x == [0, 1, 0, 1, 0]:
                reward = +1000
            else:
                reward = -1
            self.move_left()
            
        else:
            if self.x == [1, 1, 1, 1, 1]:
                reward = -1000
            else:
                reward = +1
            self.move_right()

        done = self.is_done()
        return self.x, reward, done

    def move_left(self):
        self.x.append(0)

    def move_right(self):
        self.x.append(1)

    def is_done(self):
        if len(self.x) == 6:
            return True
        else: 
            return False
        
    def reset(self):
        self.x = []
        return self.x

class QAgent():
    def __init__(self):
        self.q_table = np.zeros((127,2))       
        self.eps = 0.9
        self.alpha = 0.01

    def state(self, s):
        state = 0
        if len(s) == 0:
            state = 0
        else:
            state += int("".join([str(bit) for bit in s]), 2)
        return state

    def select_action(self, s):
        x = self.state(s)
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0,1)
        else:
            action_val = self.q_table[x,:]
            action = np.argmax(action_val)
        return action

    def update_table(self, transition):
        s, a, r, s_prime = transition
        x = self.state(s)
        next_x = self.state(s_prime)
        a_prime = self.select_action(s_prime)
        self.q_table[x, a] = self.q_table[x, a] + self.alpha * (r + np.amax(self.q_table[next_x, :]) - self.q_table[x, a])
        return r

    def anneal_eps(self):
        self.eps -= 0.01
        self.eps = max(self.eps, 0.1)

    def show_table(self):
        q_lst = self.q_table.tolist()
        print(q_lst)

def main():
    env = LR_world()
    agent = QAgent()
    best_score = -float('inf')
    best_table = []
    best_epi = []

    for n_epi in range(10000):
        done = False
        score = 0.0
        
        s = env.reset()
        while not done:
            s = s[:]
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((copy.deepcopy(s), a, r, copy.deepcopy(s_prime)) )
            s = s_prime
            score += r
        agent.anneal_eps()

        if score == 999.0:
            best_epi.append(n_epi)

        if n_epi%10==0 or n_epi<10:
            print("n_episode : {}, score : {:.1f}".format(n_epi, score))
            agent.show_table()

        if score > best_score:
            best_score = score
            best_table = agent.q_table.tolist()

    print("\nBest table score : {:.1f}, best_episode 갯수: {}".format(best_score, len(best_epi)))
    print('Best table :', best_table)


if __name__ == "__main__":
    main()