### 업그레이드 그리드 월드 ### 
# xy 0  1  2  3  4  5  6 #
# 0       #             #
# 1       #             #
# 2 S     #     #       #
# 3             #       #
# 4             #     G #
#############################
# 1. S에서 출발해서 G에 도착하면 끝
# 2. 회색 영역(벽)은 지나갈 수 없는 벽이 놓여 있는 곳
# 3. 보상은 스텝마다 -1 ( 즉 최단 거리로 G에 도달하는 것이 목적 ) 
import random
import numpy as np

class GridWorld():
    def __init__(self):
        self.x = 0
        self.y = 0

    def step(self, a):
        if a == 0:
            self.move_left()
        elif a == 1:
            self.move_up()
        elif a == 2:
            self.move_right()
        else:
            self.move_down()
        reward = -1 
        done = self.is_done()
        return (self.x, self.y), reward, done
    
    def move_left(self):
        if self.y == 0:
            pass
        elif self.y == 3 and self.x in [0, 1, 2]:
            pass
        elif self.y == 5 and self.x in [2, 3, 4]:
            pass
        else:
            self.y -= 1
    
    def move_right(self):
        if self.y == 1 and self.x in [0, 1, 2]:
            pass
        elif self.y == 3 and self.x in [2, 3, 4]:
            pass
        elif self.y == 6:
            pass
        else:
            self.y += 1
    
    def move_up(self):
        if self.x == 0:
            pass
        elif self.x == 3 and self.y == 2:
            pass
        else:
            self.x -= 1

    def move_down(self):
        if self.x == 4:
            pass
        elif self.x == 1 and self.y == 4:
            pass
        else:
            self.x += 1
    
    def is_done(self):
        if self.x == 4 and self.y == 6:
            return True
        else:
            return False
    
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class DoubleQAgent():
    def __init__(self):
        self.q1_table = np.zeros((5, 7, 4))
        self.q2_table = np.zeros((5, 7, 4))
        self.eps = 0.9
        self.alpha = 0.1

    def select_action(self, s):
        x, y = s
        coin = random.random()
        if coin < self.eps:
            action = random.randint(0, 3)
        else:
            q1_val = self.q1_table[x, y, :]
            q2_val = self.q2_table[x, y, :]
            action_val = q1_val + q2_val
            action = np.argmax(action_val)
        return action
    
    def update_table(self, transition):
        s, a, r, s_prime = transition
        x, y = s
        next_x, next_y = s_prime
        coin = random.random()
        if coin < 0.5:
            a_prime = np.argmax(self.q1_table[next_x, next_y, :])
            self.q1_table[x, y, a] = self.q1_table[x, y, a] + self.alpha * (r + self.q2_table[next_x, next_y, a_prime] - self.q1_table[x, y, a])
        else:
            a_prime = np.argmax(self.q2_table[next_x, next_y, :])
            self.q2_table[x, y, a] = self.q2_table[x, y, a] + self.alpha * (r + self.q1_table[next_x, next_y, a_prime] - self.q2_table[x, y, a])
    
    def anneal_eps(self):
        self.eps -= 0.01 
        self.eps = max(self.eps, 0.2)

    def show_table(self):
        data = np.zeros((5, 7))
        for i in range(5):
            for j in range(7):
                action_val = (self.q1_table[i, j, :] + self.q2_table[i, j, :]) / 2
                data[i, j] = np.argmax(action_val)
        print(data)

def main():
    env = GridWorld()
    agent = DoubleQAgent()

    for n_epi in range(1000):
        done = False

        s = env.reset()
        while not done:
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s,a,r,s_prime)) #  한 스텝이 끝날 때마다 Q 테이블 업데이트
            s = s_prime
        agent.anneal_eps()

    agent.show_table()

if __name__ == '__main__':
    main()