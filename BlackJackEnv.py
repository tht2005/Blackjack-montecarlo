import numpy as np

class Env:
    def initEpisode(self):
        self.cur = 0
        self.values = []
        for i in range(1, 13 + 1):
            self.values.append(i)
            self.values.append(i)
            self.values.append(i)
            self.values.append(i)
        np.random.shuffle(self.values)

    def hit(self):
        ret = self.values[self.cur]
        self.cur += 1
        return ret

