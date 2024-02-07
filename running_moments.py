import numpy as np

class running_moments: 
    def __init__(self):
        self.iter = 0
        self.mean = None
        self.M = None

    def update(self, x):
        self.iter += 1
        if self.iter == 1:
            self.mean = np.copy(x)
            self.M = np.copy(x)
        else:
            mean_new = self.mean + (x-self.mean)/self.iter
            self.M += (x-self.mean)*(x-mean_new)
            self.mean = mean_new
    
    def get_mean(self):
        return self.mean
    
    def get_var(self):
        if self.iter > 1:
            return self.M/(self.iter-1)
        else:
            return 0
        
    def get_std(self):
        return np.sqrt(self.get_var())