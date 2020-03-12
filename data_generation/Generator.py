import numpy as np
#np.set_printoptions(threshold=np.inf)

class Generator():
    def __init__(self, potential, pram_bound, energy=None, size=128):
        self.pot = potential
        self.eng = energy
        self.r_parm = pram_bound
        self.size = size
        self.samps = []
    
    def gen_samps(self,n_samps):
        x,y = np.meshgrid(np.linspace(-20,20,self.size),np.linspace(-20,20,self.size))
        i = 1
        while i <= n_samps:
            rands = []
            s = -1*np.ones(pow(self.size,2))
            for bnds in self.r_parm:
                rands.append(np.random.uniform(bnds[0],bnds[1]))
            print(rands)
            if self.eng != None:
                s = np.concatenate((np.reshape(self.pot(x,y,rands),pow(self.size,2)),np.array([self.eng(rands)])))
            else:
                s = np.concatenate((np.reshape(self.pot(x,y,rands),pow(self.size,2)),np.array([self.find_energy(rands)])))
            if not (-1 in s):
                self.samps.append(s)
                i += 1
    
    def clear_samps(self):
        self.samps = []
    
    def export_samps(self,filename):
        np.save(filename,np.array(self.samps))
            
    def find_energy(self,rands,steps=5):
        return 0
