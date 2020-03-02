import numpy as np

class Generator():
    def __init__(self, potential, pram_bound, energy=None, size=128):
        self.pot = potential
        self.eng = energy
        self.r_parm = pram_bound
        self.size = size
        self.samps = []
    
    def gen_samps(self,n_samps):
        x,y = np.meshgrid(np.linspace(-20,20,self.size),np.linspace(-20,20,self.size))
        for i in range(n_samps):
            rands = []
            for bnds in self.r_parm:
                rands.append(np.random.uniform(bnds[0],bnds[1]))
            if self.eng != None:
                self.samps.append(np.concatenate((np.reshape(self.pot(x,y,rands),self.size*self.size),np.array([self.eng(rands)]))))
            else:
                self.samps.append(np.concatenate((np.reshape(self.pot(x,y,rands),self.size*self.size),np.array([self.find_energy(rands)]))))
    
    def clear_samps(self):
        self.samps = []
    
    def export_samps(self,filename):
        np.save(filename,np.array(self.samps))
            
    def find_energy(self,pot,steps=5):
        return 0