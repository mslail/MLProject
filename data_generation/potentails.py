import numpy as np

#harmonic ocsillator potential function and ground state energy function
def harmonic_pot(x,y,rand):
    return 0.5*(rand[0]*((x-rand[2])**2)+rand[1]*((y-rand[3])**2))

def harmonic_nrg(rand):
    return 0.5*(np.sqrt(rand[0])+np.sqrt(rand[1]))

#infinite square well potential function and ground state energy function
def sqaure_pot():
    return 0

def square_nrg():
    return 0

#gradient well potential function
def grad_pot():
    return 0

#double barriar well potential function
def double_pot():
    return 0

if __name__ == "__main__":
    import time
    from Generator import Generator
    
    #note: don't generate too many samples in 1 go, you will eat upi your ram
    rand_pram = [(0.0,0.16),(0.0,0.16),(-8.0,8.0),(-8.0,8.0)]
    test_gen = Generator(harmonic_pot,rand_pram,energy=harmonic_nrg)
    
    st = time.time()
    for i in range(200):
        test_gen.gen_samps(1000)
        test_gen.export_samps("text_out"+str(i+1))
        test_gen.clear_samps()
    print("%s"%(time.time()-st))
    
    #run to test potentials with plots
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    
    rand_pram = [(0.0,0.16),(0.0,0.16),(-8.0,8.0),(-8.0,8.0)]
    test_gen = Generator(harmonic_pot,rand_pram,energy=harmonic_nrg)
    test_gen.gen_samps(10)
    
    i = 0
    x,y = np.meshgrid(np.linspace(-20,20,128),np.linspace(-20,20,128))
    for s in test_gen.samps:
        fig, ax = plt.subplots()
        c = ax.contourf(x,y,s[0],levels=50,cmap=cm.YlGnBu)
        fig.colorbar(c)
        plt.savefig("plot"+str(i)+".png")
        plt.close()
        i += 1
    '''