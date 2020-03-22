import numpy as np

#harmonic ocsillator potential function and ground state energy function
'''
rands
0 - kx
1 - ky
2 - cx
3 - cy
'''
def harmonic_pot(x,y,rand):
    return 0.5*(rand[0]*pow(x-rand[2],2)+rand[1]*pow(y-rand[3],2))

def harmonic_nrg(rand):
    return 0.5*(np.sqrt(rand[0])+np.sqrt(rand[1]))

#infinite square well potential function and ground state energy function
'''
rands
0 - cx
1 - cy
2 - Lx
3 - E
4 - swap Lx and Ly?
'''
def square_pot(x,y,rand):
    if (2*rand[3]/pow(np.pi,2))-(1./pow(rand[2],2)) < 0:
        return -1*np.ones(x.shape)
    Ly = 1/np.sqrt((2*rand[3]/pow(np.pi,2))-(1./pow(rand[2],2)))
    Lx = rand[2] if rand[4]>0.5 else Ly
    Ly = Ly if rand[4]>0.5 else rand[2]
    return np.array([0 if c else 40 for c in np.logical_and(np.logical_and(np.logical_and(0.5*(2*rand[0]-Lx)<x, \
                                                            0.5*(2*rand[0]+Lx)>=x), \
                                                            0.5*(2*rand[1]-Ly)<y), \
                                                            0.5*(2*rand[1]+Ly)>=y).reshape(-1)]).reshape(x.shape)

def square_nrg(rand):
    if (2*rand[3]/pow(np.pi,2))-(1./pow(rand[2],2)) < 0:
        return 0
    Ly = 1/np.sqrt((2*rand[3]/pow(np.pi,2))-(1./pow(rand[2],2)))
    Lx = rand[2] if rand[4]>0.5 else Ly
    Ly = Ly if rand[4]>0.5 else rand[2]
    return 0.5*pow(np.pi,2)*(pow(Lx,-2)+pow(Ly,-2))

#gradient well potential function
'''
rands
0 - cx
1 - cy
2 - Lx
3 - E
4 - swap Lx and Ly?
5 - slope x
6 - slope y
'''
def grad_pot(x,y,rand):
    if (2*rand[3]/pow(np.pi,2))-(1./pow(rand[2],2)) < 0:
        return -1*np.ones(x.shape)
    Ly = 1/np.sqrt((2*rand[3]/pow(np.pi,2))-(1./pow(rand[2],2)))
    Lx = rand[2] if rand[4]>0.5 else Ly
    Ly = Ly if rand[4]>0.5 else rand[2]
    cond = np.logical_and(np.logical_and(np.logical_and(0.5*(2*rand[0]-Lx)<x, \
                                                            0.5*(2*rand[0]+Lx)>=x), \
                                                            0.5*(2*rand[1]-Ly)<y), \
                                                            0.5*(2*rand[1]+Ly)>=y).reshape(-1)
    offx = -0.5*rand[5]*(2*rand[0]-Lx) if rand[5]>0 else -0.5*rand[5]*(2*rand[0]+Lx)
    offy = -0.5*rand[6]*(2*rand[1]-Ly) if rand[6]>0 else -0.5*rand[6]*(2*rand[1]+Ly)
    return np.array([rand[5]*x[i//x.shape[0]][i%x.shape[1]]+offx+rand[6]*y[i//x.shape[0]][i%x.shape[1]]+offy if cond[i] \
                     else 40 for i in range(len(cond))]).reshape(x.shape)

#double barriar well potential function
    '''
rands
0 - cx
1 - cy
2 - Lx
3 - E
4 - swap Lx and Ly?
5 - width%
6 - barriar height
'''
def double_pot(x,y,rand):
    if (2*rand[3]/pow(np.pi,2))-(1./pow(rand[2],2)) < 0:
        return -1*np.ones(x.shape)
    Ly = 1/np.sqrt((2*rand[3]/pow(np.pi,2))-(1./pow(rand[2],2)))
    Lx = rand[2] if rand[4]>0.5 else Ly
    Ly = Ly if rand[4]>0.5 else rand[2]
    cond1 = np.logical_and(np.logical_and(np.logical_and(0.5*(2*rand[0]-Lx)<x, \
                                                            0.5*(2*rand[0]+Lx)>=x), \
                                                            0.5*(2*rand[1]-Ly)<y), \
                                                            0.5*(2*rand[1]+Ly)>=y).reshape(-1)
    cond2 = np.zeros(len(x))
    if rand[4]>0.5:
        cond2 = np.logical_and((0.5*(2*rand[1]-Ly)+Ly/3-rand[5]*Ly/2)<=y,(0.5*(2*rand[1]-Ly)+Ly/3+rand[5]*Ly/2)>=y)+ \
                np.logical_and((0.5*(2*rand[1]-Ly)+2*Ly/3-rand[5]*Ly/2)<=y,(0.5*(2*rand[1]-Ly)+2*Ly/3+rand[5]*Ly/2)>=y)
    else:
        cond2 = np.logical_and((0.5*(2*rand[0]-Lx)+Lx/3-rand[5]*Lx/2)<=x,(0.5*(2*rand[0]-Lx)+Lx/3+rand[5]*Lx/2)>=x)+ \
                np.logical_and((0.5*(2*rand[0]-Lx)+2*Lx/3-rand[5]*Lx/2)<=x,(0.5*(2*rand[0]-Lx)+2*Lx/3+rand[5]*Lx/2)>=x)
    cond2 = cond2.reshape(-1)
    
    return np.array([40 if ~cond1[i] else rand[6] if cond2[i] else 0 for i in range(len(cond1))]).reshape(x.shape)

if __name__ == "__main__":
    rand_sqre = [(-8.0,8.0),(-8.0,8.0),(4.0,15.0),(0.0,0.4),(0.0,1.0)]
    rand_harm = [(0.0,0.16),(0.0,0.16),(-8.0,8.0),(-8.0,8.0)]
    rand_grad = [(-8.0,8.0),(-8.0,8.0),(4.0,15.0),(0.0,0.4),(0.0,1.0),(-3.0,3.0),(-3.0,3.0)]
    rand_dubl = [(-8.0,8.0),(-8.0,8.0),(4.0,15.0),(0.0,0.4),(0.0,1.0),(0.05,0.2),(2.0,18.0)]
    '''
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
    '''
    #run to test potentials with plots
    from Generator import Generator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    test_gen = Generator(double_pot,rand_dubl)
    test_gen.gen_samps(10)
    
    i = 0
    x,y = np.meshgrid(np.linspace(-20,20,128),np.linspace(-20,20,128))
    for s in test_gen.samps:
        fig, ax = plt.subplots()
        c = ax.contourf(x,y,s[:-1].reshape(x.shape),levels=50,cmap=cm.YlGnBu,vmax=20)
        fig.colorbar(c)
        plt.savefig("plot"+str(i)+".png")
        plt.close()
        i += 1
    