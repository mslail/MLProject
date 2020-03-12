import argparse
from Generator import Generator
from potentails import *

#VERSON NOTE: all potential types wil work, no energy function (or root finding) for GPW and DBW

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n_samps", type=int, help="numper of samples to generate")
    parser.add_argument("pot_type", type=str, help="3 letter code for potential [HMO,IPW,GPW,DBW]")
    parser.add_argument("-n_per", type=int, default=1000, help="number of samples per file")
    parser.add_argument("-use_nrg", type=bool, default=True, help="use predefined energy func")
    parser.add_argument("-res", type=int, default=128, help="resolution of the potential image")
    args = parser.parse_args()
    
    g = None
    if args.pot_type.upper() == "HMO":
        g = Generator(harmonic_pot, \
                      [(0.0,0.16),(0.0,0.16),(-8.0,8.0),(-8.0,8.0)], \
                      energy=harmonic_nrg if args.use_nrg else None, \
                      size=args.res)
    elif args.pot_type.upper() == "IPW":
        g = Generator(square_pot, \
                      [(-8.0,8.0),(-8.0,8.0),(4.0,15.0),(0.0,0.4),(0.0,1.0)], \
                      energy=square_nrg if args.use_nrg else None, \
                      size=args.res)
    elif args.pot_type.upper() == "GPW":
        g = Generator(grad_pot, \
                      [(-8.0,8.0),(-8.0,8.0),(4.0,15.0),(0.0,0.4),(0.0,1.0),(-3.0,3.0),(-3.0,3.0)], \
                      energy=grad_nrg if args.use_nrg else None, \
                      size=args.res)
    elif args.pot_type.upper() == "DPW":
        g = Generator(double_pot, \
                      [(-8.0,8.0),(-8.0,8.0),(4.0,15.0),(0.0,0.4),(0.0,1.0),(0.05,0.2),(2.0,18.0)], \
                      energy=double_nrg if args.use_nrg else None, \
                      size=args.res)
    else:
        print("invalid arguments given")
    
    if g != None:
        for i in range(int(args.n_samps/args.n_per)):
            g.gen_samps(args.n_per)
            g.export_samps("samples_"+str(args.pot_type)+"_["+str(args.n_per*(i+1))+" of "+str(args.n_samps)+"]")
            g.clear_samps()
        if args.n_samps%args.n_per != 0:
            g.gen_samps(args.n_samps%args.n_per)
            g.export_samps("samples_"+str(args.pot_type)+"_["+str(args.n_samps)+" of "+str(args.n_samps)+"]")
            g.clear_samps()