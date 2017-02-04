%matplotlib inline
from IPython import display #for continous display
import matplotlib.pyplot as plt #plotting

def plotwave(u,time):
    plt.clf()
    #plot the pressue field
    plt.imshow(np.transpose(u), origin='upper', extent=[0., 2., 0., 2.], vmax=2, vmin=-2) #plot the wave field
    plt.text(0.1,1.8,"time {0:.5f}".format(time)) #annotate the time
    plt.gca().set_xlim([0.,2.])
    plt.gca().set_ylim([0.,2.])
    display.clear_output(wait=True)
    display.display(plt.gcf())

plt.figure(1, figsize=(8, 8), dpi=100)
#setup waveobject
a=WaveSolverPML(nx=100, pml_length=10, CFL=.4, wavelength=0.2,
                output=5, slowdown=0, plotcallback=plotwave, sim_duration=4.)
#setup an experiment
#a.cfg_simpleconfig()
#a.cfg_singleslit(ow=0.1,pos=.4)
a.cfg_doubleslit(dw=0.6,ow=0.15,pos=.6,src_duration=4.)
#a.cfg_pointsource(src_posx=.1,src_duration=5.)
#start solving it
a.solve()