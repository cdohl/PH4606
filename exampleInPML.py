%matplotlib inline
from IPython import display #for continous display
import matplotlib.pyplot as plt #plotting

def plotwave(u,time):
    plt.clf()
    #plot the pressue field
    plt.imshow(np.flipud(np.transpose(u)), origin='upper', extent=[0., 2., 0., 2.],
               vmax=2, vmin=-2) #plot the wave field
    plt.text(0.1,1.8,"time {0:.5f}".format(time)) #annotate the time
    plt.gca().set_xlim([0.,2.])
    plt.gca().set_ylim([0.,2.])
    display.clear_output(wait=True)
    display.display(plt.gcf())

    
    
def linesource(a):
    n_p=5
    p = np.zeros(n_p, dtype={'names':['ampl', 'xpos', 'ypos'],
                                 'formats':['f8','f8','f8']})
    #horizontal line
    n=0
    dnx=2.*.5/(n_p-1)
    for i in range(n_p):
        p['ampl'][n] = 5.
        p['xpos'][n] = 2.*.5/2.+dnx*i
        p['ypos'][n] = 2.*.5/2.+dnx*i
        n += 1
    a.cfg_pointsources(p)
    return a

def halfcircle(a):
    n_p=5
    p = np.zeros(n_p, dtype={'names':['ampl', 'xpos', 'ypos'],
                                 'formats':['f8','f8','f8']})
    n=0
    for i in range(n_p):
        p['ampl'][n] = 5.
        p['xpos'][n] = .5*np.cos(np.pi/(n_p-1)*i)+1.
        p['ypos'][n] = .5*np.sin(np.pi/(n_p-1)*i)+1.
        n += 1
    a.cfg_pointsources(p)
    return a

def fresnelplate(a):
    #Fresnel zone plate
    nn=7 #order of plate
    f=1. #focus position
    l=a.c1/a.nu #wavelength
    yy=np.linspace(0.,2.,a.ny)
    ampl=np.zeros(a.ny)
    n=0
    #make a zoneplate
    for i in range(0,nn,2):
        k=np.where(((yy-1.)<(((i+1)*l*(f+(i+1)*l/4.))**.5)) &
                   ((yy-1.)>((i*l*(f+i*l/4.))**.5)))
        ampl[k]=1.
        ampl[a.ny-np.asarray(k)]=1.

    n_p=np.asarray(np.where(ampl==1.))
    p = np.zeros(n_p.size, dtype={'names':['ampl', 'xpos', 'ypos'],
                                 'formats':['f8','f8','f8']})
    p['ampl'][:]=1.
    p['xpos'][:]=0.
    p['ypos'][:]=yy[n_p]
    a.cfg_pointsources(p)
    return a

def gridofdisks(a):    
    #grid of disks

    ndx=1
    ndy=1
    d = np.zeros(ndx*ndy, dtype={'names':['radius','xpos','ypos'],
                                 'formats':['f8','f8','f8']})
    n=0 
    for i in range(ndx):
        for j in range(ndy):
            d['radius'][n] = .3
            d['xpos'][n] = 2./float(ndx)*float(i)+1.-float(ndx-1.)/float(ndx)
            d['ypos'][n] = 2./float(ndy)*float(j)+1.-float(ndy-1.)/float(ndy)
            n += 1
    a.cfg_simplediffraction(disks=d)
    return a

def simplerectangle(a):
    #reflective rectangle
    nr=1
    r = np.zeros(nr, dtype={'names':['x_up', 'y_up', 'width', 'height'],
                                 'formats':['f8','f8','f8','f8']})
    r['x_up'][0] = 0.5
    r['y_up'][0] = 1.5
    r['width'][0] = .4
    r['height'][0] = 1
    a.cfg_simplediffraction(disks=d)
    return a


plt.figure(1, figsize=(8, 8), dpi=100)

#setup waveobject

a=WaveSolverInPML(nx=256, pml_length=32, CFL=.2, wavelength=.1,
                  output=3, slowdown=0, plotcallback=plotwave, sim_duration=.6, 
                  rho1=.7, c1= 1., rho2=1, c2=.7, src_cycles=1)

a=halfcircle(a)
    
#a.cfg_doubleslit(dw=0.6,ow=0.15,pos=.6)
#a.cfg_simpleconfig()
#a.cfg_singleslit(ow=0.1,pos=.4)

while True:
    if a.solvestep():
        break
    time.sleep(0)