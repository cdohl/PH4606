import numpy as np
import time
from scipy import weave
from scipy.weave import converters

class WaveSolverInPML(object):
    
    """
    WaveSolver       Class calculates the wave equation in a homogeneous medium
    with a prefectly matched layer
    nx:              size of square calculation grid
    wavelength:      wavelength of the wave
    CFL:             CFL parameter dt*c/dx should be smaller than 1/sqrt(2)
    sim_duration:    duration of simulation
    pml_length:      size of PML layer
    plotcallback:    callback function for plotting/saving the result
                     the callback receives two parameters self.u and time
    output:          number of outputs per period of the source
    slowdown:        sleep slowdown secondes between after the calculations of output
    rho1, c1:        density and speed of sound of background phase
    rho2, c2:        density and speed of sound of scatterers
    src_cycles:      duration of the src in cycles (1/frequency)
    
    Following the numerical scheme derived in http://arxiv.org/pdf/1001.0319v1.pdf
    Marcus J. Grote and Imbo Sim for the PML, accounts for inhomogeneous medium (rho & c)
    """
    
    def __init__(self, nx=128, wavelength=.3, CFL=0.1,\
                 sim_duration=2., output=5, slowdown=0, plotcallback=None,\
                 pml_length=16, rho1=1., c1=1., rho2=5., c2=1., src_cycles=1):
        
        self.__pml_length = pml_length
        self.__nx = nx+2*self.pml_length
        self.__ny = self.__nx
        self.__size = 2. #excluding the PML
        self.__dx = self.__size/(self.__nx-2*self.pml_length-1)
        self.__c1 = c1
        self.__rho1 = rho1
        self.__c2 = c2
        self.__rho2 = rho2
        self.__sim_duration = sim_duration
        self.__slowdown = slowdown
        self.wavelength = wavelength
        self.__u  = np.zeros((self.__nx,self.__ny)) #amplitude at t
        self.__un = np.zeros((self.__nx,self.__ny)) #amplitude at t-dt
        self.__unn = np.zeros((self.__nx,self.__ny)) #amplitude at t-2*dt
        self.__rho = np.ones((self.__nx,self.__ny))*self.__rho1
        self.__cc = np.ones((self.__nx,self.__ny))*self.__c1*self.__c1
        self.CFL = CFL
        
        #PML variables
        self.__zeta1 = np.zeros(self.__nx)
        self.__zeta2 = np.zeros(self.__ny)
        self.__phix = np.zeros((self.__nx,self.__ny))
        self.__phiy = np.zeros((self.__nx,self.__ny))
        self.__phixn = np.zeros((self.__nx,self.__ny))
        self.__phiyn = np.zeros((self.__nx,self.__ny))
        self.__zetam = 20.
        
        self.output = output 
        self.__n = 0
        self.__src_function = None
        if src_cycles is 1:
            self.__src_timefunction = self.__gausspulse
            self.__src_duration = (1./self.__nu+self.__src_t0)*1.1
        else:
            self.__src_timefunction = self.__sinetrain
            self.__src_tmax=src_cycles/self.__nu+self.__src_t0           
            self.__src_duration = (self.__src_t0+src_cycles/self.__nu+1./self.__nu*0.1)
       
        if plotcallback is not None:
            self.__plotcallback = plotcallback
        else:
            self.__plotcallback = self.simplecallback

        self.__absorber_y = []
        
        self.__init_pml()

    def __init_pml(self):
        #setup the PML using a 2-times differentiable expression
        for i in range(0,self.pml_length):
            xl=float(self.pml_length-i)/self.pml_length
            self.__zeta1[i]=self.__zetam*\
                (xl-np.sin(2.*np.pi*xl)/2./np.pi)
            self.__zeta1[self.__nx-i-1]=self.__zetam*\
                (xl-np.sin(2.*np.pi*xl)/2./np.pi)
            self.__zeta2[i]=self.__zetam*\
                (xl-np.sin(2.*np.pi*xl)/2./np.pi)
            self.__zeta2[self.__nx-i-1]=self.__zetam*\
                (xl-np.sin(2.*np.pi*xl)/2./np.pi)
    
    def simplecallback(self,u,t):
        """Prints the current time"""
        print "time {0:.2}".format(t)
    
    def cfg_simple(self):
        #Source
        self.__emissionlength=self.__nx-2*self.pml_length
        self.__src_emissionlength = self.__nx-2.*self.pml_length
        self.__src_starty = int(self.__nx/2.-self.__src_emissionlength/2.)
        self.__src_function = self.__planesource
        #Timestepper
        self.__timestepper = self.__inhomogeneous_PML_stable

    def cfg_simplediffraction(self, disks=None, rects=None):

        self.__emissionlength=self.__nx-2*self.pml_length
        self.__src_emissionlength = self.__nx-2.*self.pml_length
        self.__src_starty = int(self.__nx/2.-self.__src_emissionlength/2.)
        self.__src_posx = self.pml_length
        self.__src_function = self.__planesource
        #need to account for PML size
        add_pml=self.pml_length*self.__size/(self.__nx-2.*self.pml_length)
        rx = self.__size/(self.__nx-2*self.pml_length)*self.__nx
        ry = self.__size/(self.__ny-2*self.pml_length)*self.__ny
        yy, xx = np.meshgrid(np.linspace(0., rx, self.__nx),
                             np.linspace(0., ry, self.__ny))
        #Discs
        if disks is not None:
            disks['xpos'][:] += add_pml
            disks['ypos'][:] += add_pml
            #change impedance
            for i in range(disks.size):
                a=np.where((xx-disks['xpos'][i])**2.+
                              (yy-disks['ypos'][i])**2.<(disks['radius'][i]**2.))
                self.__cc[a]=self.__c2*self.__c2
                self.__rho[a]=self.__rho2
        #Rectangles
        if rects is not None:
            rects['x_up'][:] += add_pml
            rects['y_up'][:] += add_pml
            for i in range(rects.size):
                a=np.where((xx>=rects['x_up'][i]) & (xx<=rects['x_up']+rects['width'][i]) 
                           & (yy<=rects['y_up'][i]) & (yy>=rects['y_up'][i]-rects['height'][i]))
                self.__cc[a]=self.__c2*self.__c2
                self.__rho[a]=self.__rho2     
        #update CFL
        self.CFL=self.CFL          
        #Timestepper
        self.__timestepper = self.__inhomogeneous_PML_stable
        
    def cfg_singleslit(self, ow=0.1, pos=0.3):
        """
        Single slit experiment, where slit is oriented along y-axis,
        and the wave is coming from the left. The homogeneous wave
        equation is solved.
        ow:  width of the slit
        pos: x-position in real coordinates [0;size]
        """
        #Source
        self.__src_posx = self.pml_length
        self.__src_emissionlength = int(.9*(self.__nx-2.*self.pml_length))
        self.__src_starty = int(self.__nx/2.-self.__src_emissionlength/2.)
        self.__src_function = self.__planesource
        #Absorber
        yy=np.linspace(0., self.__size, self.__ny-2*self.pml_length)
        self.__absorber_y = np.asarray(np.where((yy>self.__size/2.+ow/2.)|\
                          (yy<self.__size/2.-ow/2.))[0])+self.pml_length
        self.__absorber_x = int(pos/self.__size*(self.__nx-2*self.pml_length))+self.pml_length 
        #Timestepper
        self.__timestepper = self.__inhomogeneous_PML_stable      
             
    def cfg_doubleslit(self, ow=0.1, dw=1., pos=0.3):
        """
        Double slit experiment, where the slits are oriented along y-axis,
        and the wave is coming from the left. The homogeneous wave
        equation is solved.
        ow:  width of the slits
        dw: distance between the slits' centers 
        pos: x-position between [0;size]
        """
        #Source
        self.__src_posx = self.pml_length
        self.__src_emissionlength = int(.9*(self.__nx-2.*self.pml_length))
        self.__src_starty = int(self.__nx/2.-self.__src_emissionlength/2.)
        self.__src_function = self.__planesource
        #Absorber
        yy = np.linspace(0., self.__size, self.__ny-2*self.pml_length)
        self.__absorber_y = np.asarray(np.where((yy>self.__size/2.+dw/2.+ow/2.)|\
            (yy<self.__size/2.-dw/2.-ow/2.)|(yy>self.__size/2.-dw/2.+ow/2.) &\
            (yy<1.+dw/2.-ow/2.))[0])+self.pml_length
        self.__absorber_x = int(pos/self.__size*(self.__nx-2*self.pml_length))+self.pml_length   
        #Timestepper
        self.__timestepper = self.__inhomogeneous_PML_stable

    def cfg_pointsources(self, points):
        #Source
        self.__src_posxa = np.asarray(map(int,points['xpos']/self.__size*\
                                          (self.__nx-2*self.pml_length)))+self.pml_length
        self.__src_posya = np.asarray(map(int,points['ypos']/self.__size*
                                          (self.__ny-2*self.pml_length)))+self.pml_length
        self.__src_ampla = np.asarray(points['ampl'])    
        self.__src_function = self.__pointsource
        #Timestepper
        self.__timestepper = self.__inhomogeneous_PML_stable
    
    def __pointsource(self, t):
        if t<self.__src_duration:
            self.__un[self.__src_posxa, self.__src_posya]=\
                self.__src_ampla*self.__src_timefunction(t)
                 
    def __planesource(self, t):
        if t<self.__src_duration:
            self.__un[self.__src_posx, self.__src_starty:self.__src_starty+self.__src_emissionlength]=\
                self.__src_timefunction(t)

    def __gausspulse(self, t):
        bw= 1.4
        bwr=-6
        ref = 10.0**(bwr / 20.0)
        a = -(np.pi * self.__nu * bw) ** 2 / (4.0 * np.log(ref))
        return -np.exp(-a *(t-self.__src_t0)**2.)*np.sin(2.*np.pi*self.__nu*(t-self.__src_t0))

    def __sinetrain(self, t):
        slp=10.*self.__nu #how fast the pulse rises
        return -np.cos(2.*np.pi*self.__nu*(t-self.__src_t0))/4.\
               *(1-np.tanh((t-self.__src_tmax)*slp))*(1.+np.tanh((t-self.__src_t0)*slp))

    @property
    def pml_length(self):
        """size of PML layer"""
        return self.__pml_length
    
    @property
    def u(self):
        """Wave on the grid"""
        return self.__u
    
    @property
    def CFL(self):
        """CFL parameter (see class decription)"""
        return self.__CFL
    
    @property
    def dt(self):
        """Length of each timestep"""
        return self.__dt

    @property
    def nx(self):
        """X-size of the computational domain without PML"""
        return self.__nx-2*self.pml_length
 
    @property
    def ny(self):
        """Y-size of the computational domain without PML"""
        return self.__ny-2*self.pml_length
    
    @property
    def nu(self):
        """Frequency"""
        return self.__nu
    
    @property
    def omega(self):
        """Angular frequency"""
        return 2.*self.__nu*np.pi
    
    @property
    def nt(self):
        """Number of timesteps"""
        return int(self.__sim_duration/self.__dt)
    
    @property
    def wavelength(self):
        """Wavelength"""
        return self.__wavelength

    @property
    def c1(self):
        """Background speed of sound"""
        return self.__c1
    
    @property
    def nu(self):
        """Frequency"""
        return self.__nu
    
    @property
    def output(self):
        """List of output timesteps using the plotcallback function"""
        return self.__output
    
    @output.setter
    def output(self, output):
        self.__output=int(1./self.nu/self.__dt/output)
            
    @CFL.setter
    def CFL(self, CFL):
        self.__CFL = CFL  #CFL number < 1/sqrt(2)
        self.__dt = CFL*self.__dx/self.__cc.max()**.5
        self.__nt = int(self.__sim_duration/self.__dt)
    
    @wavelength.setter
    def wavelength(self, wavelength):
        self.__wavelength = wavelength
        self.__nu=self.__c1/self.__wavelength
        self.__src_t0=1./self.__nu
          
    def __inhomogeneous_PML_stable(self):   
        u = self.__u
        un = self.__un
        unn = self.__unn
        dt = float(self.__dt)
        dx = self.__dx
        nx = self.__nx
        ny = self.__ny
        zeta1 = self.__zeta1
        zeta2 = self.__zeta2
        cc = self.__cc
        rho = self.__rho
        phix = self.__phix
        phiy = self.__phiy
        phixn = self.__phixn
        phiyn = self.__phiyn
        
        code = """
            for (int i=1; i<nx-1; ++i) {
                for (int j=1; j<ny-1; ++j) {
                    u(i,j) = 1./(1./(dt*dt)+(zeta1(i)+zeta2(j))/(2.*dt))*
                            ( un(i,j)*(2./dt/dt-zeta1(i)*zeta2(j))
                             +unn(i,j)*((zeta1(i)+zeta2(j))/2./dt-1./dt/dt)
                             +cc(i,j)/dx/dx*((un(i-1,j)+un(i+1,j)+un(i,j-1)+un(i,j+1)-4.*un(i,j))
                                             -0.25/rho(i,j)*
                                             ((rho(i+1,j)-rho(i-1,j))*(un(i+1,j)-un(i-1,j))
                                             +(rho(i,j+1)-rho(i,j-1))*(un(i,j+1)-un(i,j-1))))
                             +0.5/dx*( phixn(i,j-1)+phixn(i,j)-phixn(i-1,j-1)-phixn(i-1,j)
                                      +phiyn(i-1,j)+phiyn(i,j)-phiyn(i-1,j-1)-phiyn(i,j-1)));
                 }
            }
            for (int i=1; i<nx-1; ++i) {
                for (int j=1; j<ny-1; ++j) {
                    phix(i,j) = 1./(1./dt+zeta1(i)/2.)*(phixn(i,j)*(1./dt-zeta1(i)/2.)
                                 +(zeta2(j)-zeta1(i))*cc(i,j)*.25/dx*
                                   ( u(i+1,j+1)+u(i+1,j)-u(i,j+1)-u(i,j)
                                    +un(i+1,j+1)+un(i+1,j)-un(i,j+1)-un(i,j)));
                    phiy(i,j) = 1./(1./dt+zeta2(j)/2.)*(phiyn(i,j)*(1./dt-zeta2(j)/2.)
                                 +(zeta1(i)-zeta2(j))*cc(i,j)*.25/dx*
                                   ( u(i+1,j+1)+u(i,j+1)-u(i+1,j)-u(i,j)
                                    +un(i+1,j+1)+un(i,j+1)-un(i+1,j)-un(i,j)));
                }
            }
        """
        weave.inline(code,['u', 'un', 'unn', 'dt', 'dx','nx','ny','zeta1','zeta2','cc','rho',
                           'phixn','phiyn','phix','phiy'],
                   type_converters = converters.blitz)#, compiler = 'gcc')
    
    def __timestep(self):
        
            #Call Source     
            if self.__src_function:
                self.__src_function(self.__n*self.__dt)
            
            #One time step        
            self.__timestepper()
            
            #Impose reflective b.c. at pml boundary                 
            self.__u[0,:] = self.__u[1,:]
            self.__u[-1,:] = self.__u[-2,:]
            self.__u[:,0] = self.__u[:,1]
            self.__u[:,-1] = self.__u[:,-2]

            #Consider an absorber using plane wave radiation b.c.
            if np.size(self.__absorber_y)>0:
                self.__u[self.__absorber_x,self.__absorber_y] = -self.__dx/self.__dt/\
                    self.__c1*\
                    (self.__u[self.__absorber_x-1,self.__absorber_y]-\
                     self.__un[self.__absorber_x-1,self.__absorber_y])+\
                     self.__u[self.__absorber_x-1,self.__absorber_y]
                self.__u[self.__absorber_x+1,self.__absorber_y] = 0.                

            #save values for the time derivative 
            self.__unn = self.__un.copy() #n-1 time step
            self.__un = self.__u.copy()   #n time step
            self.__phixn = self.__phix.copy()
            self.__phiyn = self.__phiy.copy()

    def solvestep(self):
        """
        Solves the wave PDE for the time given set by output parameter.
        Function return False as long as t < than sim_duration
        """
        if self.__n>=self.__nt:
            return True
        
        for self.__n in range (self.__n,self.__n+self.__output):
            self.__timestep()

        self.__plotcallback(self.__u[self.pml_length:-self.pml_length,
                                     self.pml_length:-self.pml_length],
                            self.__n*self.__dt)
        time.sleep(self.__slowdown)
        
        return False