import numpy as np
import time
from scipy import weave
from scipy.weave import converters

class WaveSolverPML(object):
    
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
    
    Following the numerical scheme derived in http://arxiv.org/pdf/1001.0319v1.pdf
    Marcus J. Grote and Imbo Sim
    """
    
    def __init__(self, nx=128, wavelength=.3, CFL=0.1,\
                 sim_duration=2., output=5, slowdown=0, plotcallback=None,\
                 pml_length=16):
        
        self.__pml_length=pml_length
        self.__nx=nx+2*self.pml_length
        self.__ny=self.__nx
        self.__size=2. #excluding the PML
        self.__dx=self.__size/(self.__nx-2*self.pml_length-1)
        self.__c1=1.
        self.__sim_duration=sim_duration
        self.__slowdown=slowdown
        self.CFL = CFL
        self.wavelength = wavelength
        self.__u  = np.zeros((self.__nx,self.__ny)) #amplitude at t
        self.__un = np.zeros((self.__nx,self.__ny)) #amplitude at t-dt
        self.__unn= np.zeros((self.__nx,self.__ny)) #amplitude at t-2*dt
        
        #PML variables
        self.__zeta1 = np.zeros(self.__nx)
        self.__zeta2 = np.zeros(self.__ny)
        self.__phix = np.zeros((self.__nx,self.__ny))
        self.__phiy = np.zeros((self.__nx,self.__ny))
        self.__phixn = np.zeros((self.__nx,self.__ny))
        self.__phiyn = np.zeros((self.__nx,self.__ny))
        self.__zetam = 20.
        
        self.output = output 
        self.__src_function = None
        
        if plotcallback is not None:
            self.__plotcallback = plotcallback
        else:
            self.__plotcallback = self.simplecallback

        self.__absorber_y = []
        
        self.__init_pml()

    def __init_pml(self):
        #setup the PML using a 2-times differential equation
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

    def cfg_simpleconfig(self, src_duration=0.1):
        #Source
        self.__src_duration=src_duration
        self.__emissionlength=self.__nx-2*self.pml_length
        self.__src_emissionlength = self.__nx-2.*self.pml_length
        self.__src_starty = int(self.__nx/2.-self.__src_emissionlength/2.)
        self.__src_function = self.__planesource
        #Timestepper
        self.__timestepper = self.__homogeneous_PML_stable
        
    def cfg_singleslit(self, ow=0.1, pos=0.3, src_duration=0.1):
        """
        Single slit experiment, where slit is oriented along y-axis,
        and the wave is coming from the left. The homogeneous wave
        equation is solved.
        ow:  width of the slit
        pos: x-position in real coordinates [0;size]
        """
        #Source
        self.__src_duration = src_duration
        self.__src_posx = self.pml_length
        self.__src_emissionlength = int(.9*(self.__nx-2.*self.pml_length))
        self.__src_starty = int(self.__nx/2.-self.__src_emissionlength/2.)
        self.__src_function = self.__planesource
        #Absorber
        yy=np.linspace(0., self.__size, self.__ny-2*self.pml_length)
        self.__absorber_y = np.asarray(np.where((yy>self.__size/2.+ow/2.)|\
                          (yy<self.__size/2.-ow/2.))[0])+self.pml_length
        self.__absorber_x = int(pos/self.__size*self.__nx-self.pml_length)
        #Timestepper
        self.__timestepper = self.__homogeneous_PML_stable      
             
    def cfg_doubleslit(self, ow=0.1, dw=1., pos=0.3, src_duration=0.1):
        """
        Double slit experiment, where the slits are oriented along y-axis,
        and the wave is coming from the left. The homogeneous wave
        equation is solved.
        ow:  width of the slits
        dw: distance between the slits' centers 
        pos: x-position between [0;size]
        """
        #Source
        self.__src_duration = src_duration
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
        self.__timestepper = self.__homogeneous_PML_stable

    def cfg_pointsource(self, src_posx=1., src_posy=1., src_duration=0.1):
        #Source
        self.__src_duration=src_duration
        self.__src_posx = self.pml_length
        self.__src_posx = int(src_posx/self.__size*(self.__nx-2*self.pml_length))+self.pml_length
        self.__src_posy = int(src_posy/self.__size*(self.__ny-2*self.pml_length))+self.pml_length 
        self.__src_function = self.__pointsource
        #Timestepper
        self.__timestepper = self.__homogeneous_PML_stable
    
    def __pointsource(self, time):
        #pressure point source 
        if time<(self.__src_duration+3.*self.__dt):
            self.__u[self.__src_posx, self.__src_posy]=\
                50.*np.sin(self.omega*(time))*(time<self.__src_duration) #take care it goes to 0
                
    def __planesource(self, time):
        if time<(self.__src_duration+3.*self.__dt):
            self.__u[self.__src_posx, self.__src_starty:self.__src_starty+self.__src_emissionlength]=\
                np.sin(self.omega*(time))*(time<self.__src_duration) #take care it goes to 0
                    
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
    def zeta1(self):
        return self.__zeta1
    
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
        self.__dt = CFL*self.__dx/self.__c1 
        self.__nt = int(self.__sim_duration/self.__dt)
    
    @wavelength.setter
    def wavelength(self, wavelength):
        self.__wavelength = wavelength
        self.__nu=self.__c1/self.__wavelength
          
    def __homogeneous_PML_stable(self):
    
        u = self.__u
        un = self.__un
        unn = self.__unn
        dt = self.__dt
        dx = self.__dx
        nx = self.__nx
        ny = self.__ny
        zeta1 = self.__zeta1
        zeta2 = self.__zeta2
        c = self.__c1
        phix = self.__phix
        phiy = self.__phiy
        phixn = self.__phixn
        phiyn = self.__phiyn
        
        code = """
            for (int i=1; i<nx-1; ++i) {
                for (int j=1; j<ny-1; ++j) {
                    u(i,j) = 1./(1./dt/dt+(zeta1(i)+zeta2(j))/2./dt)*
                            (  un(i,j)*(2./dt/dt-zeta1(i)*zeta2(j))
                             +unn(i,j)*((zeta1(i)+zeta2(j))/2./dt-1./dt/dt)
                             +c*c/dx/dx*(un(i-1,j)+un(i+1,j)+un(i,j-1)+un(i,j+1)-4.*un(i,j))
                             +0.5/dx*( phixn(i,j-1)+phixn(i,j)-phixn(i-1,j-1)-phixn(i-1,j)
                                      +phiyn(i-1,j)+phiyn(i,j)-phiyn(i-1,j-1)-phiyn(i,j-1)));
                 }
            }
            for (int i=1; i<nx-1; ++i) {
                for (int j=1; j<ny-1; ++j) {
                    phix(i,j) = 1./(1./dt+zeta1(i)/2.)*(phixn(i,j)*(1./dt-zeta1(i)/2.)
                                 +(zeta2(j)-zeta1(i))*c*c*.25/dx*
                                   ( u(i+1,j+1)+u(i+1,j)-u(i,j+1)-u(i,j)
                                    +un(i+1,j+1)+un(i+1,j)-un(i,j+1)-un(i,j)));
                    phiy(i,j) = 1./(1./dt+zeta2(j)/2.)*(phiyn(i,j)*(1./dt-zeta2(j)/2.)
                                 +(zeta1(i)-zeta2(j))*c*c*.25/dx*
                                   ( u(i+1,j+1)+u(i,j+1)-u(i+1,j)-u(i,j)
                                    +un(i+1,j+1)+un(i,j+1)-un(i+1,j)-un(i,j)));
                }
            }
        """
        
        weave.inline(code,['u', 'un', 'unn', 'dt','dx','nx','ny','zeta1','zeta2','c',
                           'phixn','phiyn','phix','phiy'],
                   type_converters = converters.blitz)#, compiler = 'gcc')
        return u
  
    def solve(self):
        """Solve the wave equation with perfectly matched layers"""
        for n in range(self.__nt+1): ##loop across number of time steps

            #self.__u=self.__inhomogeneous()
            self.__u=self.__timestepper()
            
            #Impose the boundary conditions                 
            self.__u[0,:] = self.__u[1,:]
            self.__phix[0,:] = self.__phix[1,:]
            self.__phiy[0,:] = self.__phiy[1,:]  

            self.__u[-1,:] = self.__u[-2,:]
            self.__phix[-1,:] = self.__phix[-2,:]
            self.__phiy[-1,:] = self.__phiy[-2,:]   

            self.__u[:,0] = self.__u[:,1]
            self.__phix[:,0] = self.__phix[:,1]
            self.__phiy[:,0] = self.__phiy[:,1]

            self.__u[:,-1] = self.__u[:,-2]
            self.__phix[:,-1] = self.__phix[:,-2]
            self.__phiy[:,-1] = self.__phiy[:,-2]
                
            if np.size(self.__absorber_y)>0:
                self.__u[self.__absorber_x,self.__absorber_y] = -self.__dx/self.__dt/\
                    self.__c1*\
                    (self.__u[self.__absorber_x-1,self.__absorber_y]-\
                     self.__un[self.__absorber_x-1,self.__absorber_y])+\
                     self.__u[self.__absorber_x-1,self.__absorber_y]
                self.__u[self.__absorber_x+1,self.__absorber_y] = 0.                

            #Call Source     
            if self.__src_function:
                self.__src_function(n*self.__dt)

            #save values for the time derivative 
            self.__unn = self.__un.copy() #n-1 time step
            self.__un = self.__u.copy()   #n time step
            self.__phixn = self.__phix.copy()
            self.__phiyn = self.__phiy.copy()

            if n % self.__output is 0:
                self.__plotcallback(self.__u[self.pml_length:-self.pml_length,self.pml_length:-self.pml_length],n*self.__dt)
                time.sleep(self.__slowdown)