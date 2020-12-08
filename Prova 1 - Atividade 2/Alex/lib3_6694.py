"""
Version: 3.0
Created for 'Sistemas Inteligentes'                                                                                                       
Library for a MLP. The classes are:
mlpdata: stores the data X(inputs) and Y(outputs) and can normalize then and split
MLP: it the mlp, a generic mlp with n layers
BEGIN: November 12,2020
FINISH: November 17,2020
ALTERATION: The input layer is given with the hidden layer nl[0]= input size

"""
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from scipy import optimize

class mlpdata:
    def __init__(self,data,Typ=None):
        """
        Storage the data from x (input) and y(output)
        x: input data (raw)
        y: output data (raw)
        wl: list with numbers of neurons in each layer
        DATE: November 12,2020
        """
        if Typ == 'acc':
            self.x = data
            self.y = None
            self.n = len(data)
        elif Typ == 'wine':
            self.x = data[:,1:]
            self.y = data[:,0]
        self.i = None 
        self.typ = Typ
        self.xn = None 
        self.yn = None

    def normalize(self):
        """
        Normalize the matrices between -1 and 1
        DATE: November 12,2020
        """
        if self.typ == 'wine':
            m = np.matrix(np.min(self.x,axis=1)).T
            M = np.matrix(np.max(self.x,axis=1)).T
            self.xn = -1 +2*(self.x -m)/(M-m)
            self.yn = (np.arange(np.max(self.y)+1)==self.y[:,None]).astype(float)
            self.yn = self.yn[:,1:]
        elif self.typ == 'acc':
            m = min(self.x)
            M = max(self.x)
            self.xn = -1 +2*(self.x -m)/(M-m)

    def split(self,EXIT=1):
        """
        Separete the complete data (sef.x and self.y) into train, validation and test
        (X,Y) train = 80%(few samples) or 60%
        (X,Y) validation = 0%(few samples) or 20%
        (X,Y) test = 20%
        EXIT: number neurons in the last layer
        The next part was made for the 'Avaliação1_parte2
        RETURN: all the inputs(X,Xv,Xt) and outputs(Y,Yv,Yt) plus the numbers of samples in each group
        DATE: November 2,2020
        """
        if self.typ == 'wine':
            X = np.vstack([self.xn[0:30,:],self.xn[59:94,:],self.xn[130:154,:]])
            Xv = np.vstack([self.xn[30:45,:],self.xn[94:112,:],self.xn[154:166,:]])
            Xt = np.vstack([self.xn[45:59,:],self.xn[112:130],self.xn[166:178,:]])
            if EXIT == 1:
                Y = np.block([self.yn[0:30],self.yn[59:94],self.yn[130:154]])
                Yv = np.block([self.yn[30:45],self.yn[94:112],self.yn[154:166]])
                Yt = np.block([self.yn[45:59],self.yn[112:130],self.yn[166:178]])
            elif EXIT == 3:
                Y = np.vstack([self.yn[0:30,:],self.yn[59:94,:],self.yn[130:154,:]])
                Yv = np.vstack([self.yn[30:45,:],self.yn[94:112,:],self.yn[154:166,:]])
                Yt = np.vstack([self.yn[45:59,:],self.yn[112:130,:],self.yn[166:178,:]])
            N,Natr = np.shape(X)# N: number of samples in X/Y, Natr: number of attributes
            return X,Y,Xv,Yv,Xt,Yt

        elif self.typ == 'acc':
            n = floor(self.n*0.6)
            nv = floor(self.n*0.2)
            x = self.xn[1:n]
            xv = self.xn[n:n+nv]
            xt = self.xn[n+nv:]
            return x,xv,xt
        
        
class MLP:
    def __init__(self,WL,Mb=0):
        """
        Create a MPL object with the layers
        wl: list with numbers of neurons in each layer
        w[0]:input layer
        w[n]: output layer
        i: number(s) of input(s)
        Mb: number od samples in the mini batch
        DATE: November 12,2020
        """
        self.wl = WL
        self.w = self.weight()
        self.nw = self.nw()
        self.mb = Mb

    def nw(self):
        """
        Sum the numbers of weights in the mpl
        """
        Nw = 0
        for ii in range(1,len(self.wl)):
            Nw += self.wl[ii] * (self.wl[ii-1]+1)
        return Nw

    def weight(self):
        """
        Create layers with neurons
        i: number of input(s)
        RETURN: the list with the weghts matrices
        DATE: November 12,2020
        """
        w = [] # Where the weights will be stored
        for ii in range(1,len(self.wl)):
            Wii = 0.1 * (-1. + 2.*np.random.rand(self.wl[ii],self.wl[ii-1]+1))
            w.append(Wii) #Change this list with a DLL or SLL
        return w

    def divide(self,X,Y,Typ):
        """
        Separete the array X and Y according to the type (online,batch,minibatch)
        X: input
        Y: output
        Typ: type of distribution
        RETURN: xl (array of inputs), yl (array of outputs)
        DATE: November 16,2020
        """
        if Typ == 'online': 
            n = X.shape[1]
            xl = np.array([X[:,0+k:k+1] for k in range(n)])
            if self.wl[-1] > 1:
                yl = np.array([Y[:,0+k:k+1] for k in range(n)])
            else:
                yl = np.array([Y[0+k:k+1] for k in range(n)])
        elif Typ == 'batch':
            xl = np.array([X])
            yl = np.array([Y])
        elif Typ == 'mini batch': #mini batch
            n = X.shape[1]
            xl = [X[:,0+k:k+self.mb] for k in range(0,n,self.mb)]
            if self.wl[-1] > 1:
                yl = [Y[:,0+k:k+self.mb] for k in range(0,n,self.mb)]
            else:
                yl = [Y[0+k:k+self.mb] for k in range(0,n,self.mb)]
        return xl,yl

    def shuffle(self,X,Y):
        """
        Shuffles the x and y in the same order
        x: input matrix
        y: output matrix
        Typ: type of distribution
        RETURN: x and y shuffled
        DATE: November 12,2020
        """
        n = X.shape[1]
        index = np.random.permutation(n)
        xs = X[:,index]
        if self.wl[-1] > 1:
            ys = Y[:,index]
        else:
            ys = Y[index]
        return xs,ys,n #xs/ys: x/y shuffled , n: number of samples in each batch
    
    def forward(self,X,Y,N,EJc=False,VJ=None,Val=False,Test=False):
        """
        The function take the input (x) and compare with the output (y)
        All the layers have the same activation function (tanh)
        X/Y: input/output
        EJ: evaluate for J
        Val: when validation input
        Test: when test input
        RETURN: yw,error,  y(for test) and Ew(for validation)
        DATE: November 14,2020
        """
        if EJc:
            w = self.update(VJ)
        else:
            w = self.w
        yw = [] #list with the output of each layer
        yw.append(X) #append the first input
        yf = 0
        x1 = X
        for ii in range(len(self.w)): #do it for each layer
            x1 = np.vstack([x1,np.ones(N)]) #the input[l] = output[l-1], ones(n)
            if ii < len(self.w)-1: #repeat except for the last
                x1 = np.tanh(w[ii] @ x1) #tanh -> activation function
                yw.append(x1) #save the output in the list
            else: #don't use an activation function in the last layer
                yf = w[ii] @ x1 #xw: output from the network
        error = yf - Y #error: last output - desired output
        yw = [ np.array(i) for i in yw] #uneable to yw[n]*yw[n] if it's not an array
        if Val: #if the method is using validation data
            verror = error.flatten()
            Ew = (0.5/N)*(verror @ verror)
            if Test: #if the method is using the test data
                return yf, Ew #for test input
            else:
                return Ew #for validation input
        else:
            return yw, error #error: error of the exit, yw: output from W[0] to W[l-1]
    
    def back_prop(self,YW,Error,N):
        """
        Takes the list of outputs and error to calculate the gradient of each weight
        yw: list of ouputs from W[l] to W[1], W[0] is the first input
        error: the error in the last ouput
        n: number of samples
        eJ: evaluete for J
        RETURN: EW (weight error) dEw (derivate) 
        DATE: November 14,2020
        """
        delt = 0
        dw = []
        yw = YW[::-1] #the layers output are flip, yw[l-1] -> yw[0]
        w = self.w[::-1] #the layers are flip, w[l] -> w[0]
        wl = self.wl[::-1][:-1]#flip the list with numbers of neurons, wl[-1] -> wl[0]
        for ii in range(len(self.w)):
            if ii == 0: #for thei last layer
                delt = 2*Error #d(f)/dv = 1
                dw.append((1/N)*(delt @ np.vstack([yw[ii], np.ones(N)]).T)) # 1/n * delt @ x2, x2 = vstack(y1,ones)
            else:
                delt = (np.array(w[ii-1][:,0:wl[ii]].T @ delt))*(1.0 - yw[ii-1]*yw[ii-1])
                dw.append((1/N)*(delt @ np.vstack([yw[ii], np.ones(N)]).T))
        dw = dw[::-1]
        dEw = np.block([ii.flatten() for ii in dw])
        verror = Error.flatten()
        Ew = (0.5/N)*(verror @ verror)
        return Ew,dEw

    def step_alpha(self,D,dEw,Wmax, Wmin):
        """
        Calculate the step for each weight
        dEw: derivative
        Wmax: maximum valor of weight
        Wmin: minimum valor of weight
        RETURN: alpha and d(direction)
        DATE: November 15,2020
        """
        vw = np.block([ii.flatten() for ii in self.w])
        alpha = np.zeros(self.nw)
        for ii in range(self.nw):
            if D[ii]>0.:
                alpha[ii] = (Wmax[ii]-vw[ii])/D[ii]
            elif D[ii]<0.:
                alpha[ii] = (Wmin[ii]-vw[ii])/D[ii]
            elif D[ii] == 0.:
                alpha[ii] = 1e9

        if all(np.equal(alpha,1e9*np.ones(self.nw))):
            alpha = np.zeros(self.nw)

        return min(alpha),vw

    def raurea(self,X, Y, N, VW, Imin, Imax, D,T_x):
        """
        Use the aurea proportion to find the best step (alpha)
        X: input data
        Y: output data
        Imin: minimum value (0)
        Imax: maximum value (alpha)
        D:direction array
        T_x: tolerance of the data
        RETURN: (a[ii]+b[ii])/2 = step
        DATE: November 15,2020
        """
        k = 0.618033988749895
        jj = 0
        a,b,vl,vmi,fl,fmi = [],[],[],[],[],[]
        a.append(Imin)
        b.append(Imax)

        vl.append(a[jj] + (1. - k)*(b[jj]-a[jj])) #lambda
        vmi.append(a[jj]+k*(b[jj]-a[jj]))
        
        fl.append(self.forward(X,Y,N,EJc=True,VJ=VW+vl[jj]*D,Val=True))
        fmi.append(self.forward(X,Y,N,EJc=True,VJ=VW+vmi[jj]*D,Val=True))
        while (b[jj]-a[jj]) >= T_x:
            if fl[jj] > fmi[jj]:
                a.append(vl[jj])
                b.append(b[jj])
                vl.append(vmi[jj])
                vmi.append(a[jj+1] + k*(b[jj+1]-a[jj+1]))
                fl.append(fmi[jj])
                fmi.append(self.forward(X,Y,N,True,VW+vmi[jj+1]*D,True))
            else:
                a.append(a[jj])
                b.append(vmi[jj])
                vmi.append(vl[jj])
                vl.append(a[jj+1] + (1.-k)*(b[jj+1]-a[jj+1]))
                fl.append(self.forward(X,Y,N,True,VW+vl[jj+1]*D,True))
                fmi.append(fl[jj])
            jj += 1
        return (a[jj]+b[jj])/2 #best alpha
                

    def update(self,VW):
        """
        This function updates the weights 
        vw: array with the new weights
        RETURN: w: layers with new weights 
        DATE: November 15,2020
        """
        w = []
        t0 = 0
        t1 = 0
        for ii in range(1,len(self.wl)): 
            t1 += self.wl[ii]*(self.wl[ii-1]+1)
            wii = np.reshape(VW[t0:t1],(self.wl[ii],self.wl[ii-1]+1))
            t0 = t1
            w.append(wii)
        return w

    def cost(self,VW,X,Y,N):
        """
        Same as forward(Val=True), created for the optimize.minimize
        VW: vector of weights
        X: input
        Y: desired output
        RETURN: Ew (Mean error)
        DATE: November 26,2020
        """
        w = self.update(VW) #rebuild the weights with vw
        yw = [] #list with the output of each layer
        yw.append(X) #append the first input
        yf = 0
        x1 = X
        for ii in range(len(self.w)): #do it for each layer
            x1 = np.vstack([x1,np.ones(N)]) #the input[l] = output[l-1], ones(n)
            if ii < len(self.w)-1: #repeat except for the last
                x1 = np.tanh(w[ii] @ x1) #tanh -> activation function
                yw.append(x1) #save the output in the list
            else: #don't use an activation function in the last layer
                yf = w[ii] @ x1 #xw: output from the network
        error = yf - Y #error: last output - desired output
        yw = [ np.array(i) for i in yw] #uneable to yw[n]*yw[n] if it's not an array
        Ew, dEw = self.back_prop(yw,error,N)
        dEw = dEw/np.linalg.norm(dEw)
        return Ew,dEw
 

    def train(self,X,Y,Xv,Yv,Epochs,Wlim,Typ,T_g,T_x,Typ2 ='SGD',Shuf=True):
        """
        Train the network with x an y, validates with xv and yv.
        X/Y: Train data
        Xv/Yv: Validation data
        Epochs: number of epochs
        Wlim: limit of the weights
        Typ: train meyhod
        T_g: gradient tolerance
        T_x: Tolerance
        Typ2: train optimization method
        Shuf: if you want to shuffle the data
        RETURN: the square error from the training and validation
        DATE: November 12,2020
        """
        EQM_T = np.zeros(Epochs+1) #Square error (train)
        EQM_V = np.zeros(Epochs+1) #Square error (validation)
        wmax = Wlim * np.ones(self.nw)
        wmin = -wmax
        k = 0
        dEw = np.ones(self.nw)
        X,Y = self.divide(X,Y,Typ)
        if Shuf and Typ == 'onilne': #shuffle before start traning for online method
            X,Y,n = self.shuffle(X,Y)
        while (np.linalg.norm(dEw) >= T_g and k <= Epochs):
            for x,y in zip(X,Y):
                if Shuf and Typ != 'online': #if shuffle = True, for method batch and mini batch
                    x,y,n = self.shuffle(x,y) #shuffle train
                else:
                    n = 1 #size for an online training
                yw,error = self.forward(x,y,n) #forward propagation
                Ew,dEw = self.back_prop(yw,error,n) #back propagation
                if Typ2 == 'SGD': # for SGD method 
                    d = -dEw/np.linalg.norm(dEw) #minimize function
                    alphamax,vw = self.step_alpha(d,dEw,wmax,wmin) #find the max value for alpha
                    alpha = self.raurea(x,y,n,vw,0,alphamax,d,T_x) #use the bisection method
                    vw += alpha*d #new weights
                else: #CG and BFGS
                    vw = np.block([i.flatten() for i in self.w])
                    r = optimize.minimize(self.cost,vw,args=(x,y,n),method=Typ2,jac=True,options={'maxiter':3})
                    vw = r.x
                self.w = self.update(vw) #rebulding the layers
            EQM_T[k] = Ew.copy() #Mean Square Error (train)
            xv,yv,nv = self.shuffle(Xv,Yv)
            EQM_V[k] = self.forward(xv,yv,nv,False,None,True).copy() #Mean Square Error (validation)
            if k%10 == 0:
                print('Epochs: {:4} EQM_T = {:.6f}   EQM_V = {:.6f}'.format(k,EQM_T[k],EQM_V[k]))
            k +=  1
        EQM_T = EQM_T[0:k]
        EQM_V = EQM_V[0:k]
        return EQM_T,EQM_V

    def test(self,XT,YT):
        """
        Test the MLP
        XT: test input
        YT: test ouput
        RETURN: y (mlp ouyput),Ew (Mean Square Error)
        DATE: November 20,2020
        """
        nt = XT.shape[1]
        y,Ew = self.forward(XT,YT,nt,EJc=False,VJ=None,Val=True,Test=True)
        return y,Ew

def prediction(X,M,Step): #Make an input and output with a shift
    """
    Make an input matrix and ouput vector for a prediction
    X: 1-D array with the data
    M: number of inputs
    Step: the step of the shift
    RETURN: xn (input matrix), yn (output vector)
    DATE: November 23,2020
    """
    n = len(X)
    xn = np.zeros((M,n-M-1))
    yn = np.zeros(n-M-1)
    vn = np.arange(M,n-Step+1)
    for ii in vn:
        xn[:,ii-M] = np.flip(X[0:ii][-M:])
        yn[ii-M] = X[ii+Step-1]
    return xn,yn

def plotEVO(N,Eqm1,Typ1,Eqm2,Typ2,Title):
    """
    Plot the error evolution for a train and validation data
    N: number of the plot
    Eqm1: MSE train
    Typ1: color of the train data
    Eqm2: MSE validation
    Typ2:color of the validation data
    Title: Title
    RETURN: plot
    DATE:November 23,2020
    """
    plot = plt.figure(N)
    t = np.arange(len(Eqm1))
    plt.plot(t,Eqm1,Typ1,t,Eqm2,Typ2, lw=0.5)
    plt.title(Title)
    plt.legend(['Train','Validation'])
    plt.xlabel('Épocas')
    plt.ylabel('EQM')
    return plot

def plotEVO2(Eqmt1,Eqmv1,Eqmt2,Eqmv2,Title1,Title2):
    """
    Plot two graphs in the same window (subplot)
    Eqmt1/Eqmt2: MSE train for mlp1/2
    Eqmv1,Eqmv2: MSE validation for mlp1/2
    Title1/Title2: titles
    RETURN: plot
    DATE: November 23,2020
    """
    t = np.arange(len(Eqmt1))
    fig, axs = plt.subplots(2)
    axs[0].set_title(Title1)
    axs[1].set_title(Title2)
    axs[0].plot(t,Eqmt1,lw=0.5,c='b',label='Train')
    axs[0].plot(t,Eqmv1,lw=0.5,c='r',label='Validation')
    axs[1].plot(t,Eqmt2,lw=0.5,c='b',label='Train')
    axs[1].plot(t,Eqmv2,lw=0.5,c='r',label='Validation')
    plt.legend()
    return fig

def plotPrediction(N,Y1,Y2,Title):
    """
    Plot the desired curve and the curve generated by the MLP
    N: number of the plot
    Y1: desired output
    Y2: MLP output
    RETURN: plot
    DATE: November 23,2020
    """
    plot = plt.figure(N)
    t = np.arange(len(Y1))
    plt.plot(t,Y1,'k',t,Y2,'r', lw=0.5)
    plt.title(Title)
    plt.legend(['Real','MLP'])
    plt.xlabel('Épocas')
    return plot

