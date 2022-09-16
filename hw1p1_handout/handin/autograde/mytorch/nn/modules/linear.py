import numpy as np

class Linear:
    
    def __init__(self, in_features, out_features, debug = False):
    
        self.W    = np.zeros((out_features, in_features), dtype="f")
        self.b    = np.zeros((out_features, 1), dtype="f")
        self.dLdW = np.zeros((out_features, in_features), dtype="f")
        self.dLdb = np.zeros((out_features, 1), dtype="f")
        
        self.debug = debug

    def forward(self, A):
    
        self.A    = A
        self.N    = A.shape[0]
        self.Ones = np.ones((self.N,1), dtype="f")
        Z         = self.A @ np.transpose(self.W) + self.Ones @ np.transpose(self.b) # TODO
        
        return Z
        
    def backward(self, dLdZ):
    
        dZdA      = np.transpose(self.W) # TODO C0XC1
        dZdW      = self.A # TODO N X C0
        dZdi      = None
        dZdb      = self.Ones# TODO
        dLdA      = dLdZ @ np.transpose(dZdA) # TODO NXC0 = dLdZ(NXC1) @ C1XC0
        dLdW      = np.transpose(dLdZ) @ dZdW # TODO C1XC0 = C1XN @ NXC0
        dLdi      = None
        dLdb      = np.transpose(dLdZ) @ dZdb # TODO C1X1 = C1XN  @ NX1
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:
            
            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdi = dZdi
            self.dZdb = dZdb
            self.dLdA = dLdA
            self.dLdi = dLdi
        
        return dLdA