# converted from https://cdcvs.fnal.gov/redmine/projects/gm2fieldsignal/repository/revisions/develop/show/Simulation
# used .h and .cxx
# .cu is same implementation as .cxx
# not yet tested

import numpy as np

class ProbeSimulator(object):
    def __init__(self):
        self.fGridSize = 0.01; //mm

        self.fCoilR = 2.3 ;
        self.fCoilL = 15.0;
        self.fCoilN = 29;
        self.fCoilPhiNSeg = 100;

        self.fSampleR = 1.25;
        self.fSampleL = 33.5;

        self.CoilOption = 0;

        self.fNFreq = 10000;

        self.fSamplingPeriod = 62.0/61.74e6;
        self.fFidSamples = 12000;
        self.fPreSamples = 0;

        self.fT2 = 0.04;
        self.fPulseEff = 1.0;
        self.fT0 = 0.0;

        self.fProbeCenter = {0.0, 0.0, 0.0};

    def SetParameter(self, Name, Val):
        if Name == "GridSize":
            self.fGridSize = Val
        elif Name == "CoilR":
            self.fCoilR = Val
        elif Name == "CoilL":
            self.fCoilL = Val
        elif Name == "CoilN":
            self.fCoilN = int(Val)
        elif Name == "CoilPhiN":
            self.fCoilPhiNSeg = int(Val)
        elif Name == "CoilOption":
           self.CoilOption = int(Val)
        elif Name == "SampleR":
            self.fSampleR = Val
        elif Name == "SampleL":
            self.fSampleL = Val
        elif Name == "CoilOption":
            self.CoilOption = int(Val)
        elif Name == "NFrequency":
            self.fNFreq = int(Val)
        elif Name == "SamplingPeriod":
            self.fSamplingPeriod = Val
        elif Name == "NFidSamples":
            self.fFidSamples = int(Val)
        elif Name == "NPreSamples":
            self.fPreSamples = int(Val)
        elif Name == "T2":
            self.fT2 = Val
        elif Name == "T0":
            self.fT0 = Val

    def ConfigProbeType(self, Name):
        if Name == "PlungingProbe":
          self.fCoilR = 7.5;
          self.fCoilL = 11.0;
          self.fCoilN = 5;

          self.fSampleR = 2.12;
          self.fSampleL = 40.0;

          self.fSamplingPeriod = 1e-7;
          self.fFidSamples = 3000000;
          self.fPreSamples = 0;
          self.fT2 = 3.0;
          self.fT0 = 0.0;

          self.fCoilShiftConfig = self.fCoilL/(self.fCoilN-1)*np.arange(0, self.CoilN)

        elif Name == "TrolleyProbe":
          self.fCoilR = 2.3;
          self.fCoilL = 7.0;
          self.fCoilN = 17;

          self.fSampleR = 1.25;
          self.fSampleL = 33.5;

          self.fSamplingPeriod = 62.0/61.74e6;
          self.fFidSamples = 16000;
          self.fPreSamples = 300;
          self.fT2 = 0.04;
          self.fT0 = -300*self.fSamplingPeriod;

          CoilShift = self.fCoilL/(9-1);
          fCoilShiftConfig = []
          for i in range(0,9):
            fCoilShiftConfig.append(i*CoilShift)
          for i in range(9, 17):
            fCoilShiftConfig.append( (i-9)*CoilShift)
          self.fCoilShiftConfig = np.array(fCoilShiftConfig)
        elif Name == "TrolleyProbeLong":
          self.fCoilR = 2.3;
          self.fCoilL = 15.0;
          self.fCoilN = 32;

          self.fSampleR = 1.25;
          self.fSampleL = 33.5;

          self.fSamplingPeriod = 62.0/61.74e6;
          self.fFidSamples = 16000;
          self.fPreSamples = 300;
          self.fT2 = 0.04;
          self.fT0 = -300*self.fSamplingPeriod;

          CoilShift = self.fCoilL/(28-1);
          fCoilShiftConfig = []
          for i in range(0, 28):
            fCoilShiftConfig.append(i*CoilShift)
          for i in range(28, 30):
            fCoilShiftConfig.append((i-28)*CoilShift)
          for i in range(30, 32):
            fCoilShiftConfig.append((i-4)*CoilShift)
          self.fCoilShiftConfig = np.array(fCoilShiftConfig)
        elif Name == "FixedProbe":
          self.fCoilR = 2.3;
          self.fCoilL = 15.0;
          self.fCoilN = 32;

          self.fSampleR = 1.25;
          self.fSampleL = 33.5;

          self.fSamplingPeriod = 1e-6;
          self.fFidSamples = 4096;
          self.fPreSamples = 410;
          self.fT2 = 0.04;
          self.fT0 = -4.4e-4;

          CoilShift = self.fCoilL/(28-1);
          fCoilShiftConfig = []
          for i in range(0, 28):
            fCoilShiftConfig.append(i*CoilShift)
          for i in range(28, 30):
            fCoilShiftConfig.append((i-28)*CoilShift)
          for i in range(30, 32):
            fCoilShiftConfig.append((i-4)*CoilShift)
          self.fCoilShiftConfig = np.array(fCoilShiftConfig)
        self.CoilOption = 1;

    def SetBFieldShape(self, InputField):
        self.fBFieldShape = InputField;

    def Init(self):
        self.fSampleDimL = floor(self.fSampleL/self.fGridSize)+1;
        self.fSampleDimT = floor(self.fSampleR/self.fGridSize)+1;

        self.fB_coil_L = np.zeros(fSampleDimL*fSampleDimT)
        self.fB_coil_T = np.zeros(fSampleDimL*fSampleDimT)
        self.fPosY = np.zeros(fSampleDimL*fSampleDimT)
        self.fPosZ = np.zeros(fSampleDimL*fSampleDimT)

        self._CalculateCoilBField()

    def UpdateProbeCenter(self):
        # Use Monte Carlo Method
        NSpins = 80000000

        rng = np.RandomState(0)
        R = np.sqrt(rng.uniform(0,1,size=NSpins))*self.fSampleR
        Phi = rng.uniform(0,2*np.pi,size=NSpins)
        Z = self.fSampleL*rng.uniform(0,1,size=NSpins)-self.fSampleL/2.

        IndexZ = np.floor((self.fSampleL/2. + Z)/self.fGridSize)
        IndexR = np.floor(R/self.fGridSize)

        avg_z = 0.0
        for r_idx, z_idx, phi, z in zip(IndexR, IndexZ, Phi, Z):
          B_Field = np.sqrt((fB_coil_L[r_idx,z_idx])**2 + (fB_coil_T[r_idx,z_idx]*np.cos(phi))**2)
          Signal = B_Field*np.sin(self.fPulseEff*B_Field*np.pi/2.0)
          avg_z += z*Signal
        avg_z /= float(NSpins)
        self.fProbeCenter[2] = avg_z

    def ForceSpectrum(self, tWeight, tFreq):
        self.fWeightFunction = tWeight
        self.fFreqBins = tFreq

    def GenerateSpins(self, NSpins):
        rng = np.RandomState()

        self.fSpinFreq = np.zeros(NSpins)
        self.fSpinSignal = np.zeros(NSpins)

        rand_r = np.sqrt(rng.uniform(0,1, size=NSpins))*self.fSampleR
        rand_phi = 2*np.pi*rng.uniform(0,1, size=NSpins)
        rand_z = self.fSampleL*rng.uniform(0,1, size=NSpins)-self.fSampleL/2.
        X = rand_r*np.cos(rand_phi)
        Y = rand_r*np.sin(rand_phi)
        ZRel = rand_z - self.fProbeCenter[2]
        self.fSpinFreq = self.fBFieldShape[0] + self.fBFieldShape[1]*X + self.fBFieldShape[2]*Y+ self.fBFieldShape[3]*ZRel + self.fBFieldShape[4]*X*X + self.fBFieldShape[5]*Y*Y+ self.fBFieldShape[6]*ZRel*ZRel;
        idx = np.floor(rand_r/self.fGridSize)*self.fSampleDimL+np.floor((self.fSampleL/2 + rand_z)/self.fGridSize)
        B_Field = np.sqrt((self.fB_coil_L[idx])**2 + (self.fB_coil_T[idx]*cos(rand_phi))**2)
        # //Signal strength should be proportional to B*sin(B). Pi/2 pulse efficiency and the induced signal amplitude. B is normalized to the center B Field, and we assume that for the center B Field the pi/2 pulse length is perfect
        self.fSpinSignal = B_Field*np.sin(self.fPulseEff*B_Field*np.pi/2.0)

        FreqMin = np.min(self.fSpinFreq)
        FreqMax = np.max(self.fSpinFreq)
        df = (FreqMax-FreqMin)/float(self.fNFreq)

        self.fWeightFunction = np.zeros(self.fNFreq)

        index = np.floor((self.fSpinFreq-FreqMin)/df);
        index[index>=self.fNFreq] = self.fNFreq-1
        for i, idx in enumerate(index):
          self.fWeightFunction[idx] += self.fSpinSignal[i]

        self.fFreqBins = FreqMin+df/2.0+df*np.arange(0, self.fNFreq)
        self.fAverageFrequency = np.sum(self.fFreqBins*self.fWeightFunction) / np.sum(self.fWeightFunction)
        # Normalize
        self.fWeightFunction /= np.sum(self.fWeightFunction)

    def GenerateFid(self):
        self.FidWf = np.zeros(self.fFidSamples)
        self.FidTime = np.zeros(self.fFidSamples)

        for f, w in zip(self.fFreqBins, self.fWeightFunction):
            for j in range(fPreSamples, fFidSamples):
    	        t = j * self.fSamplingPeriod + self.fT0
    	        self.FidWf[j] += np.cos(2 * np.pi * f * t) * exp(-t / self.fT2) * w
        for j in range(0, fFidSamples):
            self.FidTime[j] = j*self.fSamplingPeriod + self.fT0

    def GenerateEnvPhase(self):
        DistC = np.zeros(self.fFidSamples)
        DistS = np.zeros(self.fFidSamples)

        f0 = self.fBFieldShape[0]

        t = np.arange(0, fFidSamples)*self.fSamplingPeriod + self.fT0
        for f, w in zip(fFreqBins, fWeightFunction):
        	DistC += np.cos(2 * np.pi * (f-self.f0) * t) * np.exp(-t / fT2) * w;
        	DistS += np.sin(2 * np.pi * (f-self.f0) * t) * np.exp(-t / fT2) * w;

        self.fEnv = np.sqrt(DistC**2 + DistS**2)
        self.fPhase = np.arctan(DistS/DistC)

    def _CalculateCoilBField(self):
        # Calculate B field generated by 1 loop

        # Make it a little bit longer so that we don't go out of range when summing
        AuxL = (self.fCoilL+self.fSampleL)*1.01
        AuxN = np.floor(AuxL/self.fGridSize)+1

        B_loop_L = np.zeros(AuxN*fSampleDimT)
        B_loop_T = np.zeros(AuxN*fSampleDimT)

        for i in range(0,self.fSampleDimT):
          for j in range(0, AuxN/2+1):
        	  int index = i*AuxN+j

              dPhi = 2*np.pi/ float(self.fCoilPhiNSeg)
        	  dl = dPhi*self.fCoilR


        	  Phi = dPhi*np.arange(0, self.fCoilPhiNSeg)
        	  dlX = -dl*np.sin(Phi)
        	  dlY = dl*np.cos(Phi)

        	  Z = -AuxL/2.0+j*self.fGridSize
        	  Y = i*self.fGridSize-self.fCoilR*sin(Phi)
        	  X = -self.fCoilR*cos(Phi)

        	  R = np.sqrt(X**2+Y**2+Z**2)

        	  B_loop_T[index] += np.sum((-Z*dlX)/R**3)
        	  B_loop_L[index] += np.sum((Y*dlX-X*dlY)/R**3)
        	  if j < AuxN/2:
        	      B_loop_T[(i+1)*AuxN-1-j] = -B_loop_T[index]
        	      B_loop_L[(i+1)*AuxN-1-j] = B_loop_L[index]

        # Adding contributions from coils
        CoilShift = 0
        if (fCoilN>1):
          CoilShift = self.fCoilL/(self.fCoilN-1)

        for k in range(0, self.fCoilN):
          Offset0 = np.floor((AuxL+self.fCoilL-self.fSampleL)/self.fGridSize/2.0)
          for i in range(0, self.fSampleDimT):
        	for j in range(0, self.fSampleDimL):
        	  index = i*self.fSampleDimL+j
        	  index_loop = 0
        	  if (self.CoilOption==0):
        	    index_loop = i*AuxN + ((j + Offset0)*self.fGridSize - k* CoilShift)/self.fGridSize
              else:
        	    index_loop = i*AuxN + ((j + Offset0)*self.fGridSize - self.fCoilShiftConfig[k])/self.fGridSize

        	  self.fB_coil_T[index] += B_loop_T[index_loop]
        	  self.fB_coil_L[index] += B_loop_L[index_loop]
        	  if k==0:
        	    self.fPosY[index] = i*self.fGridSize
        	    self.fPosZ[index] = -self.fSampleL/2.0+j*self.fGridSize

        # Normalize
        B_Center_L = self.fB_coil_L[self.fSampleDimT/2*self.fSampleDimL+self.fSampleDimL/2]
        for i in range(0, self.fSampleDimL*self.fSampleDimT):
          self.fB_coil_T[i]/=B_Center_L
          self.fB_coil_L[i]/=B_Center_L

    def GetCoilBField(self):
        return self.fB_coil_T, self.fB_coil_L, self.fPosZ, self.fPosY

    def GetFreqDistribution(self):
        return self.fWeightFunction ,self.fFreqBins

    def GetFid(self):
        return self.FidWf

    def GetFidTime(self):
        return self.FidTime

    def GetEnvPhase(self, Env, Phase):
        return self.fEnv, self.fPhase

    def GetAverageFrequency(self):
        return self.fAverageFrequency

    def GetFidSamplingPeriod(self):
        return self.fSamplingPeriod

    def GetFidSampleNumber(self):
        return self.fFidSamples

    def GetProbeCenter(self):
        return self.fProbeCenter
