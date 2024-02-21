# CMSSW Performance verification

Do this first (if not enabled already):

```
source /cvmfs/cms.cern.ch/cmsset_default.sh
```

Then do something like this:

```
cmsrel CMSSW_14_0_0_pre3
cd CMSSW_14_0_0_pre3/src/
cmsenv && git cms-init
```

```
cmsenv
git cms-addpkg DataFormats/L1TParticleFlow
git cms-addpkg DataFormats/L1TCorrelator
git cms-addpkg L1Trigger/Phase2L1ParticleFlow
git cms-addpkg L1Trigger/TrackTrigger
git cms-addpkg SimTracker/TrackTriggerAssociation
git cms-addpkg L1Trigger/Phase2L1ParticleFlow
```

```
# scripts
git clone git@github.com:violatingcp/FastPUPPI.git -b 14_0_X

scram b -j8
```

#Useful resources

* Ntupler producer: https://github.com/violatingcp/FastPUPPI/tree/14_0_X/NtupleProducer
* L1T Phase 2 Emulator instructions: https://twiki.cern.ch/twiki/bin/view/CMSPublic/SWGuideL1TPhase2Instructions#CMSSW_14_0_0_pre3