# Tau_L1
Codes and documentation for L1Tau.

Authors: Orion Foo, Duc Hoang, Phil Harris. (MIT)

## Conda environment

Create conda environment:

```
conda-env create -f environment.yml
```

Activate the environment:

```
conda activate tau_training
```

And then do whatever you want in this environment (edit files, open notebooks, etc.). To deactivate the environment:

```
conda deactivate
```

If you make any update for the environment, please edit the `environment.yml` file and run:

```
conda env update --file environment.yml  --prune
```

Reference on conda environment here: https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html

## Running jupyter notebook on submit machine at MIT

You could run the notebook on the submit machine by ssh tunneling, first you need to ssh into the machine using: 

```
ssh -L 127.0.0.1:8722:127.0.0.1:8722 <username>@submit03.mit.edu
```

Then, `cd` to the directory of your choice:

```
conda activate tau_training
```

You could then open the notebook:

```
jupyter notebook --no-browser --port 8722 --ip 127.0.0.1
```

It would produce a link and then you can copy the link and paste it to your local browser to open the notebook!

## Related Resources (latest addition on top)

* CDS publication: 
* Public Twiki (v2): https://twiki.cern.ch/twiki/bin/view/CMSPublic/L1Phase2NNPuppiTau
* Orion's talk in Tau-Jets-MET: https://indico.cern.ch/event/1321399/#17-phase-2-nn-puppi-taus
* DP Note Tau ID v2 talk: https://indico.cern.ch/event/1298326/#14-dp-note-nn-puppi-taus
  * https://docs.google.com/presentation/d/12offcwPbynGl94C5zrsw39owu82U9SG1xQRz2dTqPkw/edit#slide=id.g278ac055907_0_61   
* Duc's retraining Tau talk at L1 General Meeting: https://indico.cern.ch/event/1271855/#202-performance-report-algorit
* Official Document on Rates: https://indico.cern.ch/event/1257943/contributions/5343040/attachments/2633523/4567965/AlgorithmsAndPerformance_2023_4.pdf
* TauID @ L1 Talk (Phil): https://www.dropbox.com/s/84jbya301jtac8l/PCH_Tau_25_03.pdf?dl=0
