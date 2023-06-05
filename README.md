# Tau_L1
Codes and documentation for L1Tau

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
