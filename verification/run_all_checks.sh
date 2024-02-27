# Check if the environment is activated
if [[ $CONDA_DEFAULT_ENV != "tau_training" ]]; then
    echo "tau_training environment is not activated. Please activate it first by running 'conda activate tau_training'."
    exit 1
fi

python plot_pt_regression.py
python plot_signal_efficiency.py