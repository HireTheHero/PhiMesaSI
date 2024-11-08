source ~/.bashrc
conda activate si
python -m wandb login
# if parameter is given, use it as --output-dir
# if not, use default value
if [ $# -eq 0 ]; then
    OUTPUT_DIR="experiment/SI"
else
    OUTPUT_DIR=$1
fi
# python script/main.py --config-path settings/train.yaml --log-wandb --wandb-project mesa --output-dir $OUTPUT_DIR
python script/main.py --config-path settings/train.yaml --is-batch --log-wandb --wandb-project mesa --output-dir $OUTPUT_DIR