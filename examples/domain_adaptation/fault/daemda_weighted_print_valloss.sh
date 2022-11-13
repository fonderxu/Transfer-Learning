

## 1
CUDA_VISIBLE_DEVICES=0 python daemda_weighted_print_valloss.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/val_loss/jnuab
