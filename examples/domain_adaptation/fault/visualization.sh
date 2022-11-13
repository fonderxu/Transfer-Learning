##Visualization
#!/usr/bin/env bash
# Cwru
#daemda
CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru --phase 'analysis'

##dmdd_adam
#CUDA_VISIBLE_DEVICES=0 python dmdd_adam.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dmdd_adam/cwru
#CUDA_VISIBLE_DEVICES=0 python dmdd_adam.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dmdd_adam/cwru --phase 'analysis'
#
##cdan
#CUDA_VISIBLE_DEVICES=0 python cdan.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/cdan/cwru
#CUDA_VISIBLE_DEVICES=0 python cdan.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/cdan/cwru --phase 'analysis'
#
#!/usr/bin/env bash
## Jnu
##daemda
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/jnu --phase 'analysis'
#
##dmdd_adam
#CUDA_VISIBLE_DEVICES=0 python dmdd_adam.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dmdd_adam/jnu --phase 'analysis'
#
##cdan
#CUDA_VISIBLE_DEVICES=0 python cdan.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/cdan/jnu --phase 'analysis'



