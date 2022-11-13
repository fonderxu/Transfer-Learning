##Normalization Experiment
#!/usr/bin/env bash
## Cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru
#
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox
#
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i0.00 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/jnu

##Ablation Experiment
#!/usr/bin/env bash
## Cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 0. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off 0. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 0. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/jnu


##Ablation Experiment : trade-off between em and adv : 0.5
#!/usr/bin/env bash
## Cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/cwru
#
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 0.5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off 0.5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off 0.5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 0.5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off 0.5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 0.5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/gearbox
#
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 0.5 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off 0.5 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 0.5 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.5/jnu

#Ablation Experiment : trade-off between em and adv: 0.1
#!/usr/bin/env bash
## Cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/cwru
#
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 0.1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off 0.1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off 0.1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 0.1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off 0.1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 0.1 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/gearbox
#
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 0.1 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off 0.1 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 0.1 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/0.1/jnu

##Ablation Experiment : trade-off between em and adv: 5
#!/usr/bin/env bash
## Cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/cwru
#
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off 5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off 5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off 5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 5 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/gearbox
#
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 5 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off 5 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 5 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/5/jnu


##Ablation Experiment : trade-off between em and adv: 10
#!/usr/bin/env bash
## Cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 10 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 10 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 10 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 10 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 10 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 10 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/cwru
#
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 10 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off 10 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off 10 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 10 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off 10 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 10 -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/gearbox
#
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 10 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off 10 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 10 -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/10/jnu


##Label Shift Experiment
#!/usr/bin/env bash
## daemda
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted_without_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t Sampled_B_06 -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/label_shift/daemda_weighted_without_weighted_cwru-A-Sampled_B_06

##dmdd_adam
#CUDA_VISIBLE_DEVICES=0 python dmdd_adam.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t Sampled_B_06 -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/label_shift/dmdd_adam_cwru-A-Sampled_B_06
#
##cdan
#CUDA_VISIBLE_DEVICES=0 python cdan.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t Sampled_B_06 -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/label_shift/cdan_cwru-A-Sampled_B_06

# daemda
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t Sampled_B_06 -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/label_shift/daemda_weighted_0._cwru-A-Sampled_B_06

##dmdd_adam
#CUDA_VISIBLE_DEVICES=0 python dmdd_adam.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t Sampled_B_06 -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/label_shift/dmdd_adam_cwru-A-Sampled_B_06

##cdan
#CUDA_VISIBLE_DEVICES=0 python cdan.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t Sampled_B_06 -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/label_shift/cdan_cwru-A-Sampled_B_06

##Visualization
#!/usr/bin/env bash
## Cwru
##daemda
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru --phase 'analysis'
#
##dmdd_adam
#CUDA_VISIBLE_DEVICES=0 python dmdd_adam.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/dmdd_adam/cwru --phase 'analysis'
#
##cdan
#CUDA_VISIBLE_DEVICES=0 python cdan.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/cdan/cwru --phase 'analysis'

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



#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/cwru --phase 'analysis'
#
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/gearbox --phase 'analysis'
#
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/jnu --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/jnu --phase 'analysis'
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i0.00 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/normalization/jnu --phase 'analysis'


## optimizer: SGD lr 0.001 weight_decay: 1e-4 --trade-off: 1.0
##!/usr/bin/env bash
## Cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 50 -lr 0.001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/jnu


## DANN epoch: 20 -i: 100 lr: 0.0001  weight-decay: 1e-4 optimizer: Adam
##!/usr/bin/env bash
## Cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/cwru
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 1. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/gearbox
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/jnu


## Experiment Details: Ablation .DANN epoch: 20 -i: 100 lr: 0.0001  weight-decay: 1e-4 optimizer: Adam
##!/usr/bin/env bash
## Cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/cwru
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch --trade-off 0. -b 205 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/gearbox
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch --trade-off 0. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch --trade-off 0. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/jnu
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch --trade-off 0. -b 220 -i 200 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/ablation/jnu
#
## Experiment Details: Ablation .DANN epoch: 20 -i: 100 lr: 0.0001  weight-decay: 1e-4 optimizer: Adam
##!/usr/bin/env bash
## Cwru
### 0.0001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0.0001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.0001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0.0001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.0001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0.0001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.0001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0.0001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.0001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0.0001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.0001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0.0001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.0001
### 0.001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0.001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0.001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0.001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0.001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0.001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.001
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0.001 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.001
### 0.01
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0.01 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.01
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0.01 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.01
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0.01 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.01
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0.01 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.01
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0.01 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.01
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0.01 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.01
### 0.1
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.1
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.1
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.1
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.1
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.1
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0.1 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.1
### 0.3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0.3 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0.3 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0.3 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0.3 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0.3 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0.3 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.3
### 0.5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0.5 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.5
### 0.8
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0.8 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.8
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 0.8 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.8
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 0.8 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.8
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 0.8 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.8
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 0.8 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.8
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 0.8 -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru0.8
### 1
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru1.0
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru1.0
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru1.0
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru1.0
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru1.0
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru1.0
### 3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 3. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 3. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 3. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 3. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 3. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru3
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 3. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru3
### 5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 5. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch --trade-off 5. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch --trade-off 5. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch --trade-off 5. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch --trade-off 5. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru5
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch --trade-off 5. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/parameter_sensitivity/cwru5

# Visualization
## cwru 1.0
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/visualization/cwru_A_B_1.0
CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/visualization/cwru_A_B_1.0 --phase 'analysis'

## cwru 0.0
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/visualization/cwru_A_B_0.0
CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch --trade-off 0. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/visualization/cwru_A_B_0.0 --phase 'analysis'


## Model Robustness on label shift
#CUDA_VISIBLE_DEVICES=0 python daemda_weighted.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t Sampled_B_06 -a alexnet --scratch --trade-off 1. -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/daemda_weighted/robustness/cwruAtoSampled_B_06





e