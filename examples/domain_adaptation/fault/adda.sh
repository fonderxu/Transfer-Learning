#SGD
##!/usr/bin/env bash
#
## Cwru
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch -b 220 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch -b 220 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch -b 220 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch -b 220 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch -b 220 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch -b 220 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
#
#
## Gearbox
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch -b 205 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch -b 205 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch -b 205 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch -b 205 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch -b 205 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch -b 205 -i 100 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
#
#
## Jnu
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch -b 220 -i 200 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/jnu
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch -b 220 -i 200 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/jnu
#CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch -b 220 -i 200 --epochs 50 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/jnu
#
#


#ADDA epoch: 20 -i: 100 lr: 0.0001  weight-decay: 1e-4 optimizer: Adam
#!/usr/bin/env bash

# Cwru
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch -b 220 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t C -a alexnet --scratch -b 220 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t D -a alexnet --scratch -b 220 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t C -a alexnet --scratch -b 220 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s B -t D -a alexnet --scratch -b 220 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/cwru -d Cwru -s C -t D -a alexnet --scratch -b 220 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/cwru


# Gearbox
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t B  -a alexnet --scratch -b 205 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t C  -a alexnet --scratch -b 205 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s A -t D  -a alexnet --scratch -b 205 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t C  -a alexnet --scratch -b 205 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s B -t D  -a alexnet --scratch -b 205 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/gearbox -d Gearbox -s C -t D  -a alexnet --scratch -b 205 -i 100 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/gearbox


# Jnu
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t B -a alexnet --scratch -b 220 -i 200 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/jnu
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/jnu -d Jnu -s A -t C -a alexnet --scratch -b 220 -i 200 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/jnu
CUDA_VISIBLE_DEVICES=0 python adda.py D:/repository/pycharm/datasets/jnu -d Jnu -s B -t C -a alexnet --scratch -b 220 -i 200 --epochs 20 -lr 0.0001 -lr-d 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/adda/jnu
