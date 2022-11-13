

# Adda
CUDA_VISIBLE_DEVICES=0 python Total_params.py D:/repository/pycharm/datasets/cwru -d Cwru -s A -t B -a alexnet --scratch -b 220 -i 100 --epochs 20 -lr 0.0001 --weight-decay 1e-4 --seed 0 -p 50 --log logs/Total_params/daemda


#    from torchstat import stat
#    stat(classifier, (1, 1, 1024))
#    stat(domain_discri, (1, 1, classifier.features_dim))
