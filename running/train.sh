python 5fold_dependent.py -sch=step --step_size=1000 --gamma=0.1 --optimizer=SGD --lr=1e-3 --epoch=100 --wd=1e-4 --test_subj=1
# python main_eegnet.py -sch=cosine --T_max=1000 --eta_min=0.999 --mode=train --lr=1e-2
# python main_eegnet.py -sch=exp --gamma=0.995 --test_subj=18 --lr=1e-4 --wd=1e-3
python 5fold_dependent.py -sch=step --step_size=1000 --gamma=0.1 --optimizer=SGD --lr=5e-3 --epoch=2000 --wd=1e-5 --test_subj=18
python main.py -sch=step --step_size=1000 --gamma=0.1 --optimizer=SGD --lr=5e-3 --epoch=2000 --wd=1e-5 --test_subj=18