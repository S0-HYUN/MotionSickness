# python main_eegnet.py -sch=step --step_size=1000 --gamma=0.1 --optimizer=SGD --lr=1e-4 --epoch=3000 --wd=1e-3
# python main_eegnet.py -sch=cosine --T_max=1000 --eta_min=0.999 --mode=train --lr=1e-2
# python main_eegnet.py -sch=exp --gamma=0.995 --test_subj=18 --lr=1e-4 --wd=1e-3

for var in $(seq 1 23); #(10 23)
do
    echo $var
    python main_eegnet.py -sch=step --step_size=1000 --gamma=0.1 --optimizer=SGD --lr=1e-4 --epoch=3000 --wd=1e-4 --test_subj $var
    # python main_eegnet.py --test_subj $var -sch=step --step_size=1000 --gamma=0.1 --optimizer=SGD --lr=1e-4 --epoch=3000 --wd=1e-3
    # python main_eegnet.py --test_subj $var -sch=cosine --T_max=1000 --eta_min=0.999 --mode=train --lr=1e-2
    # python main_eegnet.py --test_subj $var -sch=exp --gamma=0.995 --test_subj=18 --lr=1e-4 --wd=1e-3 
done