#################################################################### Global Attack ####################################################################
# Settings : epsion >= 1e-3
# V0 Baseline version
# result save at ./results/baseline
#for ((i=0.1;i<0.99;i+=0.1))
#do
#  for ((j=0;j<10;j+=1))
#  do
#    echo -e "\033[31m train percent:${i} seed ${j}\033[0m"
#    python baseline.py --dataset_name 'cora' --train_percent $i --seed $j --train_epochs 300
#  done
#done




#target attack
#python baseline.py --dataset_name 'cora' --train_percent 0.5 --seed 0 --train_epochs 300
#predicted class in clean graph is 1

#get test_id list
python baseline.py --dataset_name 'cora' --train_percent 0.8 --seed 0 --train_epochs 300
# attack the test_id list
#python target_attack.py --dataset_name 'cora' --attack_graph 'False' --node_idx 0 --desired_class 2 --feature_attack 'True' --train_percent 0.8 --sparsity 0.8 --feat_sparsity 0.9 --seed 0 --train_epochs 300 --added_node_num 2














