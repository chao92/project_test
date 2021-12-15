#!/bin/bash
#INPUT=/home/chao/Documents/Pytorch/DAGPA/results/cora/baseline/train_percent_0.8_corrected_test_ID_res.csv
#OLDIFS=$IFS
#IFS=','
#declare -i cn=0
#[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
#sed 1d $INPUT | while read -r id pred_class
#do
#	echo "ID : $id"
#	echo "class : $pred_class"
##	if [[ $pred_class -ne 0 ]]; then
##	  echo "attack"
##	  python target_attack_test.py --dataset_name 'cora' --attack_graph 'False' --node_idx $id --desired_class 0 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 10
##	fi
#	if [[ $pred_class -ne 1 ]]; then
#	  echo "attack"
#	  python target_attack_test.py --dataset_name 'cora' --attack_graph 'False' --node_idx $id --desired_class 1 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 10
#	fi
#	if [[ $pred_class -ne 2 ]]; then
#	  echo "attack"
#	  python target_attack_test.py --dataset_name 'cora' --attack_graph 'False' --node_idx $id --desired_class 2 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 10
#	fi
#	if [[ $pred_class -ne 3 ]]; then
#	  echo "attack"
#	  python target_attack_test.py --dataset_name 'cora' --attack_graph 'False' --node_idx $id --desired_class 3 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 10
#	fi
#	if [[ $pred_class -ne 4 ]]; then
#	  echo "attack"
#	  python target_attack_test.py --dataset_name 'cora' --attack_graph 'False' --node_idx $id --desired_class 4 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 10
#	fi
#	if [[ $pred_class -ne 5 ]]; then
#	  echo "attack"
#	  python target_attack_test.py --dataset_name 'cora' --attack_graph 'False' --node_idx $id --desired_class 5 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 10
#	fi
#	if [[ $pred_class -ne 6 ]]; then
#	  echo "attack"
#	  python target_attack_test.py --dataset_name 'cora' --attack_graph 'False' --node_idx $id --desired_class 6 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 10
#	fi
#done < $INPUT
#IFS=$OLDIFS

# Citeseer

#python baseline.py --dataset_name 'citeseer' --train_percent 0.8 --seed 0 --train_epochs 300

INPUT=/home/chao/Documents/Pytorch/DAGPA/results/citeseer/baseline/train_percent_0.8_corrected_test_ID_res.csv
OLDIFS=$IFS
IFS=','
declare -i cn=0
[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
sed 1d $INPUT | while read -r id pred_class
do
	echo "ID : $id"
	echo "class : $pred_class"
#	if [[ $pred_class -ne 0 ]]; then
#	  echo "attack"
#	  python target_attack_test.py --dataset_name 'citeseer' --attack_graph 'False' --node_idx $id --desired_class 0 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 20
#	fi
	if [[ $pred_class -ne 1 ]]; then
	  echo "attack"
	  python target_attack_test.py --dataset_name 'citeseer' --attack_graph 'False' --node_idx $id --desired_class 1 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 20
	fi
	if [[ $pred_class -ne 2 ]]; then
	  echo "attack"
	  python target_attack_test.py --dataset_name 'citeseer' --attack_graph 'False' --node_idx $id --desired_class 2 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 20
	fi
	if [[ $pred_class -ne 3 ]]; then
	  echo "attack"
	  python target_attack_test.py --dataset_name 'citeseer' --attack_graph 'False' --node_idx $id --desired_class 3 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 20
	fi
	if [[ $pred_class -ne 4 ]]; then
	  echo "attack"
	  python target_attack_test.py --dataset_name 'citeseer' --attack_graph 'False' --node_idx $id --desired_class 4 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 20
	fi
	if [[ $pred_class -ne 5 ]]; then
	  echo "attack"
	  python target_attack_test.py --dataset_name 'citeseer' --attack_graph 'False' --node_idx $id --desired_class 5 --feature_attack 'True' --fix_sparsity 'False' --train_percent 0.8 --sparsity 0.5 --feat_sparsity 0.5 --seed 0 --train_epochs 300 --added_node_num 20
	fi
done < $INPUT
IFS=$OLDIFS
