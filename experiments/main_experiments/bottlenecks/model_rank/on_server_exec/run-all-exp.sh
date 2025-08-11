#!/bin/sh 
for i in $(seq 1 5);
do
	sh run_exp.sh bottleneck_analysis-model-resnet18-items-96-split-None-dataset_type-imagenette 
	sh run_exp.sh bottleneck_analysis-model-resnet18-items-1024-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-resnet18-items-9216-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-resnet18-items-1024-split--3-dataset_type-imagenette_preprocessed_ssd
	sh run_exp.sh bottleneck_analysis-model-resnet152-items-96-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-resnet152-items-1024-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-resnet152-items-9216-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-resnet152-items-1024-split--3-dataset_type-imagenette_preprocessed_ssd
	sh run_exp.sh bottleneck_analysis-model-eff_net_v2_l-items-96-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-eff_net_v2_l-items-1024-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-eff_net_v2_l-items-9216-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-eff_net_v2_l-items-1024-split--3-dataset_type-imagenette_preprocessed_ssd
	sh run_exp.sh bottleneck_analysis-model-vit_l_32-items-96-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-vit_l_32-items-1024-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-vit_l_32-items-9216-split-None-dataset_type-imagenette
	sh run_exp.sh bottleneck_analysis-model-vit_l_32-items-1024-split--3-dataset_type-imagenette_preprocessed_ssd
done