declare -a tasks=('Length' 'WordContent' 'Depth' 'TopConstituents'
                        'BigramShift' 'Tense' 'SubjNumber' 'ObjNumber'
                        'OddManOut' 'CoordinationInversion')
declare -a methods=('shap' 'catboost')


for t in ${tasks[@]}
	do
	python method1_vs_method2_scatterplot.py --fi1_path outlier_dimensions/roberta-base/results/dist_from_mean_weights.npy --fi1_name outliers --fi2_path logreg/$t.npy --fi2_name logreg --output output/logreg/$t.png --task $t

	for m in ${methods[@]}
		do
		python method1_vs_method2_scatterplot.py --fi1_path outlier_dimensions/roberta-base/results/dist_from_mean_weights.npy --fi1_name outliers --fi2_path $m/$t.npy --fi2_name $m --output output/$m/$t.png --task $t --logy
	done
done
