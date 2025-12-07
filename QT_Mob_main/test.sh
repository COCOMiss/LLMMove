#!/bin/bash

for do_sample in True False; do
    if [ "$do_sample" == "True" ]; then
        for temperature in 0.3 0.5 0.7 0. 9; do
            for max_new_tokens in 512 1024; do
                # 创建输出目录
                output_dir="QT_Mob_main/results/sample_${temperature}_${max_new_tokens}"
                mkdir -p "$output_dir"
                
                python QT_Mob_main/sft_pipeline_xi.py \
                    --do_sample "$do_sample" \
                    --temperature "$temperature" \
                    --max_new_tokens "$max_new_tokens" \
                    --results_file "${output_dir}/metrics.json" \
                    --prediction_file "${output_dir}/prediction.json" \
                    --ground_truth_file "${output_dir}/ground_truth.json"
            done
        done
    else
        for num_beams in 3 5; do
            for max_new_tokens in 512 1024; do
                # 创建输出目录
                output_dir="QT_Mob_main/results/beam_${num_beams}_${max_new_tokens}"
                mkdir -p "$output_dir"
                
                python QT_Mob_main/sft_pipeline_xi.py \
                    --do_sample "$do_sample" \
                    --num_beams "$num_beams" \
                    --max_new_tokens "$max_new_tokens" \
                    --results_file "${output_dir}/metrics. json" \
                    --prediction_file "${output_dir}/prediction.json" \
                    --ground_truth_file "${output_dir}/ground_truth.json"
            done
        done
    fi
done