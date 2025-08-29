export CUDA_VISIBLE_DEVICES=0
train_data='./data/coco_train/'
upper_bound=10
threshold_type=mean
fixed_threshold_value=8
lr=2e-05
bs=128
cd ..

for mar_weight in 0.2
do
    for neg_weight in 0.2
    do
        if [ "$threshold_type" == "fixed" ]; then
            output_name=coco_hn_neg$neg_weight-atr$mar_weight-fixed$fixed_threshold_value-$lr
        else
            output_name=coco_hn_neg$neg_weight-mar$mar_weight-mean-ub$upper_bound-$lr-test_single
        fi
        output_file=./Outputs/$output_name

        if [[ -d "$output_file" ]];then
            echo "$output_name already exists"
        else
            echo "running $output_name"
            python main.py \
            --wandb-project-name open_clip \
            --train-data $train_data \
            --seed 42 \
            --dataset-type npy \
            --save-frequency 1 \
            --report-to wandb \
            --warmup 50 \
            --batch-size $bs \
            --lr $lr \
            --wd 0.1 \
            --epochs 10 \
            --workers 0 \
            --pretrained openai \
            --model ViT-B-32 \
            --logs Outputs \
            --beta1 0.9 \
            --precision amp \
            --beta2 0.98 \
            --eps 1e-06 \
            --log-every-n-steps 10 \
            --neg-loss \
            --mar-loss \
            --hardnegative \
            --threshold-type $threshold_type \
            --fixed-threshold-value $fixed_threshold_value \
            --mar-loss-weight $mar_weight \
            --neg-loss-weight $neg_weight \
            --positive-margin-loss \
            --upper-bound $upper_bound \
            --name $output_name

            if [ $? -ne 0 ]; then
                echo "Training failed. Cleaning up..."
                rm -rf $output_file
            fi
        fi
    done
done
