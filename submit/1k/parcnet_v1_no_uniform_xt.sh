MODEL=parcnet_v1_no_uniform_xt
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 main.py \
	--model ${MODEL} --drop_path 0.1 \
	--batch_size 128 --lr 4e-3 --update_freq 8 \
	--model_ema true --model_ema_eval true \
	--data_path datasets/imagenet1k \
        --output_dir output/1k/${MODEL} \
	> log/1k/${MODEL}.log 2>&1
