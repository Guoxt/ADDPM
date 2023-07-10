python train_preseg_network.py --batch_size 12 --lr 0.001 
python train_denoising_network.py --batch_size 12 --lr 0.001 --T 250
MODEL_FLAGS="--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma False --use_scale_shift_norm False --attention_resolutions 16" && DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False“ && TRAIN_FLAGS="--lr 1e-4 --batch_size 4“ && python train_addpm.py --data_dir .../train $TRAIN_FLAGS $MODEL_FLAGS $DIFFUSION_FLAGS
