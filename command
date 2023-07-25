python scripts/image_train.py --data_dir datasets/cifar_train --image_size 64 --num_channels 128 \
--num_res_blocks 3 --diffusion_steps 4000 --noise_schedule linear --lr 1e-4 --batch_size 16

mpiexec -n 3 python scripts/image_train.py --data_dir datasets/cifar_train --image_size 64 --num_channels 128 \
--num_res_blocks 3 --diffusion_steps 4000 --noise_schedule linear --lr 1e-4 --batch_size 16

Namespace(data_dir='/home/chenshang/Desktop/improved-diffusion/datasets/cifar_train', schedule_sampler='uniform', lr=0.0001, weight_decay=0.0, lr_anneal_steps=0, batch_size=16, microbatch=-1, ema_rate='0.9999', log_interval=10, save_interval=10000, resume_checkpoint='', use_fp16=False, fp16_scale_growth=0.001, image_size=64, num_channels=128, num_res_blocks=3, num_heads=4, num_heads_upsample=-1, attention_resolutions='16,8', dropout=0.0, learn_sigma=False, sigma_small=False, class_cond=False, diffusion_steps=4000, noise_schedule='linear', timestep_respacing='', use_kl=False, predict_xstart=False, rescale_timesteps=True, rescale_learned_sigmas=True, use_checkpoint=False, use_scale_shift_norm=True)


python scripts/image_sample.py --model_path /tmp/openai-2023-07-06-16-03-59-751373/ema_0.9999_050000.pt --image_size 64 --num_channels 128 --num_res_blocks 3 --diffusion_steps 4000 --noise_schedule linear

export CUDA_VISIBLE_DEVICES=1,2,3