
checkpoint_path=$1
torchrun --nproc_per_node=1 \
    --master_port=34653 \
    train.py \
    --cfg-path lavis/configs/config.yaml \
    --options \
    model.arch VideoChat2 \
    model.model_type vicuna7b \
    model.load_finetuned False \
    model.load_pretrained True \
    model.num_query_token 32 \
    model.vit_precision fp16 \
    model.freeze_vit True \
    model.memory_bank_length 16 \
    model.num_videos 16 \
    datasets.lvu_cls.history 100 \
    datasets.lvu_cls.task relationship \
    datasets.lvu_cls.stride 10 \
    run.init_lr 1e-4 \
    run.max_epoch 20 \
    run.num_beams 5 \
    run.batch_size_train 4 \
    run.batch_size_eval 4 \
    run.accum_grad_iters 1 \
    run.num_workers 6 \
    run.seed 42 \
    run.evaluate True \
    run.valid_splits "['test']" \
    run.report_metric True \
    run.prefix test \
    run.resume_ckpt_path ${checkpoint_path}

