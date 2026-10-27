"""Fine-tune RF-DETR Nano on the pseudo-labelled person dataset.

Cosine LR annealing (rfdetr's default "step" schedule drops the LR only at epoch 100, so a 60-80
epoch run would never decay). An earlier 60-epoch step-LR run reached the same test AP (0.724 vs
0.725, paired bootstrap p = 0.58), so only this cosine run is kept.

    python research/train_rfdetr.py                                        # aa1 (published run)
    python research/train_rfdetr.py --data datasets/street/coco --out runs/street_rfdetr \
        --weights runs/rfdetr_nano_v2/checkpoint_best_total.pth --epochs 50   # continue from the aa1 model
"""
import argparse

from rfdetr import RFDETRNano

import paths as P

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/coco")
    ap.add_argument("--out", default=P.RFDETR_FT_RUN)
    ap.add_argument("--weights", default=P.RFDETR_PRETRAINED)
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    model = RFDETRNano(pretrain_weights=a.weights)
    model.train(
        dataset_dir=a.data,
        output_dir=a.out,
        resolution=640,        # up from 384: most pedestrians are small in 1280x720 frames
        epochs=a.epochs,
        batch_size=4,
        grad_accum_steps=4,    # effective batch 16
        lr=1e-4,
        lr_encoder=1.5e-4,
        lr_scheduler="cosine",
        lr_scheduler_kwargs={"min_factor": 0.05},
        warmup_epochs=1,
        early_stopping=True,
        early_stopping_patience=20,
        checkpoint_interval=1000,  # skip the 467 MB periodic checkpoints; best + last are still saved
        seed=0,
        num_workers=4,
        run_test=True,
        tensorboard=True,
        progress_bar=None,
    )
