lrs=(0.0001 0.001)
epochs=(100 200 1000)
topks=(2 3 4 5)

for epoch in "${epochs[@]}"
do
  for lr in "${lrs[@]}"
  do
    for topk in "${topks[@]}"
    do
      exp_id=lr${lr}_epoch${epoch}_perseg_topk${topk}
      nohup python ft_sam.py test with topk=${topk} epochs=${epoch} lr=${lr} split=0 dataset=perseg configs/coco_sam_vit.yml exp_id=${exp_id} > images/perseg/${exp_id}.log
    done
  done
done