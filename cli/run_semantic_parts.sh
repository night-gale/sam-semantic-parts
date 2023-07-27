lrs=(0.0001 0.001 0.01)
epochs=(500)
threshold=(0.05 0.03 0.02 0.1)

for epoch in "${epochs[@]}"
do
  for lr in "${lrs[@]}"
  do
    for thres in "${threshold[@]}"
    do
      exp_id=lr${lr}_epoch${epoch}_coco_thres${thres}
      nohup python ft_sam.py test with threshold=${thres} epochs=${epoch} lr=${lr} split=0 configs/coco_sam_vit.yml exp_id=${exp_id} > images/perseg/${exp_id}.log
    done
  done
done