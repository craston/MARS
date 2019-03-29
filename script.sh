# python test_parallel.py --batch_size 1 --n_classes 400 --model resnext --model_depth 101 --resize 256 \
# --log 0 --dataset Kinetics --modality RGB --sample_duration 64 --split val  \
# --resume_path "trained_models/Kinetics/RGB_Kinetics_64f.pth"

# python test_parallel.py --batch_size 1 --n_classes 400 --model resnext --model_depth 101 --resize 256 \
# --log 0 --dataset Kinetics --modality RGB --sample_duration 16 --split val  \
# --resume_path "trained_models/Kinetics/RGB_Kinetics_16f.pth"

python test_parallel.py --batch_size 1 --n_classes 400 --model resnext --model_depth 101 --resize 256 \
--log 0 --dataset Kinetics --modality Flow --sample_duration 64 --split val \
--resume_path "trained_models/Kinetics/Flow_Kinetics_64f.pth"

python test_parallel.py --batch_size 1 --n_classes 400 --model resnext --model_depth 101 --resize 256 \
--log 0 --dataset Kinetics --modality Flow --sample_duration 16 --split val \
--resume_path "trained_models/Kinetics/Flow_Kinetics_16f.pth"

# python test_parallel.py --batch_size 1 --n_classes 400 --model resnext --model_depth 101 --resize 256 \
# --log 0 --dataset Kinetics --modality RGB --sample_duration 64 --split val \
# --resume_path "trained_models/Kinetics/MARS_Kinetics_64f.pth"

# python test_parallel.py --batch_size 1 --n_classes 400 --model resnext --model_depth 101 --resize 256 \
# --log 0 --dataset Kinetics --modality RGB --sample_duration 16 --split val \
# --resume_path "trained_models/Kinetics/MARS_Kinetics_16f.pth"

# python test_parallel.py --batch_size 1 --n_classes 400 --model resnext --model_depth 101 --resize 256 \
# --log 0 --dataset Kinetics --modality RGB --sample_duration 16 --split val \
# --resume_path "trained_models/Kinetics/MERS_Kinetics_16f.pth"


python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 --resize 256 \
--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
--resume_path "trained_models/HMDB51_final/HMDB51_1_RGB_train_batch64_resize256_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx4_varLR100.pth" \
--frame_dir "/home/ncrasto/code/workspace/action-recog-release/dataset/HMDB51/" \
--annotation_path "/nfs/team/cv/Users/ncrasto/video_db/HMDB51/HMDB51_labels" \
--result_path "results/"