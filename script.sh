python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51/" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"

python test_two_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
--resume_path2 "trained_models/HMDB51/MARS_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51/" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"