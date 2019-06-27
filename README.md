# MARS: Motion-Augmented RGB Stream for Action Recognition

By Nieves Crasto, Philippe Weinzaepfel, Karteek Alahari and Cordelia Schmid

MARS is a strategy to learn a stream that takes only RGB frames as input
but leverages both appearance and motion information from them. This is
achieved by training a network to minimize the loss between its features and
the Flow stream, along with the cross entropy loss for recognition.
For more details, please refer to our [CVPR 2019 paper](https://hal.inria.fr/hal-02140558/document) and our [website](https://europe.naverlabs.com/Research/Computer-Vision/Video-Analysis/MARS).

We release the testing code along trained models. 

<!-- #### Performance on Kinetics400

|Method 												| Stream   | Pretrain | Acc   |
|-------------------------------------------------------|:--------:|:--------:|:-----:|
|[I3D](https://arxiv.org/pdf/1705.07750.pdf)			| RGB      |ImageNet  | 71.1  |
|[ResNext101](https://arxiv.org/pdf/1711.09577.pdf)	    | RGB      |none      | 65.1  |
|[R(2+1)D](https://arxiv.org/pdf/1711.11248.pdf)	    | RGB      |Sport-1M  | 74.3  |
|[S3D-G](https://arxiv.org/pdf/1712.04851.pdf)		    | RGB      |ImageNet  | 74.7  | 
|[NL-I3D](https://arxiv.org/pdf/1711.07971.pdf)		    | RGB      |ImageNet  |**77.7**|
|**MARS**                   							| RGB      | none     |  72.7 |
|**MARS+RGB**               							| RGB      | none     |  74.8 |
|[I3D](https://arxiv.org/pdf/1705.07750.pdf)			| RGB+Flow | ImageNet |  74.2 | 
|[R(2+1)D](https://arxiv.org/pdf/1711.11248.pdf)		| RGB+Flow | Sports-1M| 75.4  |
|[S3D-G](https://arxiv.org/pdf/1712.04851.pdf)		    | RGB+Flow | ImageNet | 77.2  |
|**MARS+RGB+Flow**		          					    | RGB+Flow |  none    | 74.9  | -->

<!-- #### Performance om HMDB51, UCF101 and SomethingSomethingv1

|Method 				| Streams | Pretrain | UCF101 | HMDB51 | Something Somethingv1|
|-----------------------|:-------:|:--------:|:------:|:------:|:--------------------:|
[TRN](https://arxiv.org/pdf/1711.08496.pdf)                 |RGB      |none      | ---    | ---    | 34.4                 |
[MFNet](https://arxiv.org/pdf/1807.10037.pdf)   			|RGB      |none      | ---    | ---    | 43.9       |
[I3D](https://arxiv.org/pdf/1705.07750.pdf)					|RGB      |ImNet+Kin | 95.6   | 74.8   | ---        |
[ResNext101](https://arxiv.org/pdf/1711.09577.pdf)			|RGB      |Kinetics  | 94.5   | 70.1   | ---    	|
[S3D-G](https://arxiv.org/pdf/1712.04851.pdf)         		|RGB      |ImNet+Kin | 96.8   | 75.9   | 48.2		|
[R(2+1)D](https://arxiv.org/pdf/1711.11248.pdf)       		|RGB      |Kinetics  | 96.8   | 74.5   | ---        |
**MARS**                      								|RGB      |Kinetics  | 97.4   | 79.3   | 48.7		|
**MARS+RGB**                  								|RGB      |Kinetics  |**97.6**|**79.5**|**51.7**	|
[2-stream](https://arxiv.org/pdf/1406.2199.pdf)				|RGB+Flow |ImageNet  | 88.0   | 59.4   | --- 		|
[TRN](https://arxiv.org/pdf/1711.08496.pdf)					|RGB+Flow |none      | ---    | ---    | 42.0		|
[I3D](https://arxiv.org/pdf/1705.07750.pdf)             	|RGB+Flow |ImNet+Kin |**98.0**|**80.7**| ---		|
[R(2+1)D](https://arxiv.org/pdf/1711.11248.pdf)				|RGB+Flow |Kinetics  |  97.3  | 78.7   | --- 		|
[OFF](https://arxiv.org/pdf/1711.11152.pdf)					|RGB+Flow |none      | 96.0   | 74.2   | --- 		|
**MARS+RGB+Flow**            								|RGB+Flow |Kinetics  |**98.1**|**80.9**|**53.0**	| -->

## Citing MARS
```
@inproceedings{crasto2019mars,
  title={{MARS: Motion-Augmented RGB Stream for Action Recognition}},
  author={Crasto, Nieves and Weinzaepfel, Philippe and Alahari, Karteek and Schmid, Cordelia},
  booktitle={CVPR},
  year={2019}
}
```

## Contents
1. [Requirements](#requirements)
2. [Datasets](#datasets)
3. [Models](#models)
4. [Testing](#testing)

## Requirements
* Python3
* [Pytorch 1.0](https://pytorch.org/get-started/locally/)
```
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
* ffmpeg version 3.2.4
* OpenCV with GPU support (will not be providing support in compiling this part)



* Directory tree
 ```
    dataset/
        HMDB51/ 
            ../(dirs of class names)
                ../(dirs of video names)
        HMDB51_labels/
    results/
        test.txt
    trained_models/
        HMDB51/
            ../(.pth files)
```


## Datasets

* The datsets and splits can be downloaded from 

    [Kinetics400](https://deepmind.com/research/open-source/open-source-datasets/kinetics/)

    [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

    [HMDB51](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads)

    [SomethingSomethingv1](https://20bn.com/datasets/something-something/v1)

* To extract only frames from videos 
```
python utils1/extract_frames.py path_to_video_files path_to_extracted_frames start_class end_class
```

* To extract optical flows + frames from videos 
    * Build
    ```
    export OPENCV=path_where_opencv_is_installed

    g++ -std=c++11 tvl1_videoframes.cpp -o tvl1_videoframes -I${OPENCV}include/opencv4/ -L${OPENCV}lib64 -lopencv_objdetect -lopencv_features2d -lopencv_imgproc -lopencv_highgui -lopencv_core -lopencv_imgcodecs -lopencv_cudaoptflow -lopencv_cudaarithm
    
    python utils1/extract_frames_flows.py path_to_video_files path_to_extracted_flows_frames start_class end_class gpu_id
    ```
## Models

Trained models can be found [here](https://drive.google.com/drive/folders/1OVhBnZ_FmqMSj6gw9yyrxJJR8yRINb_G?usp=sharing). The names of the models are in the form of 

```
stream_dataset_frames.pth     

RGB_Kinetics_16f.pth indicates --modality RGB --dataset Kinetics --sample_duration 16
```

For HMDB51 and UCF101, we have only provided trained models for the first split.

## Testing script
For RGB stream:
```
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```

For Flow stream:
```
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality Flow --sample_duration 16 --split 1  \
--resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```

For single stream MARS: 

```
python test_single_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
--resume_path1 "trained_models/HMDB51/MARS_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```

For two streams RGB+MARS:
```
python test_two_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB --sample_duration 16 --split 1 --only_RGB  \
--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
--resume_path2 "trained_models/HMDB51/MARS_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```

For two streams RGB+Flow:
```
python test_two_stream.py --batch_size 1 --n_classes 51 --model resnext --model_depth 101 \
--log 0 --dataset HMDB51 --modality RGB_Flow --sample_duration 16 --split 1 \
--resume_path1 "trained_models/HMDB51/RGB_HMDB51_16f.pth" \
--resume_path2 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
--frame_dir "dataset/HMDB51/HMDB51_frames/" \
--annotation_path "dataset/HMDB51_labels" \
--result_path "results/"
```

## Training script
### For RGB stream: 
#### From scratch:
```
 python train.py --dataset Kinetics --modality RGB --only_RGB \
--n_classes 400 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101  \
--frame_dir "dataset/Kinetics" \
--annotation_path "dataset/Kinetics_labels" \
--result_path "results/"
```

#### From pretrained Kinetics400:
```
 python train.py --dataset HMDB51 --modality RGB --split 1 --only_RGB \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/RGB_Kinetics_16f.pth" \
--result_path "results/"
```

#### From checkpoint:
```
 python train.py --dataset HMDB51 --modality RGB --split 1 --only_RGB \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/RGB_Kinetics_16f.pth" \
--resume_path1 "results/HMDB51/PreKin_HMDB51_1_RGB_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx4_varLR2.pth" \
--result_path "results/"
```

### For Flow stream 
#### From scratch:
```
 python train.py --dataset Kinetics --modality Flow \
--n_classes 400 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101  \
--frame_dir "dataset/Kinetics" \
--annotation_path "dataset/Kinetics_labels" \
--result_path "results/"
```

#### From pretrained Kinetics400:
```
 python train.py --dataset HMDB51 --modality Flow --split 1 \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
--result_path "results/"
```

#### From checkpoint:
```
 python train.py --dataset HMDB51 --modality Flow --split 1 \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 32 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
--resume_path1 "results/HMDB51/PreKin_HMDB51_1_Flow_train_batch32_sample112_clip16_nestFalse_damp0.9_weight_decay1e-05_manualseed1_modelresnext101_ftbeginidx4_varLR2.pth" \
--result_path "results/"
```

### For MARS:
#### From scratch:  
```
python MARS_train.py --dataset Kinetics --modality RGB_Flow \
--n_classes 400 \
--batch_size 16 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 \
--output_layers 'avgpool' --MARS_alpha 50 \
--frame_dir "dataset/Kinetics" \
--annotation_path "dataset/Kinetics_labels" \
--resume_path1 "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
--result_path "results/" --checkpoint 1
```

#### From pretrained Kinetics400:
```
python MARS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 16 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--output_layers 'avgpool' --MARS_alpha 50 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/MARS_Kinetics_16f.pth" \
--resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
--result_path "results/" --checkpoint 1
```
#### From checkpoint:
```
python MARS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 16 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--output_layers 'avgpool' --MARS_alpha 50 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/MARS_Kinetics_16f.pth" \
--resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
--MARS_resume_path "results/HMDB51/MARS_HMDB51_1_train_batch16_sample112_clip16_lr0.001_nesterovFalse_manualseed1_modelresnext101_ftbeginidx4_layeravgpool_alpha50.0_1.pth" \
--result_path "results/" --checkpoint 1
```

### For MERS:
#### From scratch:  
```
python MERS_train.py --dataset Kinetics --modality RGB_Flow \
--n_classes 400 \
--batch_size 16 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 \
--output_layers 'avgpool' --MARS_alpha 50 \
--frame_dir "dataset/Kinetics" \
--annotation_path "dataset/Kinetics_labels" \
--resume_path1 "trained_models/Kinetics/Flow_Kinetics_16f.pth" \
--result_path "results/" --checkpoint 1
```

#### From pretrained Kinetics400:
```
python MERS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 16 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--output_layers 'avgpool' --MARS_alpha 50 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/MERS_Kinetics_16f.pth" \
--resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
--result_path "results/" --checkpoint 1
```
#### From checkpoint:
```
python MERS_train.py --dataset HMDB51 --modality RGB_Flow --split 1  \
--n_classes 400 --n_finetune_classes 51 \
--batch_size 16 --log 1 --sample_duration 16 \
--model resnext --model_depth 101 --ft_begin_index 4 \
--output_layers 'avgpool' --MARS_alpha 50 \
--frame_dir "dataset/HMDB51" \
--annotation_path "dataset/HMDB51_labels" \
--pretrain_path "trained_models/Kinetics/MARS_Kinetics_16f.pth" \
--resume_path1 "trained_models/HMDB51/Flow_HMDB51_16f.pth" \
--MARS_resume_path "results/HMDB51/MERS_HMDB51_1_train_batch16_sample112_clip16_lr0.001_nesterovFalse_manualseed1_modelresnext101_ftbeginidx4_layeravgpool_alpha50.0_1.pth" \
--result_path "results/" --checkpoint 1
```