# CKASNT

This work is accepted by ICASSP2026



## preparatory work

* Create a new folder named "experiments"
* Download the "vgg_normalised.pth" file from the following address and place it in the "experiments" folder.
* Download the "*_160000.pth" files from the following address and place it in the "experiments" folder. (this is for testing)

通过网盘分享的文件：ICASSP
链接: https://pan.baidu.com/s/17cHl-wDszGp39ZVdmxKuxQ 提取码: 6xu8

## train

Directly use model training

* Place the "content" image in "dataset/train/train2014"
* Place the "style" image in "dataset/train/wikiart"
* then run the shell:

```shell
sh train.sh
```

## test

Directly use model inference

* Place the "content" image in "dataset/test/content"
* Place the "style" image in "dataset/test/style"
* then run the shell:

```shell
sh test.sh
```



### Thank you for the open-source code provided by [ConvKAN](https://github.com/IvanDrokin/torch-conv-kan.git).
