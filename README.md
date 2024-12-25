# Using Sketches to Train Occupancy Network

## Installation

First you have to make sure that you have all dependencies in place. The simplest way to do so, is to use anaconda.

Next, compile the extension modules. You can do this via

```
python setup.py build_ext --inplace
```

## Dataset
I have given the data lists I used in abc file. Because the dataset is large, if you want to get it, please feel free to contact me via youzhenguo65@gmail.com.

Preparing the rendering lists via (If you have these lists, you can skip this step)

```
python generate_renderings.py
```

You must prepare points and occupancy values in your data, you can make this via (If you have these, you can skip this step)

```
python occgen.py
```

Then you can check your dataset via

```
python dataset.py
```

## Training

To train a new network from scratch, run

```
python train.py
```

Where there are some parameters you can choose

```
--cuda          Gpu ids
--dataRoot      Data root path
--batchSize     Inputing batch size you want, default 64
--lr            Initial learning rate, default 1e-6
--nepoch        Number of epochs to train for, default 100
--resume        Resuming training from checkpoint
--start_epoch   Starting epoch number for resume
```

## Testing

To test the network, run

```
python test.py
```

Testing will reconstruct a mesh from a single view of a sketch, and you can refine this mesh via

```
--refinement    Number of mesh refinement steps, default 30
```

## Loss

You can view train and validation loss curves via Tensorboard

```
cd log
tensorboard --logdir <log directory path> --port 6006
```

## Issues

If you meet `numpy/arrayobject.h: No such file or directory`,you can solve it via

```
sudo apt-get install python-numpy
```

If you unable to install resnet18, you can copy `resnet18-f37072fd.pth` to `/root/.cache/torch/hub/checkpoints`
