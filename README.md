


```
conda create --name DAGPA python=3.7.10
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.6 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
pip install torch-sparse==0.6.9 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
pip install torch-geometric==1.7.0
pip install torch-cluster==1.5.9 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html
pip install torch-spline-conv==1.2.1 -f https://data.pyg.org/whl/torch-1.6.0%2Bcpu.html

```

```buildoutcfg
./dataset: preprocessing the graph datasets
./attack: define the DAGPA model
attack.py: steps for executing the DAGPA attack
```
###detecting outlying objects
```buildoutcfg
conda install -c conda-forge pyod
```

