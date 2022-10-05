# DeepStreamOS
This is the official code for the paper [DeepStreamOS: Fast open-Set classification for convolutional neural networks](https://www.sciencedirect.com/science/article/pii/S0167865522000186?casa_token=5AfAKScbQmoAAAAA:BldzZgyiLI_LgYHUWQi7jyJEZ4c1oM94X5p0JmifG3dnkDEuhPFTMS50ahdQTmy4HCarhJwLFw)

![sysdescrip_tex](https://user-images.githubusercontent.com/61065458/194042529-4a4cc78b-9b73-4214-a74a-9ebb749c6676.jpg)

DeepStreamOS brings together the use of deep neural network activations with a stream-based outlier detection method for fast identification of instances that belong to unknown classes. DeepStreamOS uses all layers of a CNN to get a trajectory of the activations and applies a stream-based analysis method to determine if an instance belongs to an unknown class. 

## Requirements
* Python 3.7 or higher
* The Java executable, utilises components from the MOA framework (
Bifet, A., Holmes, G., Kirkby, R., Pfahringer, B.: MOA: Massive Online Analysis. Journal
of Machine Learning Research 11(May), 1601{1604 (2010). URL http://www.jmlr.org/papers/v11/bifet10a.html)
and Py4j (Dagenais, B.: Py4j - A Bridge between Python and Java. URL https://www.py4j.org/) to link the Python code with Java code.
Java Runtime is required to run MoaGateway.jar

## Datasets and Deep Neural Networks (DNNs)
* CIFAR-10 and Fashion-MNIST will be downloaded during first run
* DNN will be loaded at first run (Only the MobileNet DNN is available in the demo)

## Parameter Setup and Execution
Execution starts from main.py by specifying start parameters. 
For example the following line will run the experiment for the MobileNet DNN, Fashion-MNIST dataset, unknown class detection using known the classes of 0, 1, 2, 3, 4, 6 and applying unknown classes of 5 and 7:
```
start(dnnName='mobilenet', datasetName='mnistfashion',dataType='class',dataCombination='012346-57',analysis='mcod')
```
Other available setups are specified in main.py

## Comparisons
DeepStreamOS is compared to OpenMax (
A. Bendale, T. Boult “Towards Open Set Deep Networks” IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016 pdf).  OpenMax code is adapted from [https://github.com/abhijitbendale/OSDN].
The OpenMax code adapted for DeepStreamCE comparison is stored in the modules_compare/openmax_code directory.
Run the following line in main.py:
```
start(dnnName='mobilenet', datasetName='mnistfashion',dataType='class',dataCombination='012346-57',analysis='openmax')
```
DeepStreamOS is compared to EVM. [The Extreme Value Machine](https://ieeexplore.ieee.org/document/7932895).
Our EVM comparison code uses the [EVM PyPI installer](https://pypi.org/project/EVM/) associated with the paper.
To execute an EVM comaprison, run the following line in main.py:
```
start(dnnName='mobilenet', datasetName='mnistfashion',dataType='sub',dataCombination='05-1237',analysis='evm')
```
## Outputs
Outputs from all runs are stored in the output directory

## Citation
If you use this paper/code in your research, please cite:
```
@article{chambersDeepStreamOSFastOpenSet2022,
  title = {{{DeepStreamOS}}: {{Fast}} Open-{{Set}} Classification for Convolutional Neural Networks},
  shorttitle = {{{DeepStreamOS}}},
  author = {Chambers, Lorraine and Gaber, Mohamed Medhat},
  date = {2022-02-01},
  journaltitle = {Pattern Recognition Letters},
  shortjournal = {Pattern Recognition Letters},
  volume = {154},
  pages = {75--82},
  issn = {0167-8655},
  doi = {10.1016/j.patrec.2022.01.011},
  url = {https://www.sciencedirect.com/science/article/pii/S0167865522000186},
  urldate = {2022-06-06}
}
```
