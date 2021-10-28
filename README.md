# MHGCN
For WWW‘22-submission
> Multiplex Heterogeneous Graph Convolutional Network
## Dependencies
Recent versions of the following packages for Python 3 are required:
* numpy==1.21.2
* torch==1.9.1
* scipy==1.7.1
* scikit-learn==0.24.2
* pandas==0.25.0
## Datasets
The preprocessed datasets are available at:
* Alibaba https://tianchi.aliyun.com/competition/entrance/231719/information/
* Amazon http://jmcauley.ucsd.edu/data/amazon/
* Aminer https://github.com/librahu/
* IMDB https://github.com/seongjunyun/Graph_Transformer_Networks
* DBLP https://www.dropbox.com/s/yh4grpeks87ugr2/DBLP_processed.zip?dl=0
## Usage
You need to determine the dataset in Node_classification.py or Link_prediction.py, and select the number of weights and subnet information matching the corresponding dataset in Model.py and Decoding_matrix_aggregion.

Execute the following command to run the node classification task:
* python Node_Classification.py

Execute the following command to run the link prediction task:
* python Link_Prediction.py

Refer to the comments in the code for more information
