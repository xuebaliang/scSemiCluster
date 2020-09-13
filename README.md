# scSemiCluster
Introduction
-----
The identification of cell types plays an essential rolein the analysis of scRNA-seq data, which, in turn, influences the discovery of regulatory genes that induceheterogeneity. Here, we propose a flexible single cell semi-supervised clustering and annotation framework, scSemiCluster. Notably, without explicit feature alignment and batch effect correction,  scSemiCluster outperforms other state-of-the-art, single-cell supervised classification and semi-supervised clustering annotation algorithms in both simulation and real data. In addition, scSemiCluster is suitable for cell annotation on single data and cross-data. 
Requirement
-----
Python >= 3.6

Tensorflow (GPU version) >= 1.14

Keras >= 2.2

igraph 0.1.11

scanpy 1.4.3

scikit-learn >= 0.22.2

Example
-----
We have provided some explanatory descriptions for the codes implementation. Here, we use a mixed simulation data with two batches and seven cell types generated by Splatter.  to give an example. You can download this data from folder "scSemiCluster/data/simulation". You just need to download all code files and focus on the "scSemiCluster_model.py" file. We take the batch 1 as the reference data and batch 2 as target data. The sample size of reference data is half of target data. And the size of each cell type is arranged in a proportional sequence with a ratio of 0.8. You can run the following code in your command lines:

python scSemiCluster_model.py

After finishing the entire training, you can get that the annotation accuracy and clustering ARI on the target data is 0.9748 and 0.9522, respectively. Besides, the target prediction information is in the "target_prediction_matrix" variable. It is a data frame, include four columns, they are true label, true cell type, cluster label, annotation cell type. You can save it in .csv file. In order to show that scSemiCluster deals with the batch effects well,  we use UMAP to visualize the learned latent space representation in a two dimensional plane. We also use the preprocessed data after selecting highly variable genes to reduce to 32 dimensions by PCA algorithm as the basis for the existence of batch effects. From the results in the figure below, SemiCluster not only pulls the seven cell populations apart, but also guarantees the reference data and target data are well mixed in each cell type. This demonstrates that our method can be an effective batch effect correction algorithm. In the future, we would continue to improve our tool to help our users as much as possible.

![model](https://github.com/xuebaliang/scSemiCluster/blob/master/data/result/simulation_visualization.png)

Contributing
-----
Author email: clandzyy@pku.edu.cn
