# Assignment B:

## Anchor boxes:

Anchor boxes are a set of predefined bounding boxes of a certain height and width. These boxes are defined to capture the scale and aspect ratio of specific object classes you want to detect and are typically chosen based on object sizes in your training datasets. During detection, the predefined anchor boxes are tiled across the image.

### How Do Anchor Boxes Work?

The position of an anchor box is determined by mapping the location of the network output back to the input image. The process is replicated for every network output. The result produces a set of tiled anchor boxes across the entire image. Each anchor box represents a specific prediction of a class. For example, there are two anchor boxes to make two predictions per location in the image below.

![image](https://user-images.githubusercontent.com/51078583/126828685-6760cba1-11c1-4299-9a1f-db6743030334.png)

Each anchor box is tiled across the image. The number of network outputs equals the number of tiled anchor boxes. The network produces predictions for all outputs.

## COCO Dataset:

COCO (Common Objects in Context), being one of the most popular image datasets out there, with applications like object detection, segmentation, and captioning - it is quite surprising how few comprehensive but simple, end-to-end tutorials exist. The data represented in the sample dataset is as follows:
id: 0, height: 330, width: 1093, bbox:[69, 464, 312, 175],
where
- id is the Class ID
- height is the height of the image
- width is the width of the image
- bbox contains info regarding the bounding box
  - 1st element is X co-ordinate centroid of the bounding box
  - 2nd element is Y co-ordinate centroid of the bounding box
  - 3rd element is the bounding box width scaling factor
  - 4th element is the bounding box height scaling factor

## K-Means algorigthm

K-Means Clustering is an unsupervised machine learning algorithm.

Kmeans algorithm is an iterative algorithm that tries to partition the dataset into Kpre-defined distinct non-overlapping subgroups (clusters) where each data point belongs to only one group. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.

### Choosing the right number of clusters

Often times the data you’ll be working with will have multiple dimensions making it difficult to visual. As a consequence, the optimum number of clusters is no longer obvious. Fortunately, we have a way of determining this mathematically.
We graph the relationship between the number of clusters and Within Cluster Sum of Squares (WCSS) then we select the number of clusters where the change in WCSS begins to level off (elbow method).

![image](https://user-images.githubusercontent.com/51078583/126828950-5e1221ba-178f-4182-8bba-fd69842f2d17.png)

# Anchor box using K-Means in COCO Dataset Provided:
## Normalized data:

![image](https://user-images.githubusercontent.com/51078583/126817200-7214209d-387a-4345-aed8-9ffed38fb8e7.png)

## Scatter Plot of normalized Boundng box b_h and b_w:

![image](https://user-images.githubusercontent.com/51078583/126817354-724a4782-2d2b-4b5d-9073-66a74124a7ba.png)

## Distortion Plot for Finding optimum K for K means:

![image](https://user-images.githubusercontent.com/51078583/126817430-ba0dcd06-f568-4744-8f9e-87c2b0a1c4ca.png)

## K-means on the scatter Plot and Corresponding Anchor Boxes:

### When K = 3 :

| Scatter Plot | Anchor Boxes |
|--|--|
| ![image](https://user-images.githubusercontent.com/51078583/126817747-619ec0ec-6572-4a21-89fd-749cc96bc539.png) | ![image](https://user-images.githubusercontent.com/51078583/126817779-d09e9338-5f74-422c-a1c1-2f9d54b9c0e7.png) |

### When K = 4 :

| Scatter Plot | Anchor Boxes |
|--|--|
| ![image](https://user-images.githubusercontent.com/51078583/126817940-be9d427f-40fa-4240-be12-7cb7bd15a4ac.png) | ![image](https://user-images.githubusercontent.com/51078583/126817913-522f1d25-36bd-4697-b3ba-4bc1cb32b69f.png) |

### When K = 5 :

| Scatter Plot | Anchor Boxes |
|--|--|
| ![image](https://user-images.githubusercontent.com/51078583/126818030-b28b2969-4cff-44dc-8b6a-db328decad37.png) | ![image](https://user-images.githubusercontent.com/51078583/126818180-638411cd-31cd-4039-96d4-923b634c1b6c.png) |

### When K = 6 :

| Scatter Plot | Anchor Boxes |
|--|--|
| ![image](https://user-images.githubusercontent.com/51078583/126818088-495d611c-8022-4116-903e-d03aeaa4a306.png) | ![image](https://user-images.githubusercontent.com/51078583/126818199-c115bd21-a2b2-4726-befd-2290c703d040.png) |

## Contributors:    
1. Avinash Ravi
2. Nandam Sriranga Chaitanya
3. Saroj Raj Das
4. Ujjwal Gupta

