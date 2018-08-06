#This program applies kmeans clustering to the iris dataset and produces 6 graphs of all axis combinations

from sklearn.cluster import KMeans

import numpy as np

import bokeh.plotting
import bokeh.io
import bokeh.layouts
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.io import output_notebook, show

from sklearn import datasets

#function for drawing graph
def generateGraph(graphTitle, xAxisLabel, yAxisLabel, dataset):
    kmean = KMeans(n_clusters=3) #only 3 classes so only group data into 3 clusters
    kmean.fit(dataset)

    #initialize plot
    plot = figure(width=500, height=500, title=graphTitle, x_axis_label = xAxisLabel, y_axis_label = yAxisLabel)

    #plot centroid / cluster center / group mean for each group

    clus_xs = []

    clus_ys = []

    #get the  cluster x / y values from the k-means algorithm

    for entry in kmean.cluster_centers_:
       clus_xs.append(entry[0])
       clus_ys.append(entry[1])
    #the cluster center is marked by a circle, with a cross in it

    plot.circle_cross(x=clus_xs, y=clus_ys, size=40, fill_alpha=0, line_width=2, color=['red', 'blue', 'purple'])

    plot.text(text = ['setosa', 'versicolor', 'virginica'], x=clus_xs, y=clus_ys, text_font_size='30pt')

    i = 0 #counter

    #begin plotting each petal length / width

    #get x / y values from the original plot data.

    #The k-means algorithm tells us which 'colour' each plot point is,

    #and therefore which group it is a member of.

    for sample in dataset:

       #"labels_" tells us which cluster each plot point is a member of

       if kmean.labels_[i] == 0:

           plot.circle(x=sample[0], y=sample[1], size=15, color="red")

       if kmean.labels_[i] == 1:

           plot.circle(x=sample[0], y=sample[1], size=15, color="blue")

       if kmean.labels_[i] == 2:

           plot.circle(x=sample[0], y=sample[1], size=15, color="purple")

       i += 1

    return plot

bokeh.plotting.output_notebook() #initialize bokeh in jupyter

#iris contains 150 samples, each has four features: sepal width and length, petal width and length
#array row structure: [sepal length, sepal width, petal length, petal width, class]

iris = datasets.load_iris()

#data variables for each graph

sepalWidth_sepalLength = iris.data[:,:2] #up to the 3rd column
#corresponding graph
graph1 = generateGraph("Iris Sepals", "Sepal Width", "Sepal Length", sepalWidth_sepalLength)
sepalWidth_petalLength = iris.data[:,1:3] #2nd and 3rd column
#corresponding graph
graph2 = generateGraph("Iris Sepal and Petal", "Sepal Width", "Petal Length", sepalWidth_petalLength)

sepalWidth_petalWidth = iris.data[:,1:4:2] #2nd to 4th column in steps of 2
#corresponding graph
graph3 = generateGraph("Iris Sepal and Petal", "Sepal Width", "Petal Width", sepalWidth_petalWidth)

petalWidth_sepalLength = iris.data[:,0:4:3] #1st and 3rd column in steps of 3
#corresponding graph
graph4 = generateGraph("Iris Sepal and Petal", "Petal Width", "Sepal Width", petalWidth_sepalLength)

petalWidth_petalLength = iris.data[:,2:] #3rd column onwards
#corresponding graph
graph5 = generateGraph("Iris Petals", "Petal Width", "Petal Length", petalWidth_petalLength)

sepalLenth_petalLength = iris.data[:,0:3:2] #1st and 3rd column onwards
#corresponding graph
graph6 = generateGraph("Iris Sepal and Petal", "Petal Length", "Sepal Length", sepalLenth_petalLength)

#print all graphs together
grid = gridplot([[graph1, graph2, graph3], [graph4, graph5],[graph6]])

show(grid)
