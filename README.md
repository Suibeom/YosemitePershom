# YosemitePershom

This was a project to examine the topological structure of a dataset extracted from a natural image. The points are log-brightness patches found in the image, embedded in a high dimensional Euclidean space. A classic, early result in topological data analysis found all kinds of interesting topological structures inside such sets, including an embedded klein bottle!

I used a python package called Gudhi, based on its fast performance of basic Rips complex computations. Other packages do fancier analysis, but I was planning on doing that stuff by myself so I prefered speed.
