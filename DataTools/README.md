# DataTools
###### JasonGUTU
This package provides some useful functions and DataSet classes for Image processing and Low-level Computer Vision.
### Structure
- `DataSets` contains some DataSet class, all the child class of torch.utils.data.Dataset.
- `FileTools` contains tools for file management
- `Loaders` contains Image loaders
- `Prepro` contains self-customized pre-processing functions or classes
### Docs
#### DataSets.py
All the classes inherited from torch.utils.data.Dataset are self-customized Dataset classes
```[python]
# `TestDataset` is a Dataset classes
# Instantiation
dataset = TestDataset(*args, **kwargs)
# Use index to retrieve
first_data = dataset[0]
# Number of samples
length = len(dataset)
```
In this file, Datasets contain:
- SRDataSet(torch.utils.data.Dataset)
This Dataset is for loading small images like image-91. The images are small, direct loading has little effect on performance.
This
