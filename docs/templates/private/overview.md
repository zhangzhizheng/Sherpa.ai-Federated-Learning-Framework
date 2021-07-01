# Private

This package contains most of the core elements of the framework that are used in almost every line of code that you will 
write using the Sherpa.ai Federated Learning and Differential Privacy Framework.

The most important element in the framework is the [DataNode](../data_node). 
A DataNode represents a device or data silos containing private data. 
In real world scenarios, these data are typically property of a user or company, 
thus the access to it is, by default, protected. 

We normally assign a model to the DataNode to learn from the private data. 
All extra operations involving the private data (e.g. reshape, normalization etc.) 
or involving other private properties of the DataNode must be previously configured for the specific node(s). 
