import numpy as np

def configure(dataset_str):
    if dataset_str == 'cora':
        params={"lr_rate_mul":0.1,"lambda":0.5,"gamma":0.2,"wd1":0.05,"wd2":0.00019,"dropout":0.55,"hidden":64,"myf":1.2,"layer":1,"hop":4,"alpha":None,"seed":2021}
        return params
    if dataset_str == 'citeseer':
        params={"lr_rate_mul":0.7,"lambda":1,"gamma":0.2,"wd1":0.005,"wd2":0.00019,"dropout":0.55,"hidden":128,"myf":1.2,"layer":1,"hop":2,"alpha":0.5,"seed":2020}
        return params
    if dataset_str == 'pubmed':
        params={"lr_rate_mul":1,"alpha":0.1,"lambda":0.5,"gamma":0.2,"wd1":0.005,"wd2":0.000005,"dropout":0.2,"hidden":128,"myf":1.2,"layer":1,"hop":6,"alpha":None,"seed":2020}
        return params