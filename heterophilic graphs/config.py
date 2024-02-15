import numpy as np

def configure(dataset_str):
    if dataset_str == 'chameleon':
        params={"lr_rate_mul":0.1,"alpha":0.5,"lambda":0.5,"gamma":0.2,"wd1":0.05,"wd2":0.0001,"dropout":0.2,"layers":1,"hop":1,"hidden":128,"myf":1.2,"seed":2021}
        return params
    if dataset_str == 'squirrel':
        params={"lr_rate_mul":0.5,"alpha":0.2,"lambda":0.5,"gamma":0.2,"wd1":0.05,"wd2":0.00019,"dropout":0.0,"layers":1,"hop":2,"hidden":128,"myf":1.2,"seed":2021} 
        return params
    if dataset_str == 'wisconsin':
        params={"lr_rate_mul":0.7,"alpha":0.4,"lambda":0.1,"gamma":0.2,"wd1":0.05,"wd2":0.0001,"dropout":0.3,"layers":4,"hop":1,"hidden":64,"myf":1.2,"seed":2021}
        return params
    if dataset_str == 'cornell':
        params={"lr_rate_mul":0.7,"alpha":0.4,"lambda":0.1,"gamma":0.2,"wd1":0.05,"wd2":0.0001,"dropout":0.3,"layers":4,"hop":1,"hidden":64,"myf":1.2,"seed":2021}
        return params
    if dataset_str == 'texas':
        params={"lr_rate_mul":0.7,"alpha":0.4,"lambda":0.1,"gamma":0.2,"wd1":0.05,"wd2":0.0001,"dropout":0.3,"layers":4,"hop":1,"hidden":128,"myf":1.2, "seed":2021} 
        return params
    if dataset_str == "flim":
        params={"lr_rate_mul":1.2,"alpha":0.2,"lambda":0.5,"gamma":0.2,"wd1":0.05,"wd2":0.0005,"dropout":0.0,"hidden":64,"layers":2,"hop":1, "myf":1.2,"seed":2021}
        return params