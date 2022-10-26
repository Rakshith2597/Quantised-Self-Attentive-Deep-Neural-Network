cxr8_labels.json : Contains labels associated with each images in list format. Example, "00000001_000.png": ["Cardiomegaly"]
data_split.json : Contains a dictionary. keys==[train,val,test] and values == list of image names.
no_findings.json: Contains list of files with labels as no_findings. However, some files where accidently skipped.
new_no_findings.json: Contains list of files with labels as no_findings. Curated version of no_findings.json
problem_files.json: Contains images with shape [1024,1024,4]. These images were not used for experiments.