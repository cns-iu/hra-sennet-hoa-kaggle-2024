1. python create_volumetric_image.py
   - Read traning data from ../data/blood-vessel-segmentation
   - Write volumetric image to ../data/volumetric_images/all (contains sparse data) and ../data/volumetric_images/dense (dense only)

2. python create_sample_data.py
   - Read volumetric image from ../data/volumetric_images/dense
   - Write sample data to ../data/sample-data which data are used for checking accuracy when training
   
3. python train_0.py
   - Read training data from ../data/volumetric_images/all
   - Write dataset to ../data/dataset on each 4 epochs
   - Write predict results of ../data/sample-data to ../data/temp/model-0/{epoch} for checking progress
   - Save model weights to ../data/models/model-0
   
4. python submit.py
   - Read model from ../data/models/model-0
   - Read data from ../data/blood-vessel-segmentation/test
   - Write result to ./submission.csv
 
