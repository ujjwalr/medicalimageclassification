#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import urllib.request

def download(url):
    filename = url.split("/")[-1]
    if not os.path.exists(filename):
        urllib.request.urlretrieve(url, filename)


# Caltech-256 image files
download('https://s3.amazonaws.com/diabetesdata/images/sample.zip')
# Tool for creating lst file
download('https://raw.githubusercontent.com/apache/incubator-mxnet/master/tools/im2rec.py')


# In[10]:


import zipfile
with zipfile.ZipFile('sample.zip') as zipref:
    zipref.extractall('.')


# In[11]:


download('https://s3.amazonaws.com/diabetesdata/images/sample_labels.csv')


# In[12]:


import glob
import numpy as np
image_dir = 'images'
image_files = glob.glob(os.path.join(image_dir, '*.png'))
image_files.sort()
print (len(image_files))


# In[13]:


from PIL import Image
tmp = np.array(Image.open(image_files[0]))
print (tmp.shape)


# In[14]:


tmp


# In[ ]:


get_ipython().system(' mkdir imagesrgb')

for image in image_files:
    image_name = image.split('/')
    Image.open(image).convert('RGB').save('imagesrgb/'+image_name[1])


# In[28]:


get_ipython().system(' aws s3 cp imagesrgb/ s3://diabetesdata/images/imagesrgb/ --recursive --quiet')


# In[88]:


image_dir = 'imagesrgb'
image_files_rgb = glob.glob(os.path.join(image_dir, '*.png'))
image_files_rgb.sort()
print (len(image_files_rgb))


# In[89]:


tmp = np.array(Image.open(image_files_rgb[0]))
print (tmp.shape)


# In[90]:


tmp


# In[79]:


Image.open(image_files_rgb[1000])


# In[80]:


Image.open(image_files_rgb[5000])


# In[91]:


import pandas as pd
df = pd.read_csv('sample_labels.csv',header = 0)
df1 = df[['Image Index','Finding Labels']]
print (df1.shape)

from sklearn.model_selection import train_test_split

train, test = train_test_split(df1, test_size=0.2)

print (train.shape)
print (test.shape)


# In[92]:


import os, errno
get_ipython().system(' mkdir nihtrain')
for index, row in train.iterrows():
    folders = (row['Finding Labels']) 
    for names in str(folders).split('|'):
        try:
            os.makedirs('nihtrain/'+names)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# In[93]:


get_ipython().system(' mkdir nihtest')
for index, row in test.iterrows():
    folders = (row['Finding Labels']) 
    for names in str(folders).split('|'):
        try:
            os.makedirs('nihtest/'+names)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


# In[95]:


import shutil
for index, row in train.iterrows():
    image_name = row['Image Index']
    source = os.path.join(image_dir, image_name)
    folders = row['Finding Labels'] 
    for names in str(folders).split('|'):
        destination = os.path.join('nihtrain/', names)
        shutil.copy(source,destination)


# In[96]:


for index, row in test.iterrows():
    image_name = row['Image Index']
    source = os.path.join(image_dir, image_name)
    folders = row['Finding Labels'] 
    for names in str(folders).split('|'):
        destination = os.path.join('nihtest/', names)
        shutil.copy(source,destination)


# In[97]:


get_ipython().run_cell_magic('bash', '', 'python im2rec.py --list --recursive nihtrain nihtrain/\npython im2rec.py --list --recursive nihtest nihtest/')


# In[98]:


f = open('nihtest.lst','r')
lst_content = f.read()
print(lst_content)


# In[106]:


get_ipython().run_cell_magic('time', '', "import boto3\nfrom sagemaker import get_execution_role\n\nrole = get_execution_role()\n\nbucket='diabetesdata' # customize to your bucket\ncontainers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/image-classification:latest',\n              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/image-classification:latest',\n              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/image-classification:latest',\n              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/image-classification:latest'}\ntraining_image = containers[boto3.Session().region_name]")


# In[110]:


test = glob.glob(os.path.join('nihtrain/', '*/*.png'))
test.sort()

#print (np.array(Image.open(test[0])).shape)
print (len(test))


# In[107]:


# Four channels: train, validation, train_lst, and validation_lst
s3train = 's3://{}/images/train/'.format(bucket)
s3validation = 's3://{}/images/test/'.format(bucket)
s3train_lst = 's3://{}/images/train_lst/'.format(bucket)
s3validation_lst = 's3://{}/images/test_lst/'.format(bucket)

# upload the image files to train and validation channels
get_ipython().system('aws s3 cp nihtrain $s3train --recursive --quiet')
get_ipython().system('aws s3 cp nihtest $s3validation --recursive --quiet')

# upload the lst files to train_lst and validation_lst channels
get_ipython().system('aws s3 cp nihtrain.lst $s3train_lst --quiet')
get_ipython().system('aws s3 cp nihtest.lst $s3validation_lst --quiet')


# In[161]:


# The algorithm supports multiple network depth (number of layers). They are 18, 34, 50, 101, 152 and 200
# For this training, we will use 18 layers
num_layers = 18
# we need to specify the input image shape for the training data
image_shape = "3,1024,1024"
# we also need to specify the number of training samples in the training set
num_training_samples = 5601
# specify the number of output classes
num_classes = 15
# batch size for training
mini_batch_size = 32
# number of epochs
epochs = 12
# learning rate
learning_rate = 0.0001
# report top_5 accuracy
top_k = 5
# resize image before training
#resize = 256
# period to store model parameters (in number of epochs), in this case, we will save parameters from epoch 2, 4, and 6
checkpoint_frequency = 2
# Since we are using transfer learning, we set use_pretrained_model to 1 so that weights can be 
# initialized with pre-trained weights
use_pretrained_model = 0


# In[162]:


get_ipython().run_cell_magic('time', '', 'import time\nimport boto3\nfrom time import gmtime, strftime\n\n\ns3 = boto3.client(\'s3\')\n# create unique job name \njob_name_prefix = \'sagemaker-imageclassification-notebook\'\ntimestamp = time.strftime(\'-%Y-%m-%d-%H-%M-%S\', time.gmtime())\njob_name = job_name_prefix + timestamp\ntraining_params = \\\n{\n    # specify the training docker image\n    "AlgorithmSpecification": {\n        "TrainingImage": training_image,\n        "TrainingInputMode": "File"\n    },\n    "RoleArn": role,\n    "OutputDataConfig": {\n        "S3OutputPath": \'s3://{}/images/{}/output\'.format(bucket, job_name_prefix)\n    },\n    "ResourceConfig": {\n        "InstanceCount": 1,\n        "InstanceType": "ml.p2.16xlarge",\n        "VolumeSizeInGB": 50\n    },\n    "TrainingJobName": job_name,\n    "HyperParameters": {\n        "image_shape": image_shape,\n        "num_layers": str(num_layers),\n        "num_training_samples": str(num_training_samples),\n        "num_classes": str(num_classes),\n        "mini_batch_size": str(mini_batch_size),\n        "epochs": str(epochs),\n        "learning_rate": str(learning_rate),\n        "top_k": str(top_k),\n #       "resize": str(resize),\n        "checkpoint_frequency": str(checkpoint_frequency),\n        "use_pretrained_model": str(use_pretrained_model)    \n    },\n    "StoppingCondition": {\n        "MaxRuntimeInSeconds": 360000\n    },\n#Training data should be inside a subdirectory called "train"\n#Validation data should be inside a subdirectory called "validation"\n#The algorithm currently only supports fullyreplicated model (where data is copied onto each machine)\n    "InputDataConfig": [\n        {\n            "ChannelName": "train",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/images/train/\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-image",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/images/test/\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-image",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "train_lst",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/images/train_lst/\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-image",\n            "CompressionType": "None"\n        },\n        {\n            "ChannelName": "validation_lst",\n            "DataSource": {\n                "S3DataSource": {\n                    "S3DataType": "S3Prefix",\n                    "S3Uri": \'s3://{}/images/test_lst/\'.format(bucket),\n                    "S3DataDistributionType": "FullyReplicated"\n                }\n            },\n            "ContentType": "application/x-image",\n            "CompressionType": "None"\n        }\n    ]\n}\nprint(\'Training job name: {}\'.format(job_name))\nprint(\'\\nInput Data Location: {}\'.format(training_params[\'InputDataConfig\'][0][\'DataSource\'][\'S3DataSource\']))')


# In[163]:


# create the Amazon SageMaker training job
sagemaker = boto3.client(service_name='sagemaker')
sagemaker.create_training_job(**training_params)

# confirm that the training job has started
status = sagemaker.describe_training_job(TrainingJobName=job_name)['TrainingJobStatus']
print('Training job current status: {}'.format(status))

try:
    # wait for the job to finish and report the ending status
    sagemaker.get_waiter('training_job_completed_or_stopped').wait(TrainingJobName=job_name)
    training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
    status = training_info['TrainingJobStatus']
    print("Training job ended with status: " + status)
except:
    print('Training failed to start')
     # if exception is raised, that means it has failed
    message = sagemaker.describe_training_job(TrainingJobName=job_name)['FailureReason']
    print('Training failed with the following error: {}'.format(message))


# In[164]:


import json
training_info = sagemaker.describe_training_job(TrainingJobName=job_name)
status = training_info['TrainingJobStatus']
print("Training job ended with status: " + status)


# In[165]:


get_ipython().run_cell_magic('time', '', 'import boto3\nfrom time import gmtime, strftime\n\nsage = boto3.Session().client(service_name=\'sagemaker\') \n\ntimestamp = time.strftime(\'-%Y-%m-%d-%H-%M-%S\', time.gmtime())\nmodel_name="image-classification-model" + timestamp\nprint(model_name)\ninfo = sage.describe_training_job(TrainingJobName=job_name)\nmodel_data = info[\'ModelArtifacts\'][\'S3ModelArtifacts\']\nprint(model_data)\nhosting_image = containers[boto3.Session().region_name]\nprimary_container = {\n    \'Image\': hosting_image,\n    \'ModelDataUrl\': model_data,\n}\n\ncreate_model_response = sage.create_model(\n    ModelName = model_name,\n    ExecutionRoleArn = role,\n    PrimaryContainer = primary_container)\n\nprint(create_model_response[\'ModelArn\'])')


# In[166]:


from time import gmtime, strftime

timestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())
endpoint_config_name = job_name_prefix + '-epc-' + timestamp
endpoint_config_response = sage.create_endpoint_config(
    EndpointConfigName = endpoint_config_name,
    ProductionVariants=[{
        'InstanceType':'ml.p2.xlarge',
        'InitialInstanceCount':1,
        'ModelName':model_name,
        'VariantName':'AllTraffic'}])

print('Endpoint configuration name: {}'.format(endpoint_config_name))
print('Endpoint configuration arn:  {}'.format(endpoint_config_response['EndpointConfigArn']))


# In[167]:


get_ipython().run_cell_magic('time', '', "import time\n\ntimestamp = time.strftime('-%Y-%m-%d-%H-%M-%S', time.gmtime())\nendpoint_name = job_name_prefix + '-ep-' + timestamp\nprint('Endpoint name: {}'.format(endpoint_name))\n\nendpoint_params = {\n    'EndpointName': endpoint_name,\n    'EndpointConfigName': endpoint_config_name,\n}\nendpoint_response = sagemaker.create_endpoint(**endpoint_params)\nprint('EndpointArn = {}'.format(endpoint_response['EndpointArn']))")


# In[168]:


# get the status of the endpoint
response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
status = response['EndpointStatus']
print('EndpointStatus = {}'.format(status))
    
try:
    sagemaker.get_waiter('endpoint_in_service').wait(EndpointName=endpoint_name)
finally:
    resp = sagemaker.describe_endpoint(EndpointName=endpoint_name)
    status = resp['EndpointStatus']
    print("Arn: " + resp['EndpointArn'])
    print("Create endpoint ended with status: " + status)

    if status != 'InService':
        message = sagemaker.describe_endpoint(EndpointName=endpoint_name)['FailureReason']
        print('Training failed with the following error: {}'.format(message))
        raise Exception('Endpoint creation did not succeed')


# In[189]:


get_ipython().system(' rm /tmp/test.png')
get_ipython().system('aws s3 cp "s3://diabetesdata/images/test/Pleural_Thickening/00000997_004.png" /tmp/test.png')
file_name = '/tmp/test.png'
print (np.array(Image.open(file_name)).shape)
Image.open(file_name)


# In[190]:


import json
import boto3
runtime = boto3.Session().client(service_name='runtime.sagemaker') 

import numpy as np
with open(file_name, 'rb') as f:
    payload = f.read()
    payload = bytearray(payload)
response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                   ContentType='application/x-image', 
                                   Body=payload)
result = response['Body'].read()

# result will be in json format and convert it to ndarray
result = json.loads(result)
object_categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia','Pneumonia', 'Pneumothorax']

for i in range (0, len((list(result)))-1):
    print ("Label - " + object_categories[i] + ", Probability - " + str(result[i]))
index = np.argmax(result)
# the result will output the probabilities for all classes
# find the class with maximum probability and print the class index
print ("----------------------------------------------------------------------------")
index = np.argmax(result)
print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]))


# In[212]:


image_files = glob.glob(os.path.join('nihtrain', '*/*.png'))
print (len(image_files))
object_categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule', 'Pleural_Thickening', 'Pneumonia','Pneumonia', 'Pneumothorax']
res = []
for file in image_files:
    actual = str(list(file.split('/'))[1])
    with open(file, 'rb') as f:
        payload = f.read()
        payload = bytearray(payload)
    response = runtime.invoke_endpoint(EndpointName=endpoint_name, 
                                       ContentType='application/x-image', 
                                       Body=payload)
    result = response['Body'].read()
    result = json.loads(result)
    index = np.argmax(result)

    if actual == object_categories[index]:
        print("Result: label - " + object_categories[index] + ", probability - " + str(result[index]) + ", actual label - " + actual)
        res.append(object_categories[index])

print(len(res))


# In[213]:


import matplotlib.pyplot as plt

plt.hist(res)
plt.ylabel("Frequency")
plt.xlabel("Label")
plt.show()


# In[ ]:




