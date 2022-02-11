#!/usr/bin/env python
# coding: utf-8

# # Image Classifier: Animals, Buidings and Landscapes using DL and ML

# ### Dataset: The dataset consists of three classes, Animals, Buidings and Landscapes. Each class has 100 images. The images are in Jpeg format.

# ### Importing essential libraries

# In[1]:


from tensorflow.keras.layers import Conv2D,Dense,Flatten,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import tensorflow as tf
import os
import pickle
import splitfolders


# In[2]:


import warnings
warnings.filterwarnings("ignore")


# ### Checking the versions of essential libraries imported above

# In[3]:


tf.__version__


# In[4]:


pd.__version__


# In[5]:


np.__version__


# In[6]:


print(pickle.format_version)


# In[7]:


sn.__version__


# ### Splitting the folders for training, testing and validation.

# In[8]:


input_folder = "..."
output = "..."
splitfolders.ratio(input_folder, output, seed = 42, ratio=(.8, .1, .1))


# In[9]:


help(splitfolders.ratio)


# In[10]:


img_height, img_width = (224,224)
batch_size = 10

train_data_dir = "..."
valid_data_dir = "..."
test_data_dir = "..."


# ### Generating Data and dimensions of our images

# In[11]:


train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True,
                validation_split = 0.4)

train_generator = train_datagen.flow_from_directory(
                train_data_dir,
                target_size = (img_height, img_width),
                batch_size = batch_size,
                class_mode = 'categorical',
                subset='training')         #set as training data

validation_generator = train_datagen.flow_from_directory(
                valid_data_dir,       #same directory as training data
                target_size = (img_height, img_width),
                batch_size = batch_size,
                class_mode = 'categorical', #2-D one hot encoded labels
                subset='validation')    #set as validation data


# In[12]:


test_generator = train_datagen.flow_from_directory(
                test_data_dir,
                target_size = (img_height, img_width),
                batch_size = 1,
                class_mode = 'categorical',
                subset='training')


# ### Current data info.: Initially checking for no. of entries in each input data class

# In[13]:


#Number of images in Animals class in Input_data folder

Animals = len(os.listdir("..."))


# In[14]:


Animals


# In[15]:


#Number of images in Buildings class in Input_data folder

Buildings = len(os.listdir("..."))


# In[16]:


Buildings


# In[17]:


#Number of images in Landscapes class in Input_data folder

Landscapes = len(os.listdir("..."))


# In[18]:


Landscapes


# ### Displaying the three classes

# In[19]:


placed_no = [Animals,Buildings,Landscapes]


# In[20]:


stat = ['Animals', 'Buldings', 'Landscapes']


# In[21]:


colors = [ 'brown' , 'orange' , 'red']


# In[22]:


plt.pie(placed_no , labels = stat , colors=colors , autopct= '%0.1f%%')


# In[23]:


x,y = test_generator.next()
x.shape


# In[24]:


y.shape


# In[25]:


train_generator.num_classes


# ### Model Fitting

# In[26]:


#Training the CNN on the Training set and evaluating it on the Test set

base_model = ResNet50(include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_generator, epochs = 30)


# In[27]:


model.summary()


# ### Saving the model 

# In[28]:


model.save('Image Classification between Animals, Buildings and Landscapes.h5')


# ### Accuracy

# In[29]:


print(history.history.keys())


# In[30]:


# As per the training of the dataset, the plot shows the loss and accuracy of the model
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Performance')
plt.xlabel('Epoch')
plt.legend(["Loss","Accuracy"], loc='lower left')
plt.show()


# In[31]:


test_loss, test_acc = model.evaluate(test_generator, verbose=2)
print('\n Test Accuracy = ', test_acc)


# In[32]:


model = tf.keras.models.load_model('Image Classification between Animals, Buildings and Landscapes.h5')
filenames = test_generator.filenames
nb_samples = len(test_generator)
y_prob=[]
y_act=[]
test_generator.reset()
for _ in range(nb_samples):
    X_test,Y_test = test_generator.next()
    y_prob.append(model.predict(X_test))
    y_act.append(Y_test)

predicted_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_prob]
actual_class = [list(train_generator.class_indices.keys())[i.argmax()] for i in y_act]

out_df = pd.DataFrame(np.vstack([predicted_class,actual_class]).T,columns=['predicted_class','actual_class'])
confusion_matrix = pd.crosstab(out_df['actual_class'], out_df['predicted_class'], rownames=['Actual'], colnames=['Predicted'])

sn.heatmap(confusion_matrix,cmap='bone', annot=True)
plt.show()
print('Test Accuracy = {}'.format((np.diagonal(confusion_matrix).sum()/confusion_matrix.sum().sum()*100)))


# ### Predicting

# In[33]:


# Printing the predicted class

print(predicted_class)


# In[34]:


# Printing the actual class

print(actual_class)


# In[35]:


#Loading Model

model = load_model('Image Classification between Animals, Buildings and Landscapes.h5')


# In[36]:


# Loading self selected Image of an animal

image.load_img("...")


# In[37]:


# Loading self selected Image of a building

image.load_img("...")


# In[38]:


# Loading self selected Image of a landscape

image.load_img("...")


# In[39]:


# Assigning for easier prediction

mypred={0:"Animal",1:"Building",2:"Landscape"}


# In[40]:


# Predicting whether the loaded image is an Animal, Building or Landscape

test_image = image.load_img("...", target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
x=np.argmax(result)
print('Result = ',mypred[x])


# In[41]:


# Predicting whether the loaded image is an Animal, Building or Landscape

test_image = image.load_img("...", target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
x=np.argmax(result)
print('Result = ',mypred[x])


# In[42]:


# Predicting whether the loaded image is an Animal, Building or Landscape

test_image = image.load_img("...", target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
x=np.argmax(result)
print('Result = ',mypred[x])

