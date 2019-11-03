
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
def pre(total_pop):
    temp=[]
    for i in total_pop:
        k=str(i).split(',')
        x=""
        for j in k:
            x+=j
        temp.append(int(x))
    return temp


# In[2]:


df=pd.read_csv("state_pop.csv")
df2=pd.read_csv("percapita.csv")


# In[3]:


df.sort_values("S. No.",inplace=True,ascending=True,kind='quicksort')
df.head()


# In[4]:


df2.head


# In[5]:


middle_pop=pre(df["ten_lpa_pop"].values)
upper_pop=pre(df["tenPlus_pop"].values)
total=[]
for i in range(len(middle_pop)):
    total.append(int(middle_pop[i])+int(upper_pop[i]))
total     #total population that can bear byuing


# In[6]:


per_capita=df2["Per Capita Income"].values


# In[7]:


df=pd.read_csv("flipkart_com-ecommerce_sample.csv")


# In[8]:


uid=df["uniq_id"].values
p_name=df["product_name"].values
price=df["retail_price"].values
d_price=df["discounted_price"].values
brand=df["brand"].values


# In[9]:


import math
brand1=[]
for i in range(len(brand)):
    k=p_name[i].split()
    try: 
        if(math.isnan(brand[i])):
            brand1.append(k[0])
    except:
        brand1.append(brand[i])


# In[10]:


csv=pd.DataFrame({"uniq_id":uid,"product_name":p_name,"retail_price":price,"discounted_price":d_price,"brand":brand1})


# In[11]:


csv.to_csv("dataset.csv")


# In[12]:


csv["retail_price"]=csv["retail_price"].fillna(csv["retail_price"].median())


# In[13]:


csv["discounted_price"]=csv["discounted_price"].fillna(csv["discounted_price"].median())


# In[14]:


csv.to_csv("dataset.csv")


# In[15]:


discounted_price=csv["discounted_price"].values
retail_price=csv["retail_price"].values
uid=csv["uniq_id"].values


# In[16]:


"""input_vector=[]
for i in range(0,int(len(uid)//2)):
    l=[]
    for j in range(len(total)):
        l.append([[retail_price[i]],[total[j]],[per_capita[j]]])
    input_vector.append(l)
test_input_vector=[]
for i in range(len(uid)//2,len(uid)):
    l=[]
    for j in range(len(total)):
        l.append([[retail_price[i]],[total[j]],[per_capita[j]]])
    test_input_vector.append(l)
"""
input_vector=[]
for i in range(0,len(uid)):
    for j in range(len(total)):
        input_vector.append([retail_price[i],total[j],per_capita[j]])
len(input_vector[0])


# In[17]:


"""label=[]
for i in range(len(discounted_price)):
    l=[]
    for j in range(len(total)):
        l.append([discounted_price[i] *100 /per_capita[j]])
    label.append(l)
len(label[0])
"""
label=[]
for i in range(len(discounted_price)):
    label.append(discounted_price[i])
label


# In[18]:


from sklearn.neural_network import MLPRegressor


# In[33]:


clf = MLPRegressor(solver='adam', alpha=0.0001,hidden_layer_sizes=(5, 3), random_state=0, learning_rate="constant", learning_rate_init=0.9,momentum=0.9,activation="relu")


# In[34]:


clf.fit(input_vector[0:len(label)//2],label[0:len(label)//2])


# In[36]:


k=clf.predict(input_vector[:1000])


# In[37]:


k

