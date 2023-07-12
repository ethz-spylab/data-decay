# %%
import pickle
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import random

# %%
idx23_0 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2023/order0.p","rb"))
txt23_0 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2023/txt0.p","rb"))
img23_0 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2023/img0.p","rb"))

# %%
idx23_1 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2023/order1.p","rb"))
txt23_1 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2023/txt1.p","rb"))
img23_1 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2023/img1.p","rb"))

# %%
df_train = pd.read_csv("/data/cc3m/cc3m_2023/Train_GCC-training.tsv", sep='\t', names=["caption","url"], usecols=range(0,2))

# %%
print("min idx:",min(idx23_0))
print("max idx:",max(idx23_0))

# %%
print("idx23_0 length:",len(idx23_0))
print("txt23_0 shape:",txt23_0.shape)
print("img23_0 shape:",img23_0.shape)

# %%
df_train

# %%
i=1940906
j=1601006

# %%
ii = idx23_0.index(i+1)
jj = idx23_0.index(j+1)

# %%
print(ii)
print(jj)
print(idx23_0[ii])

# %%
print("caption comparison:", np.sum(txt23_0[ii] * txt23_0[jj]))
print("image comparison(embedding):", np.sum(img23_0[ii]*img23_0[jj])) 
print("image comparison:", np.sum(img23_0[ii]!=img23_0[jj])) 
# images are not same, but very similar!

# %%
rows = open("cc3m.tsv").readlines()
print(rows[i], rows[j])
#Thats what was shared on spaces, but as we can see the captions do not match. Instead we need to go for i-1, j-1

# %%
rows = open("cc3m/cc3m_2023/Train_GCC-training.tsv").readlines()
print(rows[i], rows[j])

# %%
print(df_train['caption'][i])
print(df_train['caption'][j])

# %%
print(df_train['caption'][i-1])
print(df_train['caption'][j-1])

# %%
print(df_train['url'][i-1])
print(df_train['url'][j-1])
#As we can see images are almost same. The former is a bit larger(scaled up kind of) and has watermark, but both former and latter depict ...
#same scene!

# %%
#Let's find some other matching captions to make sure we understand how indexes work
#Find the duplicated captions and pick the first one
cpt_num = 1893
cpt = df_train[df_train.duplicated(subset=['caption'],keep=False)]['caption'].iloc[cpt_num]
print("chosen duplicated caption:",cpt)
indexes = df_train[df_train['caption'] == cpt].index
print("indexes of that caption:",indexes)
#When finding the position of i in idx23_0, we searched for i+1. The indexes we have are already 1 less than i's ...
#so we now need to search for indexes + 2!
for k in range(1,len(indexes)):
    print("comparing with ",k+1,"th index: ",np.sum(txt23_0[idx23_0.index(indexes[0]+2)] * txt23_0[idx23_0.index(indexes[k]+2)]))

# %%
df_train[df_train['caption']==cpt]

# %%
#Turns out url is no longer accessible!
df_train['url'][3047811]

# %%
#I realized some url actually return an image but not included in order

# %%
max_idx23_0 = max(idx23_0)
max_idx23_0

# %%
##Check if they are in order1. They are!
print(idx23_1.index(max_idx23_0+91))
print(idx23_1.index(max_idx23_0+12345))

# %%
x = 3041366 + 12345
print(idx23_1.index(x))
df_train['url'][x-2]

# %%
#Now look at 2018 data and compare it with current data

# %%
idx18_0 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2018_b/order0.p","rb"))
txt18_0 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2018_b/txt0.p","rb"))
img18_0 = pickle.load(open("../../data/cc3m_embed/embed_cc3m_2018_b/img0.p","rb"))

# %%
idx18_1 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2018_b/order1.p","rb"))
txt18_1 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2018_b/txt1.p","rb"))
img18_1 = pickle.load(open("/data/cc3m_embed/embed_cc3m_2018_b/img1.p","rb"))

# %%
print("idx18_0 len: ", len(idx18_0))
print("idx18_1 len: ", len(idx18_1))

# %%
print("min idx18_0:",min(idx18_0))
print("max idx18_0:",max(idx18_0))
print("min idx18_1:",min(idx18_1))
print("max idx18_1:",max(idx18_1))

# %%
#2018 data has nearly all the images
print((len(idx18_0)+len(idx18_1))/len(df_train)*100)
print(len(df_train) - (len(idx18_0)+len(idx18_1)))

# %%
idx18 = idx18_0 + idx18_1
txt18 = np.concatenate((txt18_0, txt18_1), axis=0)
img18 = np.concatenate((img18_0, img18_1), axis=0)
idx23 = idx23_0 + idx23_1
txt23 = np.concatenate((txt23_0, txt23_1), axis=0)
img23 = np.concatenate((img23_0, img23_1), axis=0)

# %%
#But it has some duplicates
count_idx18 = np.zeros(max(idx18)+1)
for i in range(len(idx18)):
    count_idx18[idx18[i]] += 1
print("18 without duplicates: ", np.sum(count_idx18 > 0.5)/len(df_train)*100)
print("#ids duplicated: ",np.sum(count_idx18>1.1))
print("tot # of duplications: ", np.sum(count_idx18[count_idx18>1.1]))

# %%
#2018 data has nearly 84% of images
(len(idx23_0)+len(idx23_1))/len(df_train)*100

# %%
#some indexes are in 2023 data but not in 2018!
missing_indexes = list(set(idx23) - set(idx18))
len(missing_indexes)

# %%
x = 987
print(idx23_0.index(x))
print(idx18_0.index(x))
print(np.sum(txt23_0[idx23_0.index(x)] * txt18_0[idx18_0.index(x)]))
print(np.sum(img23_0[idx23_0.index(x)] * img18_0[idx18_0.index(x)]))

# %%
print(np.sum(img23_0[idx23_0.index(x)] * img23_0[idx23_0.index(x)]))
print(np.sum(img18_0[idx18_0.index(x)] * img18_0[idx18_0.index(x)]))

# %%
inv_idx18 = np.zeros(max(idx18)+1)
for i in range(len(idx18)):
    inv_idx18[idx18[i]] = i

missing_indexes = list(set(idx23) - set(idx18))

save_freq = 50000

with open('comparison_2.txt', 'a') as f:
    for i in range(len(idx23)):
        idx_c = idx23[i]
        if idx_c in missing_indexes:
            continue
        txt_c = np.sum(txt23[i]*txt18[int(inv_idx18[idx_c])])
        img_c = np.sum(img23[i]*img18[int(inv_idx18[idx_c])])
        f.write(str(i)+','+str(idx_c)+','+str(txt_c)+','+str(img_c)+'\n')
        if i % save_freq == 0 and i != 0:
            print("step: ",i)
print("completed")

# %%
deneme = pd.read_csv("comparison_new.txt", sep=',',names=['idx','caption','image'])

# %%
deneme

# %%
deneme2 = pd.read_csv("comparison_2.txt", sep=',',names=['idx','caption','image'])

# %%
deneme2

# %%
len(idx23)

# %%
deneme.to_csv("comparison2.tsv",index=True,sep='\t')

# %%
comparison = pd.read_csv("comparison.tsv", sep='\t',index_col=0)

# %%
comparison

# %%
comp_np = comparison.to_numpy()

# %%
comp_np[:,2]

# %%
plt.hist(comp_np[:,2],200)
plt.xlabel("img18 & img23")
plt.ylabel("#")
plt.title("image comparison,200 bins,tot 2791575")
plt.savefig('plots/img_comparison.png')
#plt.show()

# %%
plt.hist(comp_np[:,1],100)
plt.xlabel("txt18 & txt23")
plt.ylabel("#")
plt.title("caption comparison, 100 bins")
plt.savefig('plots/txt_comparison_full.png')
plt.show()

# %%
plt.hist(comp_np[comp_np[:,1]<0.995][:,1],100)
plt.xlabel("txt18 & txt23")
plt.ylabel("#")
plt.title("caption comparison with sim<0.995,100 bins,4446 tot")
plt.show()
#plt.savefig('plots/txt_comparison.png')

# %%
print("# of txt sim < 0.99: ",np.sum(comp_np[:,1]<0.995))
print("% of txt sim < 0.99: ",np.sum(comp_np[:,1]<0.995)/len(comparison)*100)

# %%
comparison[comparison['caption']<0.99]

# %%
x = 1519
np.sum(txt23[idx23.index(x)]*txt18[idx18.index(x)])

# %%
df_train['caption'][x-2]

# %%
missing_indexes = list(missing_indexes)
len(missing_indexes)

# %%
df_train['url'][missing_indexes[6]-2]

# %%


# %%
clow = comparison[comparison['caption']<0.9].sample(100)
clow

# %%
clow.to_csv("clow.tsv",index=True,sep='\t')

# %%
clow2 = pd.read_csv("clow.tsv", sep='\t',index_col=0)

# %%
clow2

# %%
chigh = comparison[comparison['caption']>0.995].sample(100)
chigh

# %%
chigh.to_csv("chigh.tsv",index=True,sep='\t')

# %%
df_train_low = df_train.loc[clow['idx'].to_numpy()-2]
df_train_low

# %%
df_train_low.to_csv("df_c_low.tsv",index=True,sep='\t')

# %%
df_train_high = df_train.loc[chigh['idx'].to_numpy()-2]
df_train_high

# %%
df_train_high.to_csv("df_c_high.tsv",index=True,sep='\t')

# %%
a = pd.read_csv("daniel/df_c_low.tsv", sep='\t',index_col=0)

# %%
a

# %%
b = pd.read_csv("daniel/df_c_low.tsv", sep='\t')

# %%
df_train['caption'].loc[10]

# %%
df_train['caption'][1331163-2]

# %%
df_train.head(11)

# %%


# %%


# %%
x=1940905
y=1601005
print(df_train['caption'][x])
print(df_train['caption'][y])
print(np.sum(txt23[idx23.index(x)]*txt18[idx18.index(x)]))
print(np.sum(img23[idx23.index(x)]*img18[idx18.index(x)]))

# %%
k = 10000
trials = np.zeros(k)
for i in range(k):
    x = random.randint(0, len(idx23))
    y = random.randint(0, len(idx18))
    trials[i] = np.sum(img23[x]*img18[y])
plt.hist(trials,50)
plt.show()

# %%
k = 10000
trials = np.zeros(k)
for i in range(k):
    x = random.randint(5, len(idx23)-5)
    y = random.randint(x-5, x+5)
    trials[i] = np.sum(img23[x]*img18[y])
plt.hist(trials,50)
plt.show()

# %%
max(idx23)

# %%
max(idx18)

# %%
len(df_train)

# %%
df_train.tail(3)

# %%
set([x for x in idx23 if idx23.count(x) > 1])

# %%
count_idx18 = np.zeros(max(idx18)+1)
for i in range(len(idx18)):
    count_idx18[idx18[i]] += 1

# %%
np.sum(count_idx18 > 0.5)/len(df_train)*100

# %%
idx18[5]

# %%
idx18.index(7387)

# %%
np.where(count_idx18>1)[0][0]

# %%
count_idx18[155882]

# %%
for i in range(len(idx18)):
    if idx18[i] == 155882:
        print(i)

# %%
idx18[153558]

# %%
idx18.index(155882)

# %%
np.sum(count_idx18>1)
np.sum(count_idx18[count_idx18>1])

# %%


# %%
print(len(df_train) - (len(idx18)))

# %%
len(missing_indexes)

# %%
df_train[df_train.duplicated(subset=['idx'],keep=False)]

# %%
count_idx23 = np.zeros(max(idx23)+1)
for i in range(len(idx23)):
    count_idx23[idx23[i]] += 1

# %%
np.sum(count_idx23>1)

# %%
missing_indexes = list(set(idx23) - set(idx18))

# %%
df_train['url'][-2+missing_indexes[0]]

# %%
img18.shape

# %%
x = 2876
np.sum(img23[idx23.index(x)]*img18[idx18.index(x)])

# %%



