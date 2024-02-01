# Clothes-recommendation
Group project for programming for AI. 
Original Competition: https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations

# The preprocessing data download link
[h-and-m-finish-preprocess.zip](https://s3.openi.org.cn/opendata/attachment/f/4/f4f52b1a-2453-4cf1-bd2a-f5dff9304c5d?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=1fa9e58b6899afd26dd3%2F20240201%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240201T181657Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22h-and-m-finish-preprocess.zip%22&X-Amz-Signature=1b4d9eec313a48ff2c0efe1326e55ab09c08cc91422e1dcc4ad045a9f1fac88e)

# Implementaion
## Feature extraction
### Static features
1. Age
2. idx

### Dynamic features
1. The mean, standard deviation of price and sales_channel_id for each user in the transaction table for a given time period.
2. The mean, standard deviation of price and sales_channel_id for each item in the transaction table for a given time period.
3. Number of purchases per item
4. purchases per user
5. user-item purchases

## Model
1. CatBoost
2. LightBGM
