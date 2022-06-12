# challenge-ilikedata
Welcome to my response to the Data Science technical challenge!

This is the summary page/entry point to my response. It consists of a few sections:
- Excecutive summary
- Responses to the 5 questions asked in Stage 4:DELIVER
- Instructions for full reproducibility

# Executive summary
Understanding customer gender is highly beneficial to THE ICONIC as a business. It enables tailored user experience, branding strategy, marketing, product and most importanly merchandising. There's great potential in uplifting cross-sell/up-sell, resulting in direct revenue increase. 

We have built a statistically robust pipepline, employing state-of-the-art methodology such as Weakly Supervised labelling and machine learning, that is capable of "inferring" customer gender from their purchase behaviour. 

By harnessing the rich customer data available in our organisation, as well as our deep SME knowledge, this pipeline is capable of fast scaling up and iterative improvement in relatively short term. 

A quick Proof of Concept can be achieved in 2-4 weeks time frame, which will provide proof, insights and data points to build a commercially viable business case. 

# Responses to Stage4: DELIEVER questions

### 1. How did you clean the data and what was wrong with it? Close to 90% of a Data Scientist's job is in cleaning data
As suggested by the instructions, there were two issues with the raw data: 
- "days_since_last_order" was actually "hours_since_last_order", I fixed it by simply divding it by 24.
- "average_discount_used" had 4 extra zeros, it's fixed by dividing by 10000

The data exploration and manipulation was done purely in pandas/numpy, you can find the full process and my comments in a Jupyter notebook [here](notebooks/stage2_clean.ipynb)

### 2. What are the features you used as-is and which one did you engineer using the given ones? What do they mean in the real world?
I consider there are 3 groups of features in the customer data provided: 
- Numeric transational features that span the customers entire life-time (orders, items, revenue etc.). Customers who have been active longer will obviously have much higher values, which wouldn't be a fair comparison to newer customers. I normalised these features by their tenure (months active). 
- Other numeric features that don't make sense or necessaily need to be normalised by tenure: days_since_first/last_order, tenure itself, addresses, devices, average_discount etc. I simply converted them to float. 
- Categorical features: is_newsletter_subscriber, payment types etc. I converted them to 1s and 0s. 

Full details can be found in [this notebook](notebooks/stage3_build_feature_engineering.ipynb)

After exploration and experimenting in Jupyter, I also refactored the feature engineering pipeline into a [Python Module](src/feature_engineering.py) that can be easily reused across multiple stages, it comes with some simple [Unit Test](test/test_feature_engineering.py) as well.

### 3. What does the output look like - how close is the accuracy of the prediction in light of data with labelled flags?
I would consider two things as the "final output":
- "Inferred" Gender labels for the entire customer dataset provided, which is stored [here](data/processed/training_set.parquet). This is produced by a programmingly labelling process called Weak Supervision. It essentially scales up my intuition on this task drastically so that I can label the entire dataset in an hour while stay true to the "SME knowledge" I possess as an individual. If true experts were involved in this same process, labels of very high quality will be produced. 
- A Machine Learning model trained on Weakly Supervised labels, that takes in new/unseen customer data and predicts customer gender. This is a generalisation of above labelling process to broader dataset and context. The whole process can be found in [this notebook](notebooks/stage3_build_train_ml.ipynb). 

When it comes to accuracy in real world, it's impossible and dangerous to call out without a ground truth hold out test set. If we do have such ground truth, it should be easy to test the accuracy and drastically improve this model. 

### 4. What other features and variables can you think of, that can make this process more robust? Can you make a recommendation of top 5 features you'd seek to find apart from the ones given here
It's more productive to work with domain SMEs to identify best features to use. That been said, based on my experience with customer/service/product data in general, here are some features I might explore:
- Customer browsing behaviour, that would help with customers who have few purchases
- Type of items in wishlist/shopping cart
- Recent purchase behaviour instead of life-time statistics, since people may share account
- Marketing campaign/newsletter click through, did they click on female/male product ads?
- Customer feedback/review/complaints data, some of them would leave pronouns and first names in there, which would be much more accurate when it comes to inferring gender using NLP

### 5. Summarize your findings in an executive summary
Please refer to the "Excecutive summar" section above. 

# Reproducibility instructions
> Disclaimer: I've tested below steps on Macbook Pro (x86, not M1 model) by deleting everything and starting from zero following these steps. However I haven't tested with other hardware therefore it's not guaranteed to work everywhere. I've put Conda dependency in a slightly relaxed way hoping Conda would workout the dependencies for different hardware automatically. 

