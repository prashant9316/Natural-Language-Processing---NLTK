# Natural-Language-Processing---NLTK
Natural Language Processing with NLTK  
The 'Restaurant_Reviews.tsv' file is our dataset for our model.  
This files contains reviews and a digit separated by tabs, that is why it is called tab separated values.  
The digit can be a 0 or 1 depending upon the the review, if it is a positive feedback, it is 1 else 0.  
## Data Preprocessing
1.) Removing symbols and taking only alphabets.  
2.) Converting all the alphabets to lower case to reduce the sizw of our final keywork lists.  
3.) splitting the sentence into a list of separate words  
4.) stemming the spliited words. i.e, converting ponies to pony. This is also called Lemmatization  
5.) Identifying stopwords and removing them from our list.  
6.) Converting the list to a string separated by spaces using join  
7.) appending  all the lists formed for each review in corpus.  

## Creating a Bag of Words Model
I am going to use CountVectorizer() class from sklearn.feature_extraction.text   
After creating it, we fill fit_transform it to our corpus.

## Using Logistic Regression to Train our model
I used the naive Bayes model to train the model  

## Predicting 
use the predict method to predict the test set.

## Saving the model for future Predictions
using pickle to dump our model into a file.
