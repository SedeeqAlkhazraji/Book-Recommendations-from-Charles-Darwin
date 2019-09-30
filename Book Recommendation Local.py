

#  1. Darwin's bibliography
# Charles Darwin is one of the few universal figures of science. 
# His most renowned work is without a doubt his "On the Origin of Species" published in 1859 which introduced the concept of natural selection. 
# But Darwin wrote many other books on a wide range of topics, including geology, plants or his personal life. 
# In this notebook, we will automatically detect how closely related his books are to each other.
# To this purpose, we will develop the bases of a content-based book recommendation system, 
# which will determine which books are close to each other based on how similar the discussed topics are.
#  The methods we will use are commonly used in text- or documents-heavy industries such as legal, 
# tech or customer support to perform some common task such as text classification or handling search engine queries.
# Let's take a look at the books we'll use in our recommendation system.




# Import library
import glob

# The books files are contained in this folder
folder = "datasets/"

# List all the .txt files and sort them alphabetically
files = glob.glob(folder + "*.txt")
files.sort()
# ... YOUR CODE FOR TASK 1 ...
files






#  2. Load the contents of each book into Python
# As a first step, we need to load the content of these books into Python and do some basic pre-processing 
# to facilitate the downstream analyses. We call such a collection of texts a corpus. 
# We will also store the titles for these books for future reference and print their respective length to get a gauge for their contents.



# Import libraries
import re, os

# Initialize the object that will contain the texts and titles
txts = []
titles = []

for n in files:
    # Open each file
    f = open (n,encoding = 'utf-8-sig')
    # Remove all non-alpha-numeric characters
    data=re.sub('[\W_]+', ' ', f.read())
    # Store the texts and titles of the books in two separate lists
    titles.append (os.path.basename(n).replace(".txt", ""))
    txts.append(data)

print (titles)
# Print the length, in characters, of each book
[len(t) for t in txts]



# 3. Find "On the Origin of Species"
# For the next parts of this analysis, we will often check the results returned by our method for a given book. 
# For consistency, we will refer to Darwin's most famous book: "On the Origin of Species."
# Let's find to which index this book is associated.



# Browse the list containing all the titles
for i in range(len(titles)):
    # Store the index if the title is "OriginofSpecies"
    if titles[i] == "OriginofSpecies":
        ori = i
        
# Print the stored index
print (ori)
txts[ori]




# 4. Tokenize the corpus
# As a next step, we need to transform the corpus into a format that is easier to deal with for the downstream analyses.
#  We will tokenize our corpus, i.e., transform each text into a list of the individual words (called tokens) it is made of.
# To check the output of our process, we will print the first 20 tokens of "On the Origin of Species".



# Define a list of stop words
stoplist = set('for a of the and to in to be which some is at that we i who whom show via may my our might as well'.split())

# Convert the text to lower case 
txts_lower_case = [str.lower(t) for t in txts]

# Transform the text into tokens 
txts_split = [t.split() for t in txts_lower_case]


# Remove tokens which are part of the list of stop words
texts = [[word for word in txt if word not in stoplist] for txt in txts_split]
texts[ori]

# Print the first 20 tokens for the "On the Origin of Species" book
texts[ori][1:20]



# 5. Stemming of the tokenized corpus
# If you have read On the Origin of Species, you will have noticed that Charles Darwin can use different words to refer to a similar concept.
#  For example, the concept of selection can be described by words such as selection, selective, select or selects.
# This will dilute the weight given to this concept in the book and potentially bias the results of the analysis.
# To solve this issue, it is a common practice to use a stemming process,
#  which will group together the inflected forms of a word so they can be analysed as a single item: the stem.
#  In our On the Origin of Species example, the words related to the concept of selection would be gathered under the select stem.
# As we are analysing 20 full books, the stemming algorithm can take several minutes to run and,
#  in order to make the process faster, we will directly load the final results from a pickle file and review the method used to generate it.



'''
# Load the Porter stemming function from the nltk package
from nltk.stem import PorterStemmer

# Create an instance of a PorterStemmer object
porter = PorterStemmer()

# For each token of each text, we generated its stem
texts_stem = [[porter.stem(token) for token in text] for text in texts]

# Save to pickle file
pickle.dump( texts_stem, open( "datasets/texts_stem.p", "wb" ) )
'''

import pickle

# Load the stemmed tokens list from the pregenerated pickle file

texts_stem = pickle.load(open("datasets/texts_stem.p","rb"))

# Print the 20 first stemmed tokens from the "On the Origin of Species" book
texts_stem[ori]




# 6. Building a bag-of-words model
# Now that we have transformed the texts into stemmed tokens, we need to build models that will be useable by downstream algorithms.
# First, we need to will create a universe of all words contained in our corpus of Charles Darwin's books, which we call a dictionary.
# Then, using the stemmed tokens and the dictionary, we will create bag-of-words models (BoW) of each of our texts.
# The BoW models will represent our books as a list of all uniques tokens they contain associated with their respective number of occurrences.
# To better understand the structure of such a model, we will print the five first elements of one of the "On the Origin of Species" BoW model.



# Load the functions allowing to create and use dictionaries
from gensim import corpora

# Create a dictionary from the stemmed tokens
dictionary = corpora.Dictionary(texts_stem)

# Create a bag-of-words model for each book, using the previously generated dictionary
bows = [dictionary.doc2bow(t) for t in texts_stem]

# Print the first five elements of the On the Origin of species' BoW model
# ... YOUR CODE FOR TASK 6 ...
bows[ori][:5]



# 7. The most common words of a given book
# The results returned by the bag-of-words model is certainly easy to use for a computer but hard to interpret for a human.
# It is not straightforward to understand which stemmed tokens are present in a given book from Charles Darwin, and how many occurrences we can find.
# In order to better understand how the model has been generated and visualize its content,
# we will transform it into a DataFrame and display the 10 most common stems for the book "On the Origin of Species".



# Import pandas to create and manipulate DataFrames
import pandas as pd

# Convert the BoW model for "On the Origin of Species" into a DataFrame
df_bow_origin = pd.DataFrame(bows[ori])

# Add the column names to the DataFrame
df_bow_origin.columns = ["index","occurrences"]
df_bow_origin[:5]
# Add a column containing the token corresponding to the dictionary index
df_bow_origin['token'] = df_bow_origin["index"].apply(lambda t: dictionary[t])

# Sort the DataFrame by descending number of occurrences and print the first 10 values
df_bow_origin = df_bow_origin.sort_values("occurrences",ascending=False)

df_bow_origin[:5]



# 8. Build a tf-idf model
# If it wasn't for the presence of the stem "speci", we would have a hard time to guess this BoW model comes from the On the Origin of Species book.
# The most recurring words are, apart from few exceptions, very common and unlikely to carry any information peculiar to the given book.
#  We need to use an additional step in order to determine which tokens are the most specific to a book.
# To do so, we will use a tf-idf model (term frequency–inverse document frequency).
# This model defines the importance of each word depending on how frequent it is in this text and how infrequent it is in all the other documents.
# As a result, a high tf-idf score for a word will indicate that this word is specific to this text.
# After computing those scores, we will print the 10 words most specific to the "On the Origin of Species" book (i.e., the 10 words with the highest tf-idf score).



# Load the gensim functions that will allow us to generate tf-idf models
from gensim.models import TfidfModel

# Generate the tf-idf model
model = TfidfModel(bows)

# Print the model for "On the Origin of Species"
model[bows[ori]]





# 9. The results of the tf-idf model
# Once again, the format of those results is hard to interpret for a human.
# Therefore, we will transform it into a more readable version and display the 10 most specific words for the "On the Origin of Species" book.



# Convert the tf-idf model for "On the Origin of Species" into a DataFrame
df_tfidf = pd.DataFrame(model[bows[ori]])

# Name the columns of the DataFrame id and score
df_tfidf.columns = ["id","score"]
# Add the tokens corresponding to the numerical indices for better readability
df_tfidf['token'] = df_tfidf['id'].apply(lambda x: dictionary[x])

# Sort the DataFrame by descending tf-idf score and print the first 10 rows.
df_tfidf = df_tfidf.sort_values('score', ascending=False)
df_tfidf.head(10)






# 10. Compute distance between texts
# The results of the tf-idf algorithm now return stemmed tokens which are specific to each book.
# We can, for example, see that topics such as selection, breeding or domestication are defining "On the Origin of Species" (and yes, in this book,
# Charles Darwin talks quite a lot about pigeons too). Now that we have a model associating tokens to how specific they are to each book,
# we can measure how related to books are between each other.
# To this purpose, we will use a measure of similarity called cosine similarity
# and we will visualize the results as a distance matrix, i.e., a matrix showing all pairwise distances between Darwin's books.



# Load the library allowing similarity computations
from gensim import similarities

# Compute the similarity matrix (pairwise distance between all texts)
sims = similarities.MatrixSimilarity(model[bows])

# Transform the resulting list into a dataframe
sim_df = pd.DataFrame(list(sims))

# Add the titles of the books as columns and index of the dataframe
sim_df.columns = titles
sim_df.index = titles

# Print the resulting matrix
sim_df




# 11. The book most similar to "On the Origin of Species"
# We now have a matrix containing all the similarity measures between any pair of books from Charles Darwin!
#  We can now use this matrix to quickly extract the information we need, i.e., the distance between one book and one or several others.
# As a first step, we will display which books are the most similar to "On the Origin of Species,"
#  more specifically we will produce a bar chart showing all books ranked by how similar they are to Darwin's landmark work.



# This is needed to display plots in a notebook

# Import libraries
import matplotlib.pyplot as plt

# Select the column corresponding to "On the Origin of Species" and 
v = sim_df['OriginofSpecies']

# Sort by ascending scores
v_sorted = v.sort_values()

# Plot this data has a horizontal bar plot
v_sorted.plot.barh(x='lab', y='val', rot=0).plot()

# Modify the axes labels and plot title for a better readability
plt.xlabel("Score")
plt.ylabel("Book")
plt.title("Similarity")
plt.show()



# 12. Which books have similar content?
# This turns out to be extremely useful if we want to determine a given book's most similar work.
# For example, we have just seen that if you enjoyed "On the Origin of Species," you can read books discussing similar concepts such as
#  "The Variation of Animals and Plants under Domestication" or "The Descent of Man, and Selection in Relation to Sex."
# If you are familiar with Darwin's work, these suggestions will likely seem natural to you.
# Indeed, On the Origin of Species has a whole chapter about domestication and The Descent of Man,
# and Selection in Relation to Sex applies the theory of natural selection to human evolution.
# Hence, the results make sense.
# However, we now want to have a better understanding of the big picture and see how Darwin's books are generally related to each other
# (in terms of topics discussed). To this purpose, we will represent the whole similarity matrix as a dendrogram,
# which is a standard tool to display such data. This last approach will display all the information about book similarities at once.
#  For example, we can find a book's closest relative but, also, we can visualize which groups of books have similar topics
# (e.g., the cluster about Charles Darwin personal life with his autobiography and letters). If you are familiar with Darwin's bibliography,
#  the results should not surprise you too much, which indicates the method gives good results. Otherwise,
# next time you read one of the author's book, you will know which other books to read next
# in order to learn more about the topics it addressed.



# Import libraries
from scipy.cluster import hierarchy

# Compute the clusters from the similarity matrix,
# using the Ward variance minimization algorithm
Z = hierarchy.linkage(sims, 'ward')

# Display this result as a horizontal dendrogram
hierarchy.dendrogram(Z, leaf_font_size=8, labels=sim_df.index, orientation='left')
plt.show()




