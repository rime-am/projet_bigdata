
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import length
from pyspark.sql.types import *
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml import Pipeline
from pyspark.sql.functions import udf, col, size
from pyspark.sql.types import ArrayType, StringType
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import html
import string
import nltk

#on crée notre session Spark
spark = SparkSession.builder.master("local[1]").appName("SparkByExamples.com").getOrCreate()
spark.conf.set('spark.sql.shuffle.partitions', '10000')

# notre fichier de données
kindle_json = spark.read.json('Kindle_Stode_51.json')

kindle_json.show(3)

kindle_json.createOrReplaceTempView('kindle_json_view')

data_json = spark.sql('''
  SELECT CASE WHEN overall<4 THEN 1
          ELSE 0
          END as label,
        reviewText as text
  FROM kindle_json_view
  WHERE length(reviewText)>2''')

data_json.groupBy('label').count().show()


# on prend une petite partie de nos données pour faciliter l'analyse
pos = data_json.where('label=0').sample(False, 0.05, seed=1220)
neg = data_json.where('label=1').sample(False, 0.25, seed=1220)
data = pos.union(neg)
data.groupBy('label').count().show()


#Longueur moyenne des avis

data.withColumn('longueur_avis', length('text')).groupBy('label').avg('longueur_avis').show()


# fonction pour preprocessing
def clean(text):
    line = html.unescape(text)
    line = line.replace("can't", 'can not')
    line = line.replace("n't", " not")
    # remplace ponctuations par espace
    pad_punct = str.maketrans({key: " {0} ".format(key) for key in string.punctuation})
    line = line.translate(pad_punct)
    line = line.lower()
    line = line.split()
    lemmatizer = nltk.WordNetLemmatizer()
    line = [lemmatizer.lemmatize(t) for t in line]

    # travail sur la négation
    # on ajoute "not_" pour les mots qui suivent "not", ou "no" jusqu'à a la fin de la phrase
    #ceci va nous aider dans notre analyse des sentiments
    tokens = []
    negated = False
    for t in line:
        if t in ['not', 'no']:
            negated = not negated
        elif t in string.punctuation or not t.isalpha():
            negated = False
        else:
            tokens.append('not_' + t if negated else t)

    invalidChars = str(string.punctuation.replace("_", ""))
    bi_tokens = list(nltk.bigrams(line))
    bi_tokens = list(map('_'.join, bi_tokens))
    bi_tokens = [i for i in bi_tokens if all(j not in invalidChars for j in i)]
    tri_tokens = list(nltk.trigrams(line))
    tri_tokens = list(map('_'.join, tri_tokens))
    tri_tokens = [i for i in tri_tokens if all(j not in invalidChars for j in i)]
    tokens = tokens + bi_tokens + tri_tokens

    return tokens


# un exemple pour montrer comment la fonction fonctionne
example = clean("This is such a good book! A love story for the ages, I can't wait for the second book!!")
print(example)

# Effectue notre preprocessing
clean_udf = udf(clean, ArrayType(StringType()))
data_tokens = data.withColumn('tokens', clean_udf(col('text')))
data_tokens.show(3)

# on separe nos données en training (70%) et testing (30%)
training, testing = data_tokens.randomSplit([0.7, 0.3], seed = 1220)
training.groupBy('label').count().show()

count_vec = CountVectorizer(inputCol='tokens', outputCol='c_vec', minDF=5.0)
idf = IDF(inputCol="c_vec", outputCol="features")

# modele Naive Bayes
nb = NaiveBayes()

pipeline_nb = Pipeline(stages=[count_vec, idf, nb])
model_nb = pipeline_nb.fit(training)
test_nb = model_nb.transform(testing)
test_nb.show(3)

roc_nb_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction', labelCol='label')
roc_nb = roc_nb_eval.evaluate(test_nb)
print("ROC de notre modele {}".format(roc_nb))

# precision de notre modele Naive Bayes
acc_nb_eval = MulticlassClassificationEvaluator(metricName='accuracy')
acc_nb = acc_nb_eval.evaluate(test_nb)
print("Precision (accuracy) de notre modele {}".format(acc_nb))


# cross validation de notre mdoele Naive Bayes
paramGrid_nb = (ParamGridBuilder()
                .addGrid(count_vec.minDF, [3.0, 5.0, 7.0, 10.0, 15.0])
                .addGrid(nb.smoothing, [0.1, 0.5, 1.0])
                .build())
cv_nb = CrossValidator(estimator=pipeline_nb, estimatorParamMaps=paramGrid_nb, evaluator=acc_nb_eval, numFolds=5)
cv_model_nb = cv_nb.fit(training)

test_cv_nb = cv_model_nb.transform(testing)
acc_nb_cv = acc_nb_eval.evaluate(test_cv_nb)
print("Precision de notre modele avec CrossValidator: {}".format(acc_nb_cv))

#on essaie maintenant notre modele avec nos propre données
review_1 = [
    "I liked the premise and most of the book. At the end parts I lost a little interest because I lost the thread of who was who. War is hell. MacLeod did his service unlike most of us."]

review_2 = [
    "Excellent first person account of the the daily life of a US Paratrooper. From training to deployment in combat situations in Afghanistan. Makes you trully understand and appreciate their sacrifices, well written and a great introduction to world history"]

review_3 = [
    "One of the worst books I have ever read, one word to describe it : trash!"]

schema = StructType([StructField("text", StringType(), True)])

text = [review_1, review_2, review_3]
review_new = spark.createDataFrame(text, schema=schema)

# preprocessing de nos nouvelels phrases
review_new_tokens = review_new.withColumn('tokens', clean_udf(col('text')))
review_new_tokens.show()


# Prediction en utilisant nottre modele Naive Bayes
result = cv_model_nb.transform(review_new_tokens)
result.select('text', 'prediction').show()
