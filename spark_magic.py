import re
import sys
import pyspark as ps
import numpy as np

from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.ml.feature import Tokenizer

def remove_urls(text, replace="<URL>"):
	# https://mathiasbynens.be/demo/url-regex
    #return re.sub('https?:\/\/.*[\r\n]*', replace, text)
    return re.sub(r'(http[s]?://)*[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)', replace, text)

negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",
			"haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",
			"wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",
			"can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",
			"mustn't":"must not"}
neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b') 

def expand_contracted_negations(text):
    # There is ofcourse other contractions which should be expanded
    # However negations are the easy ones...
	
	return neg_pattern.sub(lambda x: negations_dic[x.group()], text)

def remove_newlines_whitespace(text):
    # Remove newlines
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')
    return re.sub('\s+', ' ', text).strip()

def lower(text):
	# Trivial, yes
	return text.lower()

def remove_special(text):
    return re.sub("[^A-za-z0-9\s]+", '', text)

def preprocess(text):
    # Do some initial cleaning before tokenization
    if text:
        text = remove_newlines_whitespace(text)
        text = lower(text)
        text = remove_urls(text)
        text = remove_special(text)
        return text

if __name__ == '__main__':
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate() 

    sc = spark.sparkContext

    tokenizer = Tokenizer()
    tokenizer.setInputCol('content')
    tokenizer.setOutputCol('content_tokenized')

    df =(spark.read.format('com.databricks.spark.csv')
         .options(header=True, multiline=True, escape='"')
         .load('./news_sample.csv'))
    df = df.select('content', 'type')

    process = f.udf(preprocess, t.StringType())
    df = df.withColumn('content', process(f.col('content')))

    tokenized = tokenizer.transform(df)
    #to_string = f.udf(str, t.StringType())
    
    #tokenized = tokenized.withColumn('content_tokenized',
    #                                 to_string(f.col('content_tokenized')))
    tokenized = tokenized.drop("content")
   
    as_np = np.array(tokenized.collect())
    as_np.dump("./fakenews-numpy-dump.npx")

    #tokenized.write.format('com.databricks.spark.csv').save('./spark.out', header=True)
