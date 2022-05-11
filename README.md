# Named Entity Recognition and Entity Linking for Sentiment Analysis


_______________________
SpaCy/ NLP / NER / NEL / Huggingface / Sentiment Analysis / Streamlit
_______________________

## Executive Summary

Sentiment analysis is Natural Language Processing ('NLP') problem where bodies of texts (eg. tweets, comments, op-eds, etc.) 
are identified and classified as either neutral, negative or positive to determine the writers' views towards a particular product, person, organization, etc.. 

Social media discussions often involve a variety of topics and individuals. 
As such, sentiment analysis alone is often insufficient to generate insights that are specific enough for response,
and deeper analysis is required to ensure that analysis only includes data relevant to the analysis.

Named Entity Recognition ('NER') is the process of inspecting a body of texts to identify any entities within a text.

Named Entity linking ('NEL') is the task of linking a unique real world identity to entities (such as famous individuals, companies and organisations) identified in a given text.

By combining NER and NEL with sentiment analysis we can determine sentiment expressed about specified entities within an online community.

## Introduction

Social media platforms like Reddit, Facebook and Twitter are sites where users are encouraged to express their views on a large variety of topics.

A large number of users often express their views on various politicians, government bodies and policies online.

With sentiment analysis, organisations and political entities can monitor online conversations to gauge public opinion on government policies or
approval ratings for politicians faster than traditional methods like surveys or opinion polls.

For this project we will analyse comments taken from Reddit - Singapore to gauge sentiment towards Singaporean politicians and government bodies.



## Part 1 - Data

Using PRAW and Pushshift API, submission titles and comments made between Nov 2020 to March 2022 were extracted from scrapped from the r/Singapore subreddit.


Sample post title/comments

| Title                                                                             	|
|-----------------------------------------------------------------------------------	|
| ‘Never fudge or sugarcoat, never hide’: Josephine Teo on COVID public information 	|


| Comment                                                     	   |
|-----------------------------------------------------------------|
| Well Jo Teo cried and got 65% of the votes from her contituency |
| That's weird I don't remember JoTeo taking in MND portfolio 	   |


## Part 2 - Processing Pipeline

Due to the nature of reddit submissions titles and comments, there is a mix in the 'formality' in the text, with 
most submission titles copied directly from the titles of news articles while comments are more casual. Due to this mix we run into 3 issues:

1) Pre-trained NER models can fail to identify entities that are outside is vocabulary, 
2) Within text entities can be referred to in a variety of ways.

   - Conferencing: Separate mentions referencing the same real-world entity (eg. JoTeo, Jo Teo and Josephine Teo)
   - Anaphora: When a term (anaphor) refers to another term(antecedent) with interpretation of the anaphor determined by the antecedent 
   (eg. Well Jo Teo(antecedent) cried and got 65% of the votes from her(anaphor) contituency)
   
3) Comments or titles can often reference multiple entities in the text.



###<i>Problem 1 and 2</i>


![example-1 standard NER](./images/example_standard_NER.jpg 'standard example - JT')

Using SpaCy's pretrained english_core_web_sm model, we can see the while the NER component can pick up the terms Jo Teo and JoTeo
as entities it isn't able to identify the term 'Josephine Teo' as an entity. In addition, the model only identifies that the 2 terms ('Jo Teo', 'JoTeo) are entities but cannot identify that they are the same person.


To solve problem 1 we can train a custom NER model that is more suited to identify entities from Singapore.

For problem 2 we can incorporate an additional NEL model on-top of the NER model. The NEL model takes in any entity identified by our NER model
and from a pre-defined knowledge base, generate a list of candidates that the identified entity could be referencing.

For example, given the following comment:

![example-2 NEL](./images/NEL_process_example.jpg 'nel process example')

The NER model will identified that the word 'Louis' is an entity, the NEL then identified possible individuals that the word is referencing
and makes a selection based on text in the comment.

![example-2 NEL](./images/NEL_process_example2.jpg 'nel process example')

The NEL model also serves to disambiguate when multiple entities share the same name as seen in the example above, when the model selects Louis Chua instead of Louis Ng.

While the NEl model can address coreferencing in our specific use case, anaphora is not addressed and we will leave it for future work.

### <i>Problem 3</i>

Lastly for problem 3, some text (especially longer texts) may contain references to multiple entities in a text it is difficult 
to assess the sentiment around an entity since a comment may contain multiple sentiments towards different entities.

In the example below, we can see that in the first sentence, the author is praising Mothership, while in the second part,
the author is stating that OYK should thank Mothership[^1]. 

![example-1b sentiment parse](./images/ms_oyk_example.png 'me_example')

[^1]: post_id: pxyk9b

As a whole, the chosen classifier (see part 4) would label this comment as a positive comment. However, the positive sentiment
in this comment is specifically directed at Mothership so assigning both entities the same score would not be accurate. 

Spacy has a dependency parser to identify the grammatical relations between the words in a text.

![example-1c sentiment parse](./images/parse_example.png 'me_example')

Words that are connected by an arc are referred to as a 'Parent' and a 'Child', with the arc representing the syntatic relation between a Parent, Child pair.
As each relation forms a tree each word can be traced back to a single 'Head'. By iterating through the arcs connected to any tokens that contain an entity,
we can determine the Head of the relevant dependency tree and analyse sentiment in isolation of other unrelated trees. Using the above example,
starting with the token 'Mothership' it has a Parent 'to', which in turn has a Parent 'kudos', 'Kudos' is has no Parents and is the Head of the tree.
Combining the three tokens together we get the phrase 'Kudos to Mothership' that we use to determine sentiment towards the entity 'Mothership'.



## Part 3 - Training the NER and NEL models

The NER and NEL models will be trained using the SpaCy library. 
First to train the NER model we will need annotated data that contains the following:
- the text of the comment,
- entity label/type,
- named entities in the text (tokens); and
- the start and end character of the token (span).

| Text                                                                                                                           	| Label                                                                  	           |
|--------------------------------------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------|
| Singapore Politics is a simple game; several parties chase votes for a few weeks every 5 years, and at the PAP wins.           	| Start Character: 108 End Character: 111 Label: ORG Entity: People's Action Party 	 |
| if WP supporters adhered to safe distancing rules and were able to be more organised, it would have been super impressive haha 	| Start Character: 3 End Character: 5 Label: ORG Entity: Workers' Party            	 |

The NEL model also requires a Knowledge Base that contains a list of candidates, possible aliases, associated entities and entity vector.
From the KB the NEL model takes in the entities identified by the NER model and generates possible candidates given a textual mention of an entity. 


To generate this data we can utilise the following process:

1) Starting with the pretrained english_core_web_sm model as a base, 
2) Using SpaCy's rules based Entity-Ruler component we can add name variations that are out of the models vocabulary (eg. initials, abbreviations or aliases) 
3) Use this model to run through a portion of our dataset and annotate the data,
4) Review the annotated data and relabel/ drop incorrectly annotated examples.


Once the training/validation data is generated we can train the models. We can utilise the configuration system from Thinc, which allows us to train a model by simply creating a config.cfg file and running the spacy train command in the command line.

![table-1 model_performance](./images/model-performance.jpg 'model performance')

Using the same example from earlier, we can see our custom NER model can now identify all relevant entities in the texts, 
and the NEL model is able to determine that all 3 entities identified are referencing the same person.

![example-3 custom NER](./images/example_custom_NER.jpg 'custom example - JT')



## Part 4 - Sentiment Analysis

SpaCy does not have models for sentiment analysis, instead we will use pre-trained models available on huggingface or the Vader library.

<u>Performance: Vader</u>

![example-4 vader](./images/vader_class.jpg 'vader performance')
![example-4 vader](./images/vader_cm.jpg 'vader CM')


<u>Performance: Huggingface - Cardiffnlp/twitter-roberta-base-sentiment </u>

![example-4 hugg](./images/hugg_class.jpg 'vader performance')
![example-4 hugg](./images/hugg_cm.jpg 'vader CM')


For sentiment analysis, huggingface outshines Vader across all metrics and we will use it to classify and label our comments.


## Part 5 - Sentiment Dashboard

The data is summarised and visualised within a dashboard and deployed on streamlit to give users tools to explore the data and how perception towards individuals changed over time.

![example-5 dashboard](./images/dashboard.jpg 'dashboard')


## Conclusion

Although the current pipeline is able to accurately identify named entities and classify sentiment, there are still several limitations in the analysis.

Our analysis uses data from the Singapore subreddit, a safe assumption would be that users are predominantly Singaporean and will use Singlish.
However our chosen pre-trained sentiment analysis model was trained on tweets written in english, as such the model might not be able to accurately classify comments
that use Singlish terms.

In addition, online communities are known to develop new words or new definitions of existing words that are outside the vocabulary of the pre-trained model. 
For example, in online discussions the term '160' is used somewhat frequently to describe or make reference to journalism in Singapore.
In this context, the term '160' might have negative connotations to it which the model does not account for, resulting in the sentiment classification model to make predictions that are more positive/negative when compared to how a human would interpret it.

![example-7 160](./images/160_example.jpg '160')

Lastly, while the pipeline can determine the target entity in a given comment, 
the pipeline lacks topic classification component to determine what aspect of the entity the commenter is reacting to. 

## Future Work

As a start, incorporating an topic classification component into the pipeline would be the most impactful change,
allowing users to not only identify how online communities are reacting to individuals and organisations, 
but the specific aspect that people are reacting to (eg. personality, proposed ideas, poor communication, etc.).
With such data, users can build strategies to tackle specific issues.

The sentiment classifier should also be improved by expand its vocabulary to incorporate Singlish outside its vocabulary.

