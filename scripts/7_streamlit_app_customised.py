#streamlit run 7_streamlit_app_customised.py
import streamlit as st
import spacy
import spacy_streamlit
import pandas as pd
import altair as alt
from collections import Counter
spacy.prefer_gpu(gpu_id=0)

def load_df(person):
    return pd.read_pickle(f'../data/indiv_data/{person}.pk1')

def create_sent_plot (people, date):
    data = pd.DataFrame(columns = ['severity_h'])
    for person in people:
        data_person = load_df(person)
        data_person = data_person[data_person.year_month >= date]
        data_person['date'] = [pd.to_datetime(d_t) for d_t in data_person['date']]
        data_person.index = data_person['date']
        data_person = data_person[data_person.label_h != 0]
        data_person = data_person[['severity_h']].resample('M').mean().interpolate()
        data_person['rolling_score'] = data_person[['severity_h']].rolling(2).mean()
        data_person['rolling_score'] =  data_person['rolling_score'].round(2)
        data_person['entity'] = person
        data  = pd.concat([data, data_person], axis = 0)
    data = data.reset_index()
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',fields=['index'], empty='none')
    fig = alt.Chart(data).mark_line().encode(x = alt.X('index', axis = alt.Axis(format = '%b %y', title = 'Date')),
                                             y = alt.Y('rolling_score', axis = alt.Axis(title = 'Sentiment')),
                                             color = 'entity')
    selectors = alt.Chart(data).mark_point().encode(x='index',opacity=alt.value(0),).add_selection(nearest)
    points = fig.mark_point().encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
    # Draw text labels near the points, and highlight based on selection
    text = fig.mark_text(align='left', dx=5, dy=-5).encode(text=alt.condition(nearest, 'rolling_score', alt.value(' ')))
    # Draw a rule at the location of the selection
    rules = alt.Chart(data).mark_rule(color='gray').encode(x='index').transform_filter(nearest)
    plot = alt.layer(fig, selectors, points, rules, text)#.properties( height=400, width = 1000)
    return plot

def area_chart(person):
    p_df = load_df(person)
    p_df['label'] = [int(item) for item in p_df.label_h]
    p_df['perc'] = abs(p_df['label_h']/p_df.groupby('year_month')['label_h'].transform('count'))
    perc = p_df.groupby(['year_month', 'label_h'])['perc'].sum()
    perc = perc.unstack()
    perc.fillna(0, inplace = True)
    perc.columns = ['Negative', 'Neutral', 'Positive']
    perc['Neutral'] =  1 - (perc['Negative'] + perc['Positive'])

    perc = perc.reset_index()
    perc  = perc[['year_month','Negative', 'Positive']]
    perc_melt = pd.melt(perc, id_vars = ['year_month'], value_vars = ['Negative', 'Positive'], value_name = 'sent')
    c = alt.Chart(perc_melt).mark_area(opacity = 0.2).encode(x = alt.X('year_month', axis = alt.Axis(title = 'Date')),
                                                             y =alt.Y('sent', axis = alt.Axis(title = '% of Comments'), stack = None), color = 'variable')
    return c

def box_charts(people,date):
    data = pd.DataFrame(columns = ['entities','severity_h'])
    for person in people:
        data_person = load_df(person)
        data_person = data_person[data_person.year_month >= date]
        data_person = data_person[['entities','severity_h']]
        data  = pd.concat([data, data_person], axis = 0)
    data = data.reset_index()
    c = alt.Chart(data).mark_boxplot(extent = 'min-max').encode(x = alt.X('severity_h', axis = alt.Axis(title = 'Sentiment')),
                                                                y = alt.Y('entities', axis = alt.Axis(title = 'Entity'))
                                                                )
    return c
    #data = df[df.year_month >= date]
    #data.date = [pd.to_datetime(date) for date in data.date]
    #c = alt.Chart(data).mark_boxplot(extent = 'min-max').encode(x = 'polarity',y = 'entities')

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    nlp = spacy.load(model_path)
    return (nlp)

nlp = load_model('../model')

def word_freq(person, option ,model =nlp):
    #open_class = ['ADJ','ADV', 'INTJ', 'VERB', 'NOUN', 'PROPN']
    df = load_df(person)
    if option == "Negative":
        option = -1
    elif option == 'Neutral':
        option = 0
    elif option == 'Positive':
        option =1
    else: option = 2
    if option < 2:
        selection = option
        comments = [comment for comment in df[df.label_h==selection]['clean_comments']]
        #with model.select_pipes(enable=["parser", "tagger"]):
        docs = model.pipe(comments)
        words = [token.text for doc in docs for token in doc if token.text != '>' and token.is_stop == False and token.is_punct ==False]
        word_count = Counter(words)
        common_words = word_count.most_common(10)
        common_df = pd.DataFrame(data = common_words, columns = ['word', 'frequency'])

    else:
        comments = df['clean_comments']
        docs = model.pipe(comments)
        words = [token.text for doc in docs for token in doc if token.text != '>' and token.is_stop == False and token.is_punct ==False]
        word_count = Counter(words)
        common_words = word_count.most_common(10)
        common_df = pd.DataFrame(data = common_words, columns = ['word', 'frequency'])

    c = alt.Chart(common_df).mark_bar().encode(x = alt.X('frequency', sort = '-x'),
                                               y = alt.Y('word', sort = '-x'))
    return c


st.title('Sentiment Analysis Dashboard')

all_df = pd.read_pickle('../data/comment_truncated.pk1')
print(all_df.groupby(by = 'label_h').count())







unique_entities = pd.read_csv('../data/entity_list.csv')
sorted_dates = pd.read_csv('../data/date_list.csv')


with st.sidebar:
    optionmulti = st.multiselect('Select people/Organisation (single or multiple)',
                             unique_entities.comment,
                             default=['Lee Hsien Loong'])

    date_slider = st.select_slider('date_slider', options = sorted_dates.month)
    option1 = st.selectbox('Select a person for area chart', unique_entities.comment)
    option2 = st.selectbox('Select a person for frequency table', unique_entities.comment)
    filter_options = st.radio('Filter', ['Negative', 'Neutral', 'Positive', 'All'], index = 1)



st.altair_chart(create_sent_plot(people = optionmulti, date = date_slider).interactive(), use_container_width = True)
st.altair_chart(box_charts(people = optionmulti, date = date_slider), use_container_width = True)




with st.spinner(text="Legend, Wait for it..."):
    st.altair_chart(area_chart(person = option1), use_container_width=True)
    #except:
        #pass


with st.spinner(text="Legend, Wait for it..."):
    try:
        st.altair_chart(word_freq(person = option2, option = filter_options))
    except:
        print('error - freq table')
st.success('dary, Legendary!')



default_text = 'Lee Hsien Loong is the Prime Minister of Singapore'
text = st.text_area('Text to analyze', default_text, height = 100)


col1, col2 = st.columns(2)

#@st.cache(allow_output_mutation=True)
with col1:
    spacy_model = '../model'
    nlp = load_model(spacy_model)
    #doc = spacy_streamlit.process_text(spacy_model, text)
    doc = nlp(text)
    spacy_streamlit.visualize_ner(doc,
                              show_table = False,
                              title = 'names')
    st.text(f'analyzed using custom model')

#@st.cache(allow_output_mutation=True)
with col2:
    spacy_model = 'en_core_web_sm'
    nlp = load_model(spacy_model)
    doc = nlp(text)

    spacy_streamlit.visualize_ner(doc,
                                  show_table = False,
                                  title = 'names')
    st.text(f'analyzed using Spacy english model')
