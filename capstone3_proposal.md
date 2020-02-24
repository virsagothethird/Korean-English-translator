## A Korean-English Translator
### Description
This project will aim to create a neural machine translator using a sequence to sequence model from Tensorflow to translate Korean sentences to English, and vice versa. Sentences will be input into an Encoder model which would then be passed into a Decoder model to output the translated sentence.

The steps to achieve this are documented in detail at:

https://www.tensorflow.org/tutorials/text/nmt_with_attention

### Approach
Currently, Google Translate does an excellent job at translating Korean to English, with Naver's Papago (Korea's Google Translate equivalent) said to be even more accurate. Trying to compete with those in a week long project is impossible, but I am curious how close I can get with limited time and resources.

The dataset currently has approximately 1000 English/Korean sentences. As a large amount of tranlated text will be required, a possible source of dataset enrichment for this project would be song lyric translations. As K-Pop has risen in popularity internationally in recent years, I believe that song translations, official or otherwise, have reached a point where they can be reliably used for this project.

A potential problem for this project would be with the dataset. Ideally, enrichment with song lyrics would allow for more sentences to train on, which should result in a better model. A problem with using Korean song lyrics is that English words can be mixed in with the Korean lyrics. A method to filter out those sentences would be required.

The work will be presented on a flask app that will allow an input sentence and output the translation. Along with the translated sentence, the model will display the attention plot that shows which parts of the input sentence has the model's 'attention' when tanslating.