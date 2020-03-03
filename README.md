# Korean-English Translator


## Objective

The objective for this project was to create a Neural Machine Translator that could translate English phrases to Korean.

You're probably wondering: Google Translate's a thing, why not just use that?

I have lived in South Korea for the past 9 years, and when I first moved there, Google Translate was not very good at translating Korean-English and vice-versa. Having found this out after several attempts at translation, I ended up using my trusty dictionary when I encountered phrases that I did not understand, looking up each word individually. Unfortunately, this distrust of Google Translate persists today even though it has vastly improved since, and I continue to work inefficiently with the dictionary. So, in order to make my life a little easier, I decided to try and make my own translator.


## English- Korean Translations are difficult

Here's a quick look at Korean vs English:

#### 안녕하세요. 재 이름은 김주경입니다.  ==  Hello. My name is Joo Kyung Kim

The first thing that you notice is that the alphabet is completely different. That's not the main issue when translating, though. One of the biggest issues is the grammar difference.

If we were to take a Korean simple sentence...

#### 톰은 한국어 공부해

...and translate it word for word into English, we would get this:

#### Tom Korean studies

Korean follows a Subject-Object-Verb grammar structure compared to English's Subject-Verb-Object structure. Seeing as how this is a simple example, it's easy to understand what I am trying to convey with this word-for-word translation. However, as the complexity of the sentence increases, the complexity of the translation also increases accordingly as seen below:


![grammer](https://github.com/virsagothethird/Korean-English-translator/blob/master/korean_english_grammar.jpg)


The use of honorifics is also highly important in Korean. Depending on who I speak to, I will adjust my speech accordingly even if I was conveying the exact same message. These intricacies can prove to be quite difficult to pick up for a machine algorithm. These are just a few of the reasons why students in Korea sometimes struggle when learning English.


## Initial EDA

I started with a dataset of a little over 3000 English-Korean sentence pairs from http://www.manythings.org/anki/. I further enriched my dataset by using a custom webscraper that scraped through thousands of KPOP song lyrics and lyric translations from https://colorcodedlyrics.com/ and obtained an addictional 95,000 English-Korean sentence pairs.

Seeing the recent rise in KPOP internationally, I reasoned that the quality of song lyric translations would have risen in proportion to it's popularity as many more record labels now release official translations to their songs.

This left us with a total dataset size of **98,161 sentence pairs** after cleaning with an English vocabulary size of **12,251 unique words** and a Korean vocabulary size of **58,663 unique words**.

Looking at the top 20 words in our English vocabulary:

![bar](
