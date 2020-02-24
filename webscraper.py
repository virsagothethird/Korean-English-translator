import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random



def get_song_urls_list(url):
    song_urls_list = []

    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')

    for a in soup.find_all('a', {'rel': 'bookmark'}):
        song_urls_list.append(a['href'])
    return song_urls_list


def scrape_first_page(start_url, empty_df):
    
    start_urls_list = get_song_urls_list(start_url)
    
    for url in start_urls_list:
        
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        tags = soup.find_all(['td','p'])
        td_tags = soup.find_all(['td'])
        p_tags = soup.find_all(['p'])
        
        kor = pd.DataFrame()
        eng = pd.DataFrame()

        if (td_tags[-1].text=='N/A') | (td_tags[-1].text=='soon') | (len(tags)<25) | (p_tags[-3].text[0:6]!='Korean'):
            time.sleep(random.randint(5,8))
            continue
        else:
            c = len(tags) - 3
            d = c - 7
            e = int(d/3)
            a = e + 7
            b = e + a

            for i in range(a,b):
                kor_lines = tags[i]
                kor_lines_series = pd.Series(kor_lines.text.split('\n'))
                kor = pd.concat([kor, kor_lines_series])
                kor = kor[kor[0]!=""]
            kor = kor.rename(columns={0:'kor'}).reset_index().drop(columns='index')

            for i in range(b,c):
                eng_lines = tags[i]
                eng_lines_series = pd.Series(eng_lines.text.split('\n'))
                eng = pd.concat([eng, eng_lines_series])
                eng = eng[eng[0]!=""]
            eng = eng.rename(columns={0:'eng'}).reset_index().drop(columns='index')
            
            if eng.shape != kor.shape:
                continue
            else:
                concatted = pd.concat([eng, kor],axis=1)
                concatted = concatted.drop_duplicates()
                concatted = concatted.query("eng != kor")

                empty_df = pd.concat([empty_df,concatted])

                time.sleep(random.randint(6,9))
    return empty_df


def scrape_it(start_url, empty_df, end):
    
    z=1
    # print(f'Starting page: {z}')
    # df = scrape_first_page(start_url, empty_df)
    df = pd.DataFrame(columns = ['eng','kor'])
    for i in range(101,end+1):
        
        # z+=1
        print(f'Starting page: {i}')
        next_url = f'https://colorcodedlyrics.com/category/krn/page/{i}'

        urls_list = get_song_urls_list(next_url)
        if (i==100) | (i==200) | (i==300) | (i==400) | (i==500) | (i==600) | (i==700):
            df.to_csv('lyrics_save{}.csv'.format(i), sep='\t')
            df.to_csv('lyrics_save{}.txt'.format(i), sep='\t')
            print(f'saved up tp page: {i}')

        for url in urls_list:
            if url == 'https://colorcodedlyrics.com/2019/11/sulli-amp-hara':
                continue
            r = requests.get(url)
            soup = BeautifulSoup(r.text, 'html.parser')
            tags = soup.find_all(['td','p'])
            td_tags = soup.find_all(['td'])
            p_tags = soup.find_all(['p'])
            
            kor = pd.DataFrame()
            eng = pd.DataFrame()

            if (len(td_tags)<5) | (len(tags)<25) | (len(p_tags)<7):
                continue
            if (td_tags[-1].text=='N/A') | (td_tags[-1].text=='soon') | (td_tags[-1].text=='Coming Soon!') | (p_tags[-3].text[0:6]!='Korean'):
                time.sleep(random.randint(5,8))
                continue
            else:
                c = len(tags) - 3
                d = c - 7
                e = int(d/3)
                a = e + 7
                b = e + a

                for i in range(a,b):
                    kor_lines = tags[i]
                    kor_lines_series = pd.Series(kor_lines.text.split('\n'))
                    kor = pd.concat([kor, kor_lines_series])
                    kor = kor[kor[0]!=""]
                kor = kor.rename(columns={0:'kor'}).reset_index().drop(columns='index')

                for i in range(b,c):
                    eng_lines = tags[i]
                    eng_lines_series = pd.Series(eng_lines.text.split('\n'))
                    eng = pd.concat([eng, eng_lines_series])
                    eng = eng[eng[0]!=""]
                eng = eng.rename(columns={0:'eng'}).reset_index().drop(columns='index')

                if eng.shape != kor.shape:
                    continue

                concatted = pd.concat([eng, kor],axis=1)
                concatted = concatted.drop_duplicates()
                concatted = concatted.query("eng != kor")

                df = pd.concat([df,concatted])

                time.sleep(random.randint(5,8))

        
    return df

if __name__ == "__main__":
    start_url = 'https://colorcodedlyrics.com/category/krn'

    empty = pd.DataFrame(columns = ['eng','kor'])

    df = scrape_it(start_url=start_url, empty_df=empty, end=800)

    df.to_csv('lyrics.csv',sep='\t')
    df.to_csv('lyrics.txt',sep='\t')