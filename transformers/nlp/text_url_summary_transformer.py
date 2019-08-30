"""Extract text from URL and summarizes it"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np

class TextURLSummaryTransformer(CustomTransformer):

    _numeric_output = False
    _modules_needed_by_name = ["gensim==3.8.0","beautifulsoup4==4.8.0"]  
    _display_name = 'TextURLSummaryTransformer'  

    
    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    #Parses URLs and gets summary 
    def parse_url(self,url):
        
        from bs4 import BeautifulSoup
        from gensim.summarization.summarizer import summarize
        import requests

        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
        summaries = []

        #URL cleaning
        if url is not None:
            url = url.strip('\'"')
                
        try:
            
            page = requests.get(url,headers=headers,stream=True)

            soup = BeautifulSoup(page.content,"lxml")
            #print ('got soup')
            text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
            #print ('got text')
            text_summary = summarize(text)
            #print ('got summary')
        except: 
            text_summary = ''
                
        summaries.append(text_summary)
        
        return summaries
    
    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):

        import pandas as pd
        import multiprocessing as mp

        num_workers = mp.cpu_count()  

        XX = X.to_pandas().iloc[:, 0].values
        urls = XX
        summaries = []
        
        #Start parallel process with n jobs
        p = mp.Pool(num_workers)  
        #Call parse_url function with list of all urls
        all_summaries = p.map(self.parse_url, urls)
        p.terminate()
        p.join()
        
        #Flatten list of lists
        summaries = [ent for sublist in all_summaries for ent in sublist]
       
        ret = pd.DataFrame({'URLSummary':summaries})
    
        return ret

    
