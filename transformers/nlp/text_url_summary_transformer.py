"""Transformer for creating text summary from URL"""
from h2oaicore.transformer_utils import CustomTransformer
import datatable as dt
import numpy as np

class TextURLSummaryTransformer(CustomTransformer):

    _numeric_output = False
    _modules_needed_by_name = ["gensim==3.8.0","beautifulsoup4==4.8.0"]  
    _display_name = 'URLSummaryTransformer'  
    _is_reproducible = False
    
    @staticmethod
    def get_default_properties():
        return dict(col_type="text", min_cols=1, max_cols=1, relative_importance=1)

    def fit_transform(self, X: dt.Frame, y: np.array = None):
        return self.transform(X)

    def transform(self, X: dt.Frame):

        from bs4 import BeautifulSoup
        from gensim.summarization.summarizer import summarize
        import requests

        XX = X.to_pandas().iloc[:, 0].values
        urls = XX
        summaries = []
        
       
        headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'}
        
        #Iterate through list of url and extract summary
        for url in urls:
            
            #URL cleaning
            if url is not None:
                url = url.strip('\'"')
            
            #Parse valid URL's
            try:

                #req = Request(url,headers=headers)
                page = requests.get(url,headers=headers,stream=True)
                soup = BeautifulSoup(page.content,'html.parser')
                #Join valid paragraphs in Beautiful soup output
                text = ' '.join(map(lambda p: p.text, soup.find_all('p')))
            
                text_summary = summarize(text)

            except:
                text_summary = '' 
                
            summaries.append(text_summary)
       
        ret = pd.DataFrame({'URLSummary':summaries})
    
        return ret

    
