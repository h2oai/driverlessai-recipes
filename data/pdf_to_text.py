"""Data recipe to transform pdf documents to text

This data recipe makes the following steps:
1. Reads pdf file
2. Parses each page into text blob
3. Concatenates page text blobs into single text
4. Attaches text column with parsed text

"""

import datatable as dt
import os
import numpy as np
import pandas as pd
from h2oaicore.data import BaseData
from typing import Union, List, Dict

_global_modules_needed_by_name = ["PyPDF2>=1.26.0"]  # Optional global package requirements, for multiple custom recipes in a file

import PyPDF2

PDF_COLNAME = 'file'
TXT_COLNAME = 'pdf_text_all_pages'
DATA_DIR = "data/"
PDF_DIR = "pdfs/"

class PDFtoTextData(BaseData):
    """
    Given column name that contains pdf file path opens and parses file into text
    and attaches text column to the dataset.
    """

    """Specify the python package dependencies (will be installed via pip install mypackage==1.3.37)"""
    # _modules_needed_by_name = ["PyPDF2>=1.26.0"]


    @staticmethod
    def create_data(X: dt.Frame = None) -> Union[
        str, List[str],
        dt.Frame, List[dt.Frame],
        np.ndarray, List[np.ndarray],
        pd.DataFrame, List[pd.DataFrame],
        Dict[str, str],  # {data set names : paths}
        Dict[str, dt.Frame],  # {data set names : dt frames}
        Dict[str, np.ndarray],  # {data set names : np arrays}
        Dict[str, pd.DataFrame],  # {data set names : pd frames}
    ]:

        if X is None:
            return []

        # Path to the directory with videos
        if not os.path.exists(DATA_DIR):
            return []
        files_dir = os.path.join(DATA_DIR, PDF_DIR)

        text_list = list(range(len(pdf_files)))
        if PDF_COLNAME in X.columns:
            pdf_files = X[PDF_COLNAME].to_list()[0]

            # Path to a .csv with labels. First column is video name, second column is label
            for file_name in pdf_files:
                try:
                    path_to_pdf = os.path.join(files_dir, file_name)
                    pdf_handle = open(path_to_pdf, 'rb')
                    pdfReader = PyPDF2.PdfFileReader(pdf_handle)
                    numOfPages = pdfReader.getNumPages()
                    for page in range(0, numOfPages):
                        pageObj = pdfReader.getPage(i)
                        pageTxt = pageObj.extractText()
                        cleanTxt = " ".join(pageTxt.split())
                        text_list[i] = cleanTxt
                finally:
                    close(pdf_handle)
            txtFrame = dt.Frame(text_list)
            txtFrame.names = TXT_COLNAME
            X.cbind(txtFrame)

        return X

