"""Create titles and episodes datasets from IMDB tables"""

"""Create IMDb datasets for main titles and episodes. Main titles include info for both movies and series,
   while episodes contains series episodes details and ratings. Thus, there is one-to-many relationship
   between titles (one side) and episodes (many side which may be 0)
   Description: https://www.imdb.com/interfaces/
   Source: https://datasets.imdbws.com/
"""

from typing import Union, List, Dict
from h2oaicore.data import CustomData
import datatable as dt
import numpy as np
import pandas as pd
import os
from h2oaicore.systemutils_more import download
from h2oaicore.systemutils import config, user_dir


class IMDbTitleRatingsData(CustomData):
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
        # Download files
        # Location in DAI file system where we will save the data set
        temp_path = os.path.join(user_dir(), config.contrib_relative_directory)
        os.makedirs(temp_path, exist_ok=True)

        # URL of desired data, this comes from the City of Seattle
        link_basics = "https://datasets.imdbws.com/title.basics.tsv.gz"
        link_ratings = "https://datasets.imdbws.com/title.ratings.tsv.gz"
        link_episodes = "https://datasets.imdbws.com/title.episode.tsv.gz"

        # Download the files
        file_basics = download(link_basics, dest_path=temp_path)
        file_ratings = download(link_ratings, dest_path=temp_path)
        file_episodes = download(link_episodes, dest_path=temp_path)

        # get COVID19 new cases data from Our World in Data github
        basics = dt.fread(file_basics, fill=True)
        ratings = dt.fread(file_ratings, fill=True)
        episodes = dt.fread(file_episodes, na_strings=['\\N'], fill=True)

        # remove files
        os.remove(file_basics)
        os.remove(file_ratings)
        os.remove(file_episodes)

        # Create Title with Ratings dataset
        # join titles with non-null ratings
        ratings = ratings[~dt.isna(dt.f.averageRating), :]
        ratings.key = "tconst"
        basics_ratings = basics[:, :, dt.join(ratings)]

        # Create Episodes dataset
        episodes = episodes[~dt.isna(dt.f.seasonNumber) & ~dt.isna(dt.f.episodeNumber), :]
        episode_ratings = episodes[:, :, dt.join(ratings)]
        episode_ratings.names = {'tconst': 'episodeTconst', 'parentTconst': 'tconst',
                                 'averageRating': 'episodeAverageRating', 'numVotes': 'episodeNumVotes'}
        basics_ratings.key = 'tconst'
        title_episode_ratings = episode_ratings[:, :, dt.join(basics_ratings)]

        # enumerate series episodes from 1 to N
        title_episode_ratings = title_episode_ratings[:, :, dt.sort(dt.f.tconst, dt.f.seasonNumber, dt.f.episodeNumber)]
        result = title_episode_ratings[:, dt.count(), dt.by(dt.f.tconst)][:, 'count'].to_list()
        from itertools import chain
        cumcount = chain.from_iterable([i + 1 for i in range(n)] for n in result[0])
        title_episode_ratings['episodeSequence'] = dt.Frame(tuple(cumcount))

        # return datasets
        return {f"imdb_title_ratings": basics_ratings,
                f"imdb_episode_ratings": title_episode_ratings}
