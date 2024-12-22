# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

path = 'D:/My completed Projects/ML&DM Programming/ML&DM-Chapter2-film6(IMDB)'

r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv(path + 'u.data', sep='\t', names=r_cols, usecols=range(3))

m_cols = ['movie_id', 'title']
movies = pd.read_csv(path + 'u.item', sep='|', names=m_cols, usecols=range(2))


ratings_full = pd.merge(movies, ratings)

userMovies = ratings_full.pivot_table(index=['user_id'], columns=['title']
            , values='rating')


starWarsRating = userMovies['Star Wars (1977)']
similarMovies = userMovies.corrwith(starWarsRating)
similarMovies = similarMovies.dropna()
df = pd.DataFrame(similarMovies)
similarMovies.sort_values(axis=0, ascending=False, inplace=True)


movieStats = ratings_full.groupby('title').agg({'rating':[np.size, np.mean]})

popluarMovies = movieStats['rating']['size'] >= 100
statsSorted = movieStats[popluarMovies].sort_values([('rating', 'mean')], ascending=False)[:15]

df = movieStats[popluarMovies].join(pd.DataFrame(similarMovies, columns=['similarity']))

df.sort_values(['similarity'], ascending=False)[:15]
