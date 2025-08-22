from datetime import timedelta

import pandas as pd


class DataFrameIterator:
    def __init__(
        self,
        df: pd.DataFrame,
        size: str | timedelta = '1d',
        overlap: str | timedelta = '15min',
    ) -> None:
        size = pd.Timedelta(size).to_pytimedelta()
        overlap = pd.Timedelta(overlap).to_pytimedelta()

        self._index = 0
        self.df = df
        self.size = size
        self.overlap = overlap
        self.chunks = self._get_chunks(df, size, overlap)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self.chunks):
            chunk = self._load_chunk(self._index)

            self._index += 1
            return chunk

        else:
            raise StopIteration

    def _get_chunks(
        self,
        df: pd.DataFrame,
        size: timedelta,
        overlap: timedelta,
    ) -> pd.DataFrame:
        start = df.index.min()
        end = df.index.max()

        chunks = pd.date_range(start=start, end=end, freq=size, normalize=False).to_frame(index=False, name='start')
        chunks.index.name = 'chunk'
        chunks['end'] = chunks['start'] + size
        chunks.iat[0, 0] = start
        chunks.iat[-1, -1] = end

        chunks['start_overlap'] = chunks['start'] - overlap  # type: ignore
        chunks['end_overlap'] = chunks['end'] + overlap

        return chunks

    def _load_chunk(self, chunk: int) -> pd.DataFrame:
        if len(self.chunks) == 1:
            df = self.df.copy()
            df['overlap'] = False

            return df

        chunk_record = self.chunks.iloc[chunk]
        start, end = chunk_record['start'], chunk_record['end']
        start_overlap, end_overlap = chunk_record['start_overlap'], chunk_record['end_overlap']

        df = self.df  # type: pd.DataFrame
        df = df.loc[(df.index >= start_overlap) & (df.index < end_overlap)].copy()
        df['overlap'] = True
        df.loc[(df.index >= start) & (df.index < end), 'overlap'] = False

        return df
